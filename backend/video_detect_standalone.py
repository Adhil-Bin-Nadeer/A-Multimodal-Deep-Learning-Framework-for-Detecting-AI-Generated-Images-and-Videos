import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

try:
    import onnx
    from onnx2pytorch import ConvertModel
except Exception as exc:  # pragma: no cover - optional runtime dependency
    onnx = None
    ConvertModel = None
    _DEDICATED_MODEL_IMPORT_ERROR = str(exc)
else:
    _DEDICATED_MODEL_IMPORT_ERROR = None


BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(BACKEND_DIR, "checkpoints")
ONNX_PATH = os.path.join(CHECKPOINTS_DIR, "efficientnet.onnx")
PYTORCH_PATH = os.path.join(CHECKPOINTS_DIR, "model.pth")
DEFAULT_SAMPLE_FRAMES = 6
DEDICATED_VIDEO_THRESHOLD = 0.60
FALLBACK_VIDEO_THRESHOLD = 0.65
FALLBACK_FRAME_THRESHOLD = 0.68

_DEDICATED_MODEL = None
_DEDICATED_MODEL_ERROR = None


def _build_timestamp(frame_index: int, fps: float) -> str:
    if fps <= 0:
        return f"frame {frame_index}"

    seconds = frame_index / fps
    minutes = int(seconds // 60)
    remainder = seconds - (minutes * 60)
    return f"{minutes:02d}:{remainder:05.2f}"


def _decision_confidence(ai_probability: float, ai_threshold: float, is_ai: bool) -> float:
    if is_ai:
        relative_margin = (ai_probability - ai_threshold) / max(1.0 - ai_threshold, 1e-6)
    else:
        relative_margin = (ai_threshold - ai_probability) / max(ai_threshold, 1e-6)

    relative_margin = float(max(0.0, min(1.0, relative_margin)))
    return 0.5 + (0.5 * relative_margin)


def _get_frame_indices(frame_count: int, n_frames: int) -> List[int]:
    if frame_count <= 0:
        return []

    sample_count = max(1, min(frame_count, n_frames))
    return sorted({int(index) for index in np.linspace(0, frame_count - 1, sample_count)})


def _sample_video_frames(input_video: str, n_frames: int = DEFAULT_SAMPLE_FRAMES) -> Dict[str, object]:
    capture = cv2.VideoCapture(input_video)
    if not capture.isOpened():
        raise ValueError("Unable to open the uploaded video.")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    target_indices = set(_get_frame_indices(frame_count, n_frames))

    frames = []
    frame_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        if not target_indices or frame_index in target_indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(
                {
                    "frame_index": frame_index,
                    "timestamp": _build_timestamp(frame_index, fps),
                    "image": frame_rgb,
                }
            )

            if target_indices and len(frames) >= len(target_indices):
                break
            if not target_indices and len(frames) >= n_frames:
                break

        frame_index += 1

    capture.release()

    if not frames:
        raise ValueError("No readable frames were extracted from the uploaded video.")

    return {
        "frames": frames,
        "fps": fps,
        "frame_count": frame_count if frame_count > 0 else len(frames),
    }


def _preprocess_frame_for_checkpoint_model(frame_rgb: np.ndarray) -> torch.Tensor:
    frame_rgb = frame_rgb.astype(np.float32) / 255.0
    frame_rgb = cv2.resize(frame_rgb, (256, 256))
    return torch.unsqueeze(torch.tensor(frame_rgb), dim=0)


def _checkpoint_model_setup_error() -> Optional[str]:
    if _DEDICATED_MODEL_IMPORT_ERROR:
        return f"Dedicated video model dependencies are unavailable: {_DEDICATED_MODEL_IMPORT_ERROR}"

    missing_paths = [path for path in (ONNX_PATH, PYTORCH_PATH) if not os.path.exists(path)]
    if missing_paths:
        missing_str = ", ".join(missing_paths)
        return (
            "Dedicated video checkpoints are missing. "
            f"Expected files: {missing_str}"
        )

    return None


def _load_checkpoint_model():
    global _DEDICATED_MODEL, _DEDICATED_MODEL_ERROR

    if _DEDICATED_MODEL is not None:
        return _DEDICATED_MODEL

    setup_error = _checkpoint_model_setup_error()
    if setup_error:
        _DEDICATED_MODEL_ERROR = setup_error
        raise RuntimeError(setup_error)

    try:
        onnx_model = onnx.load(ONNX_PATH)
        pytorch_model = ConvertModel(onnx_model)
        checkpoint = torch.load(PYTORCH_PATH, map_location=torch.device("cpu"))
        state_dict = checkpoint.get("rgb_encoder", checkpoint)
        pytorch_model.load_state_dict(state_dict, strict=True)
        pytorch_model.eval()
    except Exception as exc:
        _DEDICATED_MODEL_ERROR = str(exc)
        raise RuntimeError(f"Dedicated video model failed to load: {exc}") from exc

    _DEDICATED_MODEL = pytorch_model
    _DEDICATED_MODEL_ERROR = None
    return _DEDICATED_MODEL


def _predict_with_checkpoint_model(input_video: str) -> Dict[str, object]:
    model = _load_checkpoint_model()
    sampled = _sample_video_frames(input_video, n_frames=3)
    frames = sampled["frames"]

    real_scores = []
    fake_scores = []

    with torch.no_grad():
        for frame_info in frames:
            face = _preprocess_frame_for_checkpoint_model(frame_info["image"])
            logits = model.forward(face)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().detach().numpy()
            real_scores.append(float(probabilities[0]))
            fake_scores.append(float(probabilities[1]))

    real_mean = float(np.mean(real_scores))
    fake_mean = float(np.mean(fake_scores))
    ai_probability = max(0.0, min(1.0, fake_mean))
    label = "FAKE" if ai_probability >= DEDICATED_VIDEO_THRESHOLD else "REAL"
    confidence = _decision_confidence(ai_probability, DEDICATED_VIDEO_THRESHOLD, label == "FAKE")

    result_text = (
        f"The video is {label}. Confidence score: {confidence * 100:.1f}%"
    )

    return {
        "result": result_text,
        "label": label,
        "confidence": confidence * 100.0,
        "ai_probability": ai_probability * 100.0,
        "source": "Video AI Detector",
        "explainability": {
            "metrics": {
                "frames_analyzed": len(frames),
                "average_ai_probability": ai_probability * 100.0,
                "peak_ai_probability": ai_probability * 100.0,
                "flagged_frames": len(frames) if ai_probability >= DEDICATED_VIDEO_THRESHOLD else 0,
            },
            "artifacts": [
                f"Frames analyzed: {len(frames)}",
            ]
        },
    }


def _get_frame_predictor(predictor=None):
    if predictor is not None:
        return predictor

    from combine_model import AIEnsemblePredictor

    return AIEnsemblePredictor()


def _predict_with_frame_fallback(input_video: str, predictor=None) -> Dict[str, object]:
    sampled = _sample_video_frames(input_video, n_frames=DEFAULT_SAMPLE_FRAMES)
    frames = sampled["frames"]
    predictor = _get_frame_predictor(predictor)

    frame_scores = []
    for frame_info in frames:
        try:
            pil_image = Image.fromarray(frame_info["image"]).convert("RGB")
            ai_probability = float(predictor.predict_proba_from_pil(pil_image))
        except Exception:
            continue

        frame_scores.append(
            {
                "frame_index": frame_info["frame_index"],
                "timestamp": frame_info["timestamp"],
                "ai_probability": ai_probability,
            }
        )

    if not frame_scores:
        raise ValueError("Video fallback analysis could not score any frames.")

    probabilities = np.array([frame["ai_probability"] for frame in frame_scores], dtype=np.float32)
    mean_probability = float(np.mean(probabilities))
    peak_probability = float(np.max(probabilities))
    flagged_frames = int(np.sum(probabilities >= FALLBACK_FRAME_THRESHOLD))
    final_ai_probability = float(np.clip((0.7 * mean_probability) + (0.3 * peak_probability), 0.0, 1.0))

    label = "FAKE" if final_ai_probability >= FALLBACK_VIDEO_THRESHOLD else "REAL"
    confidence = _decision_confidence(final_ai_probability, FALLBACK_VIDEO_THRESHOLD, label == "FAKE")

    result_text = (
        f"The video is {label}. Confidence score: {confidence * 100:.1f}%"
    )

    return {
        "result": result_text,
        "label": label,
        "confidence": confidence * 100.0,
        "ai_probability": final_ai_probability * 100.0,
        "source": "Video AI Detector",
        "explainability": {
            "metrics": {
                "frames_analyzed": len(frame_scores),
                "average_ai_probability": mean_probability * 100.0,
                "peak_ai_probability": peak_probability * 100.0,
                "flagged_frames": flagged_frames,
            },
            "artifacts": [
                f"Frames analyzed: {len(frame_scores)} of {sampled['frame_count']}",
                f"Final video AI score: {final_ai_probability * 100:.1f}%",
            ],
            "frames": [
                {
                    "frame_index": frame["frame_index"],
                    "timestamp": frame["timestamp"],
                    "ai_probability": frame["ai_probability"] * 100.0,
                }
                for frame in frame_scores
            ],
        },
    }


def deepfakes_video_predict(input_video: str, predictor=None) -> Dict[str, object]:
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Video file not found at {input_video}")

    try:
        return _predict_with_checkpoint_model(input_video)
    except Exception:
        pass

    fallback_result = _predict_with_frame_fallback(input_video, predictor=predictor)
    return fallback_result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python backend/video_detect_standalone.py <video_path>")
        raise SystemExit(1)

    outcome = deepfakes_video_predict(sys.argv[1])
    print(outcome["result"])

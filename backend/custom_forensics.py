import os
import math
from typing import Dict, List, Tuple, Union

import numpy as np
from PIL import Image

try:
    import mediapipe as mp
except Exception as exc:
    mp = None
    mp_hands = None
    mp_pose = None
    _MEDIAPIPE_ERROR = str(exc)
else:
    mp_hands = None
    mp_pose = None
    _MEDIAPIPE_ERROR = None
    try:
        mp_hands = mp.solutions.hands
        mp_pose = mp.solutions.pose
    except Exception as exc:
        _MEDIAPIPE_ERROR = str(exc)

try:
    from skimage.measure import shannon_entropy
except Exception:
    shannon_entropy = None


CUSTOM_WEIGHTS = {
    "resnet": 0.50,
    "frequency": 0.20,
    "anatomy": 0.20,
    "entropy": 0.10,
}

FORENSIC_SUPPORT_THRESHOLD = 0.60
FORENSIC_NEGATIVE_THRESHOLD = 0.15

_POSE = None
_HANDS = None
_BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TASKS_VISION = getattr(getattr(mp, "tasks", None), "vision", None) if mp is not None else None
_BASE_OPTIONS = getattr(getattr(mp, "tasks", None), "BaseOptions", None) if mp is not None else None
_POSE_TASK_MODEL_PATH = None
_HAND_TASK_MODEL_PATH = None
_POSE_ENUM = None


def _resolve_task_model_path(env_var: str, default_name: str) -> str:
    candidates = []
    env_path = os.environ.get(env_var)
    if env_path:
        candidates.append(env_path)

    candidates.extend([
        os.path.join(_BASE_DIR, "model_output", "mediapipe", default_name),
        os.path.join(_BASE_DIR, "artifacts", "mediapipe", default_name),
    ])

    for path in candidates:
        if path and os.path.exists(path):
            return path
    return ""


def _can_use_mediapipe_tasks() -> bool:
    return _TASKS_VISION is not None and _BASE_OPTIONS is not None


def _ensure_tasks_model_paths() -> None:
    global _POSE_TASK_MODEL_PATH, _HAND_TASK_MODEL_PATH
    if _POSE_TASK_MODEL_PATH is None:
        _POSE_TASK_MODEL_PATH = _resolve_task_model_path("MEDIAPIPE_POSE_MODEL_PATH", "pose_landmarker.task")
    if _HAND_TASK_MODEL_PATH is None:
        _HAND_TASK_MODEL_PATH = _resolve_task_model_path("MEDIAPIPE_HAND_MODEL_PATH", "hand_landmarker.task")


def _mediapipe_unavailable_message() -> str:
    if mp_pose is not None and mp_hands is not None:
        return ""

    if _can_use_mediapipe_tasks():
        _ensure_tasks_model_paths()
        missing = []
        if not _POSE_TASK_MODEL_PATH:
            missing.append("pose_landmarker.task")
        if not _HAND_TASK_MODEL_PATH:
            missing.append("hand_landmarker.task")

        if missing:
            return (
                "MediaPipe uses the newer Tasks API in this environment. "
                f"Add {', '.join(missing)} under model_output/mediapipe/ "
                "or set MEDIAPIPE_POSE_MODEL_PATH / MEDIAPIPE_HAND_MODEL_PATH."
            )

    if _MEDIAPIPE_ERROR:
        return f"MediaPipe import is incomplete: {_MEDIAPIPE_ERROR}"

    return "MediaPipe anatomy validation is unavailable in this environment."


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _as_rgb_array(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))

    array = np.asarray(image)
    if array.ndim == 2:
        return np.stack([array] * 3, axis=-1)
    if array.ndim == 3 and array.shape[2] >= 3:
        return array[:, :, :3]
    raise ValueError("Unsupported image format for forensic analysis")


def _resize_with_pil(image: np.ndarray, max_side: int = 512) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image

    scale = max_side / float(longest)
    new_size = (max(32, int(round(width * scale))), max(32, int(round(height * scale))))
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    return np.array(pil_image.resize(new_size, _BILINEAR))


def _neighbor_peak_mask(values: np.ndarray) -> np.ndarray:
    peak_mask = np.ones_like(values, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(values, dy, axis=0), dx, axis=1)
            peak_mask &= values >= shifted
    return peak_mask


def analyze_frequency_artifacts(image: Union[Image.Image, np.ndarray]) -> Dict:
    rgb = _resize_with_pil(_as_rgb_array(image), max_side=512)
    gray = np.mean(rgb.astype(np.float32), axis=2)
    gray = gray - float(np.mean(gray))

    height, width = gray.shape
    window = np.outer(np.hanning(height), np.hanning(width)).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(gray * window))
    power = np.abs(spectrum) ** 2
    power_density = power / (np.sum(power) + 1e-12)
    log_psd = np.log1p(power_density * power_density.size)

    cy, cx = height // 2, width // 2
    yy, xx = np.indices((height, width))
    norm_y = (yy - cy) / max(float(cy), 1.0)
    norm_x = (xx - cx) / max(float(cx), 1.0)
    radius = np.sqrt(norm_y ** 2 + norm_x ** 2)

    high_mask = (radius > 0.22) & (radius < 0.88)
    if not np.any(high_mask):
        return {
            "available": True,
            "ai_score": 0.0,
            "detected": False,
            "message": "Frequency map too small for checkerboard analysis.",
            "symmetric_peak_pairs": 0,
            "grid_energy_ratio": 1.0,
            "peak_prominence": 0.0,
        }

    high_values = log_psd[high_mask]
    baseline = float(np.median(high_values))
    high_std = float(np.std(high_values))
    if high_std < 0.015:
        return {
            "available": True,
            "ai_score": 0.0,
            "detected": False,
            "message": "Frequency response is too flat for a reliable checkerboard analysis.",
            "symmetric_peak_pairs": 0,
            "grid_energy_ratio": 1.0,
            "peak_prominence": 0.0,
        }

    threshold = float(np.percentile(high_values, 99.8))
    min_peak_contrast = max(0.05, 0.5 * high_std)
    peak_mask = (
        high_mask &
        (log_psd >= threshold) &
        ((log_psd - baseline) >= min_peak_contrast) &
        _neighbor_peak_mask(log_psd)
    )

    candidate_coords = np.argwhere(peak_mask)
    symmetric_pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    pair_prominences: List[float] = []
    pair_alignment_scores: List[float] = []

    for y, x in candidate_coords:
        my = (2 * cy) - int(y)
        mx = (2 * cx) - int(x)
        if not (0 <= my < height and 0 <= mx < width):
            continue
        if not peak_mask[my, mx]:
            continue
        if (y > my) or (y == my and x >= mx):
            continue

        pair_power = min(float(log_psd[y, x]), float(log_psd[my, mx]))
        pair_prominences.append(max(0.0, pair_power - baseline))
        symmetric_pairs.append(((int(y), int(x)), (int(my), int(mx))))

        axis_x = abs(abs((x - cx) / max(float(cx), 1.0)) - 0.25)
        axis_x = min(axis_x, abs(abs((x - cx) / max(float(cx), 1.0)) - 0.50))
        axis_x = min(axis_x, abs(abs((x - cx) / max(float(cx), 1.0)) - 0.75))
        axis_y = abs(abs((y - cy) / max(float(cy), 1.0)) - 0.25)
        axis_y = min(axis_y, abs(abs((y - cy) / max(float(cy), 1.0)) - 0.50))
        axis_y = min(axis_y, abs(abs((y - cy) / max(float(cy), 1.0)) - 0.75))
        pair_alignment_scores.append(1.0 if min(axis_x, axis_y) < 0.045 else 0.0)

    def _harmonic_distance(values: np.ndarray) -> np.ndarray:
        distances = [np.abs(values - harmonic) for harmonic in (0.25, 0.50, 0.75)]
        return np.minimum.reduce(distances)

    harmonic_x = _harmonic_distance(np.abs(norm_x))
    harmonic_y = _harmonic_distance(np.abs(norm_y))
    grid_mask = high_mask & ((harmonic_x < 0.04) | (harmonic_y < 0.04))
    grid_energy = float(np.mean(log_psd[grid_mask])) if np.any(grid_mask) else baseline
    grid_energy_ratio = grid_energy / (float(np.mean(high_values)) + 1e-6)

    peak_prominence = float(np.mean(pair_prominences)) if pair_prominences else 0.0
    pair_count = len(symmetric_pairs)
    alignment_ratio = float(np.mean(pair_alignment_scores)) if pair_alignment_scores else 0.0

    peak_score = _clip01((peak_prominence - 0.9) / 2.6)
    pair_score = _clip01(pair_count / 6.0)
    grid_score = _clip01((grid_energy_ratio - 1.02) / 0.45)
    alignment_score = _clip01(alignment_ratio)

    ai_score = _clip01(
        (0.40 * peak_score) +
        (0.25 * pair_score) +
        (0.25 * grid_score) +
        (0.10 * alignment_score)
    )

    detected = ai_score >= 0.55 or (pair_count >= 4 and grid_energy_ratio > 1.18)
    if detected:
        message = (
            f"Checkerboard-like FFT peaks found ({pair_count} mirrored peak pairs, "
            f"grid energy ratio {grid_energy_ratio:.2f})."
        )
    else:
        message = "No strong checkerboard-style up-sampling peaks detected in the FFT power spectrum."

    return {
        "available": True,
        "ai_score": ai_score,
        "detected": bool(detected),
        "message": message,
        "symmetric_peak_pairs": pair_count,
        "grid_energy_ratio": grid_energy_ratio,
        "peak_prominence": peak_prominence,
        "power_spectrum_density_mean": float(np.mean(power_density[high_mask])),
    }


def _get_pose_model():
    global _POSE
    global _POSE_ENUM

    if _POSE is not None:
        return _POSE

    if mp_pose is not None:
        _POSE = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.35,
        )
        _POSE_ENUM = mp_pose.PoseLandmark
        return _POSE

    if _can_use_mediapipe_tasks():
        _ensure_tasks_model_paths()
        if _POSE_TASK_MODEL_PATH:
            options = _TASKS_VISION.PoseLandmarkerOptions(
                base_options=_BASE_OPTIONS(model_asset_path=_POSE_TASK_MODEL_PATH),
                running_mode=_TASKS_VISION.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.35,
                min_pose_presence_confidence=0.35,
                min_tracking_confidence=0.35,
            )
            _POSE = _TASKS_VISION.PoseLandmarker.create_from_options(options)
            _POSE_ENUM = _TASKS_VISION.PoseLandmark
            return _POSE

    return _POSE


def _get_hands_model():
    global _HANDS
    if _HANDS is not None:
        return _HANDS

    if mp_hands is not None:
        _HANDS = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.35,
        )
        return _HANDS

    if _can_use_mediapipe_tasks():
        _ensure_tasks_model_paths()
        if _HAND_TASK_MODEL_PATH:
            options = _TASKS_VISION.HandLandmarkerOptions(
                base_options=_BASE_OPTIONS(model_asset_path=_HAND_TASK_MODEL_PATH),
                running_mode=_TASKS_VISION.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.35,
                min_hand_presence_confidence=0.35,
                min_tracking_confidence=0.35,
            )
            _HANDS = _TASKS_VISION.HandLandmarker.create_from_options(options)
            return _HANDS

    return _HANDS


def _landmark_xy(landmark) -> np.ndarray:
    return np.array([float(landmark.x), float(landmark.y)], dtype=np.float32)


def _distance(a, b) -> float:
    return float(np.linalg.norm(_landmark_xy(a) - _landmark_xy(b)))


def _angle(a, b, c) -> float:
    ba = _landmark_xy(a) - _landmark_xy(b)
    bc = _landmark_xy(c) - _landmark_xy(b)
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _ratio_anomaly(ratio: float, upper_ok: float, hard_upper: float) -> float:
    if ratio <= upper_ok:
        return 0.0
    if ratio >= hard_upper:
        return 1.0
    return _clip01((ratio - upper_ok) / (hard_upper - upper_ok))


def _body_anatomy_score(landmarks, pose_enum) -> Tuple[float, List[str], Dict]:

    core_indexes = [
        pose_enum.LEFT_SHOULDER.value,
        pose_enum.RIGHT_SHOULDER.value,
        pose_enum.LEFT_ELBOW.value,
        pose_enum.RIGHT_ELBOW.value,
        pose_enum.LEFT_WRIST.value,
        pose_enum.RIGHT_WRIST.value,
        pose_enum.LEFT_HIP.value,
        pose_enum.RIGHT_HIP.value,
        pose_enum.LEFT_KNEE.value,
        pose_enum.RIGHT_KNEE.value,
        pose_enum.LEFT_ANKLE.value,
        pose_enum.RIGHT_ANKLE.value,
    ]
    visible = [
        idx for idx in core_indexes
        if float(getattr(landmarks[idx], "visibility", 1.0)) >= 0.45
    ]
    if len(visible) < 6:
        return 0.0, ["Body landmarks were too incomplete for a stable anatomy check."], {
            "visible_core_landmarks": len(visible),
        }

    pairs = {
        "upper_arm_ratio": (
            _distance(landmarks[pose_enum.LEFT_SHOULDER.value], landmarks[pose_enum.LEFT_ELBOW.value]),
            _distance(landmarks[pose_enum.RIGHT_SHOULDER.value], landmarks[pose_enum.RIGHT_ELBOW.value]),
        ),
        "lower_arm_ratio": (
            _distance(landmarks[pose_enum.LEFT_ELBOW.value], landmarks[pose_enum.LEFT_WRIST.value]),
            _distance(landmarks[pose_enum.RIGHT_ELBOW.value], landmarks[pose_enum.RIGHT_WRIST.value]),
        ),
        "upper_leg_ratio": (
            _distance(landmarks[pose_enum.LEFT_HIP.value], landmarks[pose_enum.LEFT_KNEE.value]),
            _distance(landmarks[pose_enum.RIGHT_HIP.value], landmarks[pose_enum.RIGHT_KNEE.value]),
        ),
        "lower_leg_ratio": (
            _distance(landmarks[pose_enum.LEFT_KNEE.value], landmarks[pose_enum.LEFT_ANKLE.value]),
            _distance(landmarks[pose_enum.RIGHT_KNEE.value], landmarks[pose_enum.RIGHT_ANKLE.value]),
        ),
    }

    ratio_scores = []
    details = {"visible_core_landmarks": len(visible)}
    messages: List[str] = []

    for key, (left_len, right_len) in pairs.items():
        ratio = max(left_len, right_len) / (min(left_len, right_len) + 1e-6)
        details[key] = ratio
        ratio_scores.append(_ratio_anomaly(ratio, upper_ok=1.35, hard_upper=2.10))
        if ratio > 1.55:
            messages.append(f"{key.replace('_', ' ')} looks asymmetrical ({ratio:.2f}x left/right difference).")

    elbow_left = _angle(
        landmarks[pose_enum.LEFT_SHOULDER.value],
        landmarks[pose_enum.LEFT_ELBOW.value],
        landmarks[pose_enum.LEFT_WRIST.value],
    )
    elbow_right = _angle(
        landmarks[pose_enum.RIGHT_SHOULDER.value],
        landmarks[pose_enum.RIGHT_ELBOW.value],
        landmarks[pose_enum.RIGHT_WRIST.value],
    )
    knee_left = _angle(
        landmarks[pose_enum.LEFT_HIP.value],
        landmarks[pose_enum.LEFT_KNEE.value],
        landmarks[pose_enum.LEFT_ANKLE.value],
    )
    knee_right = _angle(
        landmarks[pose_enum.RIGHT_HIP.value],
        landmarks[pose_enum.RIGHT_KNEE.value],
        landmarks[pose_enum.RIGHT_ANKLE.value],
    )

    body_angles = {
        "left_elbow_angle": elbow_left,
        "right_elbow_angle": elbow_right,
        "left_knee_angle": knee_left,
        "right_knee_angle": knee_right,
    }
    details.update(body_angles)

    angle_scores = []
    for name, angle_value in body_angles.items():
        if angle_value < 8.0:
            angle_scores.append(_clip01((12.0 - angle_value) / 12.0))
            messages.append(f"{name.replace('_', ' ')} is implausibly acute ({angle_value:.1f} degrees).")
        elif angle_value > 178.0:
            angle_scores.append(_clip01((angle_value - 176.0) / 4.0))
        else:
            angle_scores.append(0.0)

    outside_frame = 0
    for idx in visible:
        point = landmarks[idx]
        if point.x < -0.20 or point.x > 1.20 or point.y < -0.20 or point.y > 1.20:
            outside_frame += 1
    details["outside_frame_landmarks"] = outside_frame

    frame_score = _clip01(outside_frame / 4.0)
    body_score = _clip01(
        (0.55 * float(np.mean(ratio_scores))) +
        (0.30 * float(np.mean(angle_scores))) +
        (0.15 * frame_score)
    )

    if body_score <= 0.15 and not messages:
        messages.append("Body pose landmarks look biomechanically plausible.")

    return body_score, messages, details


def _hand_anatomy_score(hand_landmarks_list) -> Tuple[float, List[str], Dict]:
    details = {
        "hands_detected": 0,
        "geometry_anomaly_score": 0.0,
        "more_than_two_hands": False,
    }
    if not hand_landmarks_list:
        return 0.0, ["No hands were detected for a hand anatomy check."], details

    hands = hand_landmarks_list
    details["hands_detected"] = len(hands)
    if len(hands) > 2:
        details["more_than_two_hands"] = True
        return 1.0, ["More than two hands were inferred in a single image."], details

    anomaly_scores = []
    messages: List[str] = []
    finger_chains = (
        ("index", (5, 6, 7, 8)),
        ("middle", (9, 10, 11, 12)),
        ("ring", (13, 14, 15, 16)),
        ("pinky", (17, 18, 19, 20)),
    )

    for hand_idx, hand in enumerate(hands, start=1):
        palm_size = max(_distance(hand[0], hand[9]), 1e-6)
        for finger_name, chain in finger_chains:
            seg1 = _distance(hand[chain[0]], hand[chain[1]])
            seg2 = _distance(hand[chain[1]], hand[chain[2]])
            seg3 = _distance(hand[chain[2]], hand[chain[3]])
            finger_length = seg1 + seg2 + seg3

            ratio_12 = max(seg1, seg2) / (min(seg1, seg2) + 1e-6)
            ratio_23 = max(seg2, seg3) / (min(seg2, seg3) + 1e-6)
            palm_ratio = finger_length / palm_size

            finger_score = max(
                _ratio_anomaly(ratio_12, upper_ok=1.95, hard_upper=3.20),
                _ratio_anomaly(ratio_23, upper_ok=1.95, hard_upper=3.20),
                _clip01((0.45 - palm_ratio) / 0.25) if palm_ratio < 0.45 else _clip01((palm_ratio - 2.40) / 1.20),
            )
            anomaly_scores.append(finger_score)

            if finger_score > 0.6:
                messages.append(
                    f"Hand {hand_idx} {finger_name} finger has implausible segment proportions."
                )

    geometry_score = float(np.mean(anomaly_scores)) if anomaly_scores else 0.0
    details["geometry_anomaly_score"] = geometry_score

    if geometry_score <= 0.15 and not messages:
        messages.append("Hand landmark geometry looks plausible.")

    return geometry_score, messages, details


def analyze_anatomical_consistency(image: Union[Image.Image, np.ndarray]) -> Dict:
    if mp is None:
        return {
            "available": False,
            "ai_score": 0.0,
            "detected": False,
            "message": "MediaPipe is not installed, so anatomy validation was skipped.",
        }

    rgb = _resize_with_pil(_as_rgb_array(image), max_side=768)
    pose_model = _get_pose_model()
    hands_model = _get_hands_model()
    if pose_model is None or hands_model is None or _POSE_ENUM is None:
        return {
            "available": False,
            "ai_score": 0.0,
            "detected": False,
            "message": _mediapipe_unavailable_message(),
        }

    if mp_pose is not None and mp_hands is not None:
        pose_result = pose_model.process(rgb)
        hands_result = hands_model.process(rgb)
        pose_landmark_sets = [pose_result.pose_landmarks.landmark] if pose_result is not None and pose_result.pose_landmarks is not None else []
        hand_landmark_sets = [hand.landmark for hand in hands_result.multi_hand_landmarks] if hands_result is not None and hands_result.multi_hand_landmarks else []
    else:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        pose_result = pose_model.detect(mp_image)
        hands_result = hands_model.detect(mp_image)
        pose_landmark_sets = list(getattr(pose_result, "pose_landmarks", []) or [])
        hand_landmark_sets = list(getattr(hands_result, "hand_landmarks", []) or [])

    messages: List[str] = []
    details: Dict = {}
    scores = []

    if pose_landmark_sets:
        body_score, body_messages, body_details = _body_anatomy_score(pose_landmark_sets[0], _POSE_ENUM)
        scores.append(body_score)
        messages.extend(body_messages)
        details["body"] = body_details
    else:
        details["body"] = {"visible_core_landmarks": 0}
        messages.append("No reliable body pose was detected.")

    hand_score, hand_messages, hand_details = _hand_anatomy_score(hand_landmark_sets)
    details["hands"] = hand_details
    if hand_details.get("hands_detected", 0) > 0:
        scores.append(hand_score)
    messages.extend(hand_messages)

    ai_score = _clip01(float(np.mean(scores))) if scores else 0.0
    detected = ai_score >= 0.50
    message = messages[0] if messages else "No anatomical inconsistency was detected."

    return {
        "available": True,
        "ai_score": ai_score,
        "detected": bool(detected),
        "message": message,
        "details": details,
        "messages": messages[:4],
    }


def analyze_semantic_entropy(image: Union[Image.Image, np.ndarray]) -> Dict:
    if shannon_entropy is None:
        return {
            "available": False,
            "ai_score": 0.0,
            "detected": False,
            "message": "scikit-image is not installed, so entropy analysis was skipped.",
        }

    rgb = _resize_with_pil(_as_rgb_array(image), max_side=512)
    gray = np.mean(rgb.astype(np.float32), axis=2).astype(np.uint8)
    height, width = gray.shape

    grid_rows = 4 if height >= 256 else 3
    grid_cols = 4 if width >= 256 else 3
    row_edges = np.linspace(0, height, grid_rows + 1, dtype=int)
    col_edges = np.linspace(0, width, grid_cols + 1, dtype=int)

    patch_entropies = []
    edge_entropies = []
    center_entropies = []

    for row in range(grid_rows):
        for col in range(grid_cols):
            patch = gray[row_edges[row]:row_edges[row + 1], col_edges[col]:col_edges[col + 1]]
            if patch.size < 64:
                continue
            entropy_value = float(shannon_entropy(patch))
            patch_entropies.append(entropy_value)

            is_edge_patch = row in (0, grid_rows - 1) or col in (0, grid_cols - 1)
            if is_edge_patch:
                edge_entropies.append(entropy_value)
            else:
                center_entropies.append(entropy_value)

    if not patch_entropies:
        return {
            "available": True,
            "ai_score": 0.0,
            "detected": False,
            "message": "Image was too small for a stable local entropy analysis.",
        }

    patch_array = np.array(patch_entropies, dtype=np.float32)
    edge_mean = float(np.mean(edge_entropies)) if edge_entropies else float(np.mean(patch_array))
    center_mean = float(np.mean(center_entropies)) if center_entropies else float(np.mean(patch_array))
    entropy_spread = float(np.percentile(patch_array, 90) - np.percentile(patch_array, 10))
    global_entropy = float(shannon_entropy(gray))
    low_entropy_fraction = float(np.mean(patch_array < max(2.2, global_entropy * 0.45)))

    center_gap = max(0.0, center_mean - edge_mean)
    mushy_background_score = _clip01((center_gap - 0.6) / 2.4)
    spread_score = _clip01((entropy_spread - 1.1) / 2.4)
    bland_background_score = _clip01((low_entropy_fraction - 0.35) / 0.45)

    # Very low-detail scenes should not be treated as suspicious by themselves.
    if entropy_spread < 0.45 and center_gap < 0.15:
        bland_background_score = 0.0

    if global_entropy < 1.0 and entropy_spread < 0.25:
        return {
            "available": True,
            "ai_score": 0.0,
            "detected": False,
            "message": "Entropy map is uniformly low-detail and is not treated as suspicious by itself.",
            "global_entropy": global_entropy,
            "center_entropy": center_mean,
            "edge_entropy": edge_mean,
            "local_entropy_spread": entropy_spread,
            "low_entropy_fraction": low_entropy_fraction,
        }

    ai_score = _clip01(
        (0.45 * mushy_background_score) +
        (0.35 * spread_score) +
        (0.20 * bland_background_score)
    )
    detected = ai_score >= 0.45

    if detected:
        message = (
            f"Local entropy varies sharply between subject and background "
            f"(center-edge gap {center_gap:.2f}, spread {entropy_spread:.2f})."
        )
    else:
        message = "Local entropy distribution looks consistent with natural scene detail."

    return {
        "available": True,
        "ai_score": ai_score,
        "detected": bool(detected),
        "message": message,
        "global_entropy": global_entropy,
        "center_entropy": center_mean,
        "edge_entropy": edge_mean,
        "local_entropy_spread": entropy_spread,
        "low_entropy_fraction": low_entropy_fraction,
    }


def get_final_custom_score(
    image: Union[Image.Image, np.ndarray],
    resnet_ai_prob: float,
) -> Dict:
    resnet_ai_prob = _clip01(float(resnet_ai_prob))
    frequency = analyze_frequency_artifacts(image)
    anatomy = analyze_anatomical_consistency(image)
    entropy = analyze_semantic_entropy(image)

    weighted_ai_prob = (
        (CUSTOM_WEIGHTS["resnet"] * resnet_ai_prob) +
        (CUSTOM_WEIGHTS["frequency"] * float(frequency.get("ai_score", 0.0))) +
        (CUSTOM_WEIGHTS["anatomy"] * float(anatomy.get("ai_score", 0.0))) +
        (CUSTOM_WEIGHTS["entropy"] * float(entropy.get("ai_score", 0.0)))
    )

    weighted_ai_prob = _clip01(weighted_ai_prob)

    module_map = {
        "frequency": frequency,
        "anatomy": anatomy,
        "entropy": entropy,
    }
    available_modules = []
    supporting_modules = []
    negative_modules = []
    forensics_only_weight = 0.0
    forensics_only_score = 0.0
    max_module_score = 0.0

    for module_name, module_result in module_map.items():
        if not module_result.get("available"):
            continue

        module_score = _clip01(float(module_result.get("ai_score", 0.0)))
        available_modules.append(module_name)
        forensics_only_weight += CUSTOM_WEIGHTS[module_name]
        forensics_only_score += CUSTOM_WEIGHTS[module_name] * module_score
        max_module_score = max(max_module_score, module_score)

        if module_result.get("detected") or module_score >= FORENSIC_SUPPORT_THRESHOLD:
            supporting_modules.append(module_name)
        elif module_score <= FORENSIC_NEGATIVE_THRESHOLD:
            negative_modules.append(module_name)

    if forensics_only_weight > 0:
        forensics_only_ai_prob = _clip01(forensics_only_score / forensics_only_weight)
    else:
        forensics_only_ai_prob = 0.0

    if supporting_modules:
        if (
            len(supporting_modules) == 1 and
            supporting_modules[0] == "frequency" and
            forensics_only_ai_prob < 0.70
        ):
            final_ai_prob = (0.70 * resnet_ai_prob) + (0.30 * weighted_ai_prob)
            calibration_note = (
                "Only frequency evidence supported AI, which can be caused by compression artifacts; "
                "conservative calibration was applied."
            )
        elif len(supporting_modules) >= 2 or forensics_only_ai_prob >= 0.60:
            final_ai_prob = max(weighted_ai_prob, (0.80 * resnet_ai_prob) + (0.20 * forensics_only_ai_prob))
            calibration_note = "Multiple forensic checks supported the AI hypothesis."
        else:
            final_ai_prob = max(weighted_ai_prob, (0.85 * resnet_ai_prob) + (0.15 * forensics_only_ai_prob))
            calibration_note = f"Forensic support was limited to {', '.join(supporting_modules)}."
    else:
        final_ai_prob = (0.60 * resnet_ai_prob) + (0.40 * weighted_ai_prob)
        if len(available_modules) >= 2 and max_module_score <= 0.20:
            final_ai_prob -= 0.10
            calibration_note = "No forensic check supported the AI hypothesis, so the score was reduced."
        elif forensics_only_ai_prob <= 0.12:
            final_ai_prob -= 0.05
            calibration_note = "Forensic checks were largely negative, so the score was reduced."
        else:
            calibration_note = "Forensic checks were inconclusive, so the score was conservatively calibrated."

    final_ai_prob = _clip01(final_ai_prob)

    artifacts = [
        f"ResNet AI score: {resnet_ai_prob * 100:.1f}%",
        f"Frequency artifact score: {float(frequency.get('ai_score', 0.0)) * 100:.1f}%",
        f"Anatomy artifact score: {float(anatomy.get('ai_score', 0.0)) * 100:.1f}%",
        f"Entropy artifact score: {float(entropy.get('ai_score', 0.0)) * 100:.1f}%",
        f"Forensics-only AI score: {forensics_only_ai_prob * 100:.1f}%",
        f"Weighted custom AI score: {weighted_ai_prob * 100:.1f}%",
        f"Calibrated custom AI score: {final_ai_prob * 100:.1f}%",
        calibration_note,
    ]

    if frequency.get("message"):
        artifacts.append(f"Frequency module: {frequency['message']}")
    if anatomy.get("message"):
        artifacts.append(f"Anatomy module: {anatomy['message']}")
    if entropy.get("message"):
        artifacts.append(f"Entropy module: {entropy['message']}")

    return {
        "source": "Custom AI model (ResNet50 + frequency/anatomy/entropy forensics)",
        "weights": CUSTOM_WEIGHTS.copy(),
        "resnet_ai_prob": resnet_ai_prob,
        "frequency": frequency,
        "anatomy": anatomy,
        "entropy": entropy,
        "available_modules": available_modules,
        "supporting_modules": supporting_modules,
        "negative_modules": negative_modules,
        "forensics_only_ai_prob": forensics_only_ai_prob,
        "weighted_ai_prob": weighted_ai_prob,
        "final_ai_prob": final_ai_prob,
        "calibration_note": calibration_note,
        "artifacts": artifacts,
    }

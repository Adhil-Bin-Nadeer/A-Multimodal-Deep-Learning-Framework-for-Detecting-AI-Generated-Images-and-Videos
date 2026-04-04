import os
import warnings
from typing import Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.exceptions import InconsistentVersionWarning
from torchvision import models, transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

from custom_forensics import get_final_custom_score as run_custom_forensics


# -------- CONFIG --------
# Paths relative to the project root (one level up from backend/)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESNET_PATH = os.path.join(_BASE_DIR, "model_output", "resnet50_finetuned_benchmark.pth")
META_LEARNER_PATH = os.path.join(_BASE_DIR, "ai_detector_meta_learner.joblib")
POLY_TRANSFORM_PATH = os.path.join(_BASE_DIR, "polynomial_transformer.joblib")

VIT_NAME = "dima806/ai_vs_real_image_detection"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AI_LABEL_HINTS = ("ai", "fake", "synthetic", "generated", "deepfake", "manipulated")
REAL_LABEL_HINTS = ("real", "authentic", "human", "natural")


def _resolve_ai_index_from_labels(id2label: dict) -> Optional[int]:
    normalized = {}
    for idx_raw, label_raw in (id2label or {}).items():
        try:
            idx = int(idx_raw)
        except (TypeError, ValueError):
            continue
        normalized[idx] = str(label_raw).lower().replace("_", " ").replace("-", " ")

    for idx, label in normalized.items():
        if any(hint in label for hint in AI_LABEL_HINTS):
            return idx

    if len(normalized) == 2:
        for idx, label in normalized.items():
            if any(hint in label for hint in REAL_LABEL_HINTS):
                for other_idx in normalized.keys():
                    if other_idx != idx:
                        return other_idx

    return None


class AIEnsemblePredictor:
    def __init__(self):
        print(f"Loading models to {DEVICE}...")
        self.device = DEVICE
        self.resnet_ai_index = int(os.environ.get("RESNET_AI_INDEX", "1"))
        if self.resnet_ai_index not in (0, 1):
            raise ValueError("RESNET_AI_INDEX must be 0 or 1")
        self.use_tta = os.environ.get("ENABLE_TTA", "1") == "1"

        self.vit = None
        self.vit_processor = None
        self.vit_ai_index = None
        self.vit_error = None

        self.meta_model = None
        self.poly = None
        self.meta_error = None
        self.meta_version_warning = False

        # 1. Load ResNet50
        self.resnet = models.resnet50(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)

        if os.path.exists(RESNET_PATH):
            self.resnet.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE))
            print("ResNet loaded.")
        else:
            raise FileNotFoundError(f"Could not find ResNet model at {RESNET_PATH}")

        self.resnet.to(self.device).eval()

        self.res_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self._load_vit()
        self._load_meta_learner()

    def _load_vit(self):
        try:
            self.vit = AutoModelForImageClassification.from_pretrained(VIT_NAME).to(self.device).eval()
            self.vit_processor = AutoImageProcessor.from_pretrained(VIT_NAME)
            self.vit_ai_index = _resolve_ai_index_from_labels(getattr(self.vit.config, "id2label", {}))
            if self.vit_ai_index is None:
                self.vit_ai_index = int(os.environ.get("VIT_AI_INDEX", "1"))
            vit_num_labels = int(getattr(self.vit.config, "num_labels", 2))
            if self.vit_ai_index < 0 or self.vit_ai_index >= vit_num_labels:
                raise ValueError(f"VIT_AI_INDEX must be between 0 and {vit_num_labels - 1}")
            print(f"ViT loaded. (AI class index: {self.vit_ai_index})")
        except Exception as exc:
            self.vit = None
            self.vit_processor = None
            self.vit_ai_index = None
            self.vit_error = str(exc)
            print(f"Warning: ViT model unavailable; continuing with ResNet and forensic modules only. ({exc})")

    def _load_meta_learner(self):
        if not (os.path.exists(META_LEARNER_PATH) and os.path.exists(POLY_TRANSFORM_PATH)):
            self.meta_error = "Meta-learner files not found."
            print("Warning: Meta-learner files not found; continuing without meta fusion.")
            return

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.meta_model = joblib.load(META_LEARNER_PATH)
                self.poly = joblib.load(POLY_TRANSFORM_PATH)

            for warning_item in caught:
                if isinstance(warning_item.message, InconsistentVersionWarning):
                    self.meta_version_warning = True

            if self.meta_version_warning:
                print("Warning: Meta-Learner version differs from current sklearn; using stabilized fusion.")
            print("Meta-Learner loaded.")
        except Exception as exc:
            self.meta_model = None
            self.poly = None
            self.meta_error = str(exc)
            print(f"Warning: Meta-learner unavailable; continuing without meta fusion. ({exc})")

    def _get_image_views(self, img: Image.Image):
        views = [img]
        if self.use_tta:
            views.append(ImageOps.mirror(img))
        return views

    def _get_resnet_scores_from_pil(self, img: Image.Image):
        views = self._get_image_views(img)
        res_scores = []

        with torch.no_grad():
            for view in views:
                res_input = self.res_transform(view).unsqueeze(0).to(self.device)
                res_logits = self.resnet(res_input)
                res_probs = torch.softmax(res_logits, dim=1)[0]
                if self.resnet_ai_index >= res_probs.shape[0]:
                    raise ValueError("RESNET_AI_INDEX is out of range for model output")
                res_scores.append(float(res_probs[self.resnet_ai_index].item()))

        return float(np.mean(res_scores)), len(views)

    def _get_vit_score_from_pil(self, img: Image.Image) -> Optional[float]:
        if self.vit is None or self.vit_processor is None or self.vit_ai_index is None:
            return None

        views = self._get_image_views(img)
        vit_scores = []

        with torch.no_grad():
            for view in views:
                vit_inputs = self.vit_processor(images=view, return_tensors="pt").to(self.device)
                vit_logits = self.vit(**vit_inputs).logits
                vit_probs = torch.softmax(vit_logits, dim=1)[0]
                if self.vit_ai_index >= vit_probs.shape[0]:
                    raise ValueError("VIT_AI_INDEX is out of range for model output")
                vit_scores.append(float(vit_probs[self.vit_ai_index].item()))

        return float(np.mean(vit_scores))

    def _get_component_scores_from_pil(self, img: Image.Image):
        res_prob, views_count = self._get_resnet_scores_from_pil(img)
        vit_prob = self._get_vit_score_from_pil(img)
        return res_prob, vit_prob, views_count

    def _combine_model_scores(self, res_prob: float, vit_prob: Optional[float]):
        if vit_prob is None:
            return res_prob, {
                'base_blend_ai_score': res_prob * 100.0,
                'meta_ai_score': None,
                'fusion_note': 'ViT unavailable; using ResNet score as the legacy baseline.',
            }

        base_blend = float((res_prob + vit_prob) / 2.0)
        if self.meta_model is None or self.poly is None:
            return base_blend, {
                'base_blend_ai_score': base_blend * 100.0,
                'meta_ai_score': None,
                'fusion_note': 'Meta-learner unavailable; using the ResNet/ViT average as the legacy baseline.',
            }

        raw_scores = np.array([[res_prob, vit_prob]], dtype=np.float32)
        poly_features = self.poly.transform(raw_scores)
        meta_prob = float(self.meta_model.predict_proba(poly_features)[0, 1])

        if not np.isfinite(meta_prob):
            raise ValueError("Model produced a non-finite probability")

        if self.meta_version_warning:
            final_ai_prob = (0.55 * base_blend) + (0.45 * meta_prob)
            fusion_note = "Meta model version differs; combined with base model average."
        else:
            drift = abs(meta_prob - base_blend)
            if drift > 0.35:
                final_ai_prob = (0.50 * base_blend) + (0.50 * meta_prob)
                fusion_note = "Model outputs differed strongly; used balanced fusion."
            else:
                final_ai_prob = meta_prob
                fusion_note = "Meta model score used directly."

        final_ai_prob = float(max(0.0, min(1.0, final_ai_prob)))
        return final_ai_prob, {
            'base_blend_ai_score': base_blend * 100.0,
            'meta_ai_score': meta_prob * 100.0,
            'fusion_note': fusion_note,
        }

    def _build_artifacts(
        self,
        label: str,
        res_prob: float,
        vit_prob: Optional[float],
        views_count: int,
        legacy_ai_prob: float,
        custom_result: dict,
        final_ai_prob: float,
        ai_threshold: float,
        decision_source: str,
        fusion_note: str,
    ):
        artifacts = [
            f"Final AI score: {final_ai_prob * 100:.1f}%",
            f"AI decision threshold: {ai_threshold * 100:.1f}%",
            f"Decision source: {decision_source}",
            f"ResNet score: {res_prob * 100:.1f}%",
            f"Legacy ensemble score: {legacy_ai_prob * 100:.1f}%",
            f"Custom forensic score: {custom_result['final_ai_prob'] * 100:.1f}%",
            f"Image views checked: {views_count}",
            fusion_note,
        ]

        if vit_prob is not None:
            artifacts.append(f"ViT score: {vit_prob * 100:.1f}%")
        elif self.vit_error:
            artifacts.append(f"ViT unavailable: {self.vit_error}")

        artifacts.extend(custom_result.get('artifacts', []))

        if label == "AI Image":
            artifacts.append("Layer 3 classified the image as AI-generated.")
        else:
            artifacts.append("Layer 3 classified the image as real/authentic.")

        return artifacts

    def _resolve_ai_threshold(self, vit_prob: Optional[float], custom_result: dict, final_ai_prob: float) -> float:
        support_count = len(custom_result.get('supporting_modules', []))
        negative_count = len(custom_result.get('negative_modules', []))
        forensics_only_ai_prob = float(custom_result.get('forensics_only_ai_prob', 0.0))

        if support_count >= 2:
            return 0.55 if vit_prob is not None else 0.58
        if support_count == 1:
            return 0.60 if vit_prob is not None else 0.64

        # No forensic support: require very high model certainty.
        if support_count == 0 and forensics_only_ai_prob <= 0.35:
            return 0.88 if vit_prob is not None else 0.90

        # If forensic evidence is weak/negative, require stronger model certainty.
        if negative_count >= 2 or forensics_only_ai_prob <= 0.15:
            return 0.84 if vit_prob is not None else 0.86
        if negative_count == 1 or forensics_only_ai_prob <= 0.25:
            return 0.82 if vit_prob is not None else 0.85
        if forensics_only_ai_prob <= 0.35:
            return 0.78 if vit_prob is not None else 0.82

        if final_ai_prob >= 0.90:
            return 0.62 if vit_prob is not None else 0.66
        return 0.72 if vit_prob is not None else 0.75

    def _decision_confidence(self, ai_probability: float, ai_threshold: float, is_ai: bool) -> float:
        if is_ai:
            relative_margin = (ai_probability - ai_threshold) / max(1.0 - ai_threshold, 1e-6)
        else:
            relative_margin = (ai_threshold - ai_probability) / max(ai_threshold, 1e-6)

        relative_margin = float(max(0.0, min(1.0, relative_margin)))
        return 0.5 + (0.5 * relative_margin)

    def _predict_ai_probability_from_pil(self, img: Image.Image) -> float:
        res_prob, vit_prob, _ = self._get_component_scores_from_pil(img)
        legacy_ai_prob, _ = self._combine_model_scores(res_prob, vit_prob)
        custom_result = run_custom_forensics(img, res_prob)
        custom_ai_prob = float(custom_result['final_ai_prob'])

        if vit_prob is None:
            final_ai_prob = custom_ai_prob
        elif custom_result.get('supporting_modules'):
            final_ai_prob = max(legacy_ai_prob, custom_ai_prob)
        else:
            final_ai_prob = float((0.70 * legacy_ai_prob) + (0.30 * custom_ai_prob))

        return float(max(0.0, min(1.0, final_ai_prob)))

    def predict_proba(self, image_path: str) -> float:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        with Image.open(image_path) as opened:
            img = opened.convert("RGB")
        return self._predict_ai_probability_from_pil(img)

    def predict_proba_from_pil(self, image: Image.Image) -> float:
        return self._predict_ai_probability_from_pil(image.convert("RGB"))

    def get_final_custom_score(self, image):
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found at {image}")
            with Image.open(image) as opened:
                pil_image = opened.convert("RGB")
        else:
            pil_image = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image)).convert("RGB")

        res_prob, _ = self._get_resnet_scores_from_pil(pil_image)
        return run_custom_forensics(pil_image, res_prob)

    def predict(self, image_path, return_details: bool = False):
        if not os.path.exists(image_path):
            if return_details:
                return {'status': 'error', 'error': f"Image not found at {image_path}"}
            return "Error", f"Image not found at {image_path}"

        try:
            with Image.open(image_path) as opened:
                img = opened.convert("RGB")
        except Exception as exc:
            if return_details:
                return {'status': 'error', 'error': f"Invalid image file: {exc}"}
            return "Error", f"Invalid image file: {exc}"

        try:
            res_prob, vit_prob, views_count = self._get_component_scores_from_pil(img)
            legacy_ai_prob, fusion_info = self._combine_model_scores(res_prob, vit_prob)
            custom_result = run_custom_forensics(img, res_prob)

            custom_ai_prob = float(custom_result['final_ai_prob'])
            if vit_prob is None:
                final_ai_prob = custom_ai_prob
                decision_source = 'Calibrated ResNet50 + forensic cross-checks'
            elif custom_ai_prob >= legacy_ai_prob:
                final_ai_prob = custom_ai_prob
                decision_source = custom_result.get('source', 'Custom AI model')
            else:
                final_ai_prob = float((0.70 * legacy_ai_prob) + (0.30 * custom_ai_prob))
                decision_source = 'Existing ensemble baseline with conservative forensic calibration'
        except Exception as exc:
            if return_details:
                return {'status': 'error', 'error': f"Model inference failed: {exc}"}
            return "Error", f"Model inference failed: {exc}"

        ai_threshold = self._resolve_ai_threshold(vit_prob, custom_result, final_ai_prob)
        is_ai = final_ai_prob >= ai_threshold

        if is_ai:
            label = "AI Image"
            confidence = self._decision_confidence(final_ai_prob, ai_threshold, True)
        else:
            label = "Real Image"
            confidence = self._decision_confidence(final_ai_prob, ai_threshold, False)

        if return_details:
            return {
                'status': 'complete',
                'label': label,
                'confidence': confidence,
                'confidence_percent': confidence * 100.0,
                'ai_probability': final_ai_prob,
                'ai_probability_percent': final_ai_prob * 100.0,
                'ai_decision_threshold': ai_threshold * 100.0,
                'source': decision_source,
                'model_scores': {
                    'resnet_ai_score': res_prob * 100.0,
                    'vit_ai_score': None if vit_prob is None else vit_prob * 100.0,
                    'legacy_ensemble_ai_score': legacy_ai_prob * 100.0,
                    'custom_forensic_ai_score': custom_result['final_ai_prob'] * 100.0,
                    'forensics_only_ai_score': custom_result.get('forensics_only_ai_prob', 0.0) * 100.0,
                    'weighted_custom_ai_score': custom_result['weighted_ai_prob'] * 100.0,
                    'final_ai_score': final_ai_prob * 100.0,
                    'ai_decision_threshold': ai_threshold * 100.0,
                    'base_blend_ai_score': fusion_info.get('base_blend_ai_score'),
                    'meta_ai_score': fusion_info.get('meta_ai_score'),
                    'views_checked': views_count,
                },
                'forensic_modules': {
                    'frequency': custom_result.get('frequency'),
                    'anatomy': custom_result.get('anatomy'),
                    'entropy': custom_result.get('entropy'),
                    'weights': custom_result.get('weights'),
                    'available_modules': custom_result.get('available_modules', []),
                    'supporting_modules': custom_result.get('supporting_modules', []),
                    'negative_modules': custom_result.get('negative_modules', []),
                    'calibration_note': custom_result.get('calibration_note'),
                },
                'artifacts': self._build_artifacts(
                    label,
                    res_prob,
                    vit_prob,
                    views_count,
                    legacy_ai_prob,
                    custom_result,
                    final_ai_prob,
                    ai_threshold,
                    decision_source,
                    fusion_info.get('fusion_note', 'Model fusion applied.'),
                ),
            }

        return label, confidence


# -------- EXECUTION --------
if __name__ == "__main__":
    predictor = AIEnsemblePredictor()

    print("\n" + "=" * 40)
    print("Single Image Detector Ready")
    print("=" * 40)

    while True:
        img_path = input("\nEnter path to image (or 'q' to quit): ").strip().strip('"')

        if img_path.lower() == 'q':
            break

        label, score = predictor.predict(img_path)

        if label == "Error":
            print(f"Error: {score}")
        else:
            color = "\033[91m" if label == "AI Image" else "\033[92m"
            reset = "\033[0m"
            print(f"Prediction: {color}{label}{reset}")
            print(f"Confidence: {score * 100:.2f}%")
            print("-" * 20)

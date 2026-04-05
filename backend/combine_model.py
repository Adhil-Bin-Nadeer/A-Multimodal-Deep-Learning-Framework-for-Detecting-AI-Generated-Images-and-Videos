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
        # Skip ViT on environments where download speed causes port timeout
        if os.environ.get("SKIP_VIT", "0") == "1":
            self.vit_error = "ViT skipped on this environment (SKIP_VIT=1)"
            print("==> ViT loading skipped. Using ResNet + forensics only.")
            return

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

    def _compute_score_consensus(
        self,
        res_prob: float,
        vit_prob: Optional[float],
        legacy_ai_prob: float,
    ):
        if vit_prob is None:
            ai_consensus = float(np.clip((0.80 * legacy_ai_prob) + (0.20 * res_prob), 0.0, 1.0))
        else:
            ai_consensus = float(np.clip(
                (0.50 * float(vit_prob)) +
                (0.30 * float(legacy_ai_prob)) +
                (0.20 * float(res_prob)),
                0.0,
                1.0,
            ))

        real_consensus = float(np.clip(1.0 - ai_consensus, 0.0, 1.0))
        score_margin = float(ai_consensus - real_consensus)
        return ai_consensus, real_consensus, score_margin

    def _refine_with_score_consensus(
        self,
        res_prob: float,
        vit_prob: Optional[float],
        legacy_ai_prob: float,
        custom_result: dict,
        final_ai_prob: float,
    ):
        ai_consensus, real_consensus, score_margin = self._compute_score_consensus(
            res_prob=res_prob,
            vit_prob=vit_prob,
            legacy_ai_prob=legacy_ai_prob,
        )

        if vit_prob is None:
            return final_ai_prob, ai_consensus, real_consensus, score_margin, False

        support_count = len(custom_result.get('supporting_modules', []))
        negative_modules = list(custom_result.get('negative_modules', []) or [])
        forensics_only_ai_prob = float(custom_result.get('forensics_only_ai_prob', 0.0))
        disagreement = abs(float(vit_prob) - float(res_prob))

        refined_ai_prob = float(final_ai_prob)
        refinement_applied = False

        # In high-disagreement, weak-forensics cases, nudge toward score consensus
        # to reduce false AI outcomes without inflating weak-evidence AI scores.
        if support_count == 0 and forensics_only_ai_prob <= 0.25 and disagreement >= 0.45:
            consensus_target = min(ai_consensus, float(final_ai_prob))
            refined_ai_prob = float(np.clip((0.82 * final_ai_prob) + (0.18 * consensus_target), 0.0, 1.0))

            if len(negative_modules) >= 1 and ai_consensus > final_ai_prob:
                weak_support_target = float(np.clip(
                    (0.70 * float(res_prob)) +
                    (0.30 * forensics_only_ai_prob),
                    0.0,
                    1.0,
                ))
                refined_ai_prob = float(np.clip((0.74 * final_ai_prob) + (0.26 * weak_support_target), 0.0, 1.0))

            refinement_applied = abs(refined_ai_prob - final_ai_prob) > 1e-6
        elif support_count >= 1 and ai_consensus >= 0.75 and final_ai_prob < ai_consensus:
            refined_ai_prob = float(np.clip((0.88 * final_ai_prob) + (0.12 * ai_consensus), 0.0, 1.0))
            refinement_applied = abs(refined_ai_prob - final_ai_prob) > 1e-6

        return refined_ai_prob, ai_consensus, real_consensus, score_margin, refinement_applied

    def _apply_vit_disagreement_guard(
        self,
        res_prob: float,
        vit_prob: Optional[float],
        legacy_ai_prob: float,
        custom_result: dict,
        final_ai_prob: float,
    ):
        """
        Guard against strong ResNet/ViT disagreement causing false Real outcomes.

        This only activates when ViT is highly confident for AI and there is
        at least mild forensic corroboration.
        """
        if vit_prob is None:
            return final_ai_prob, False, ""

        vit_prob = float(vit_prob)
        res_prob = float(res_prob)
        if vit_prob < 0.86 or res_prob > 0.22 or (vit_prob - res_prob) < 0.55:
            return final_ai_prob, False, ""

        forensics_only_ai_prob = float(custom_result.get('forensics_only_ai_prob', 0.0))
        supporting_modules = set(custom_result.get('supporting_modules', []) or [])
        negative_modules = list(custom_result.get('negative_modules', []) or [])
        non_frequency_support = any(module_name != 'frequency' for module_name in supporting_modules)

        frequency_score = float((custom_result.get('frequency') or {}).get('ai_score', 0.0))

        has_forensic_corroboration = (
            non_frequency_support or
            forensics_only_ai_prob >= 0.28 or
            (frequency_score >= 0.48 and vit_prob >= 0.92)
        )

        if not has_forensic_corroboration:
            return final_ai_prob, False, ""

        if len(negative_modules) >= 2 and forensics_only_ai_prob <= 0.25:
            return final_ai_prob, False, ""

        guard_ai_prob = float(np.clip(
            (0.62 * vit_prob) +
            (0.23 * legacy_ai_prob) +
            (0.15 * max(forensics_only_ai_prob, frequency_score)),
            0.0,
            1.0,
        ))

        if guard_ai_prob <= final_ai_prob:
            return final_ai_prob, False, ""

        return guard_ai_prob, True, 'High-confidence ViT disagreement guard applied.'

    def _apply_real_disagreement_guard(
        self,
        res_prob: float,
        vit_prob: Optional[float],
        legacy_ai_prob: float,
        custom_result: dict,
        final_ai_prob: float,
    ):
        """
        Guard against false AI outcomes when ViT is moderately high but
        corroborating forensic evidence is weak.
        """
        if vit_prob is None:
            return final_ai_prob, False, ""

        vit_prob = float(vit_prob)
        res_prob = float(res_prob)
        if vit_prob < 0.66 or res_prob > 0.18 or (vit_prob - res_prob) < 0.42:
            return final_ai_prob, False, ""

        forensics_only_ai_prob = float(custom_result.get('forensics_only_ai_prob', 0.0))
        supporting_modules = set(custom_result.get('supporting_modules', []) or [])
        negative_modules = list(custom_result.get('negative_modules', []) or [])
        non_frequency_support = any(module_name != 'frequency' for module_name in supporting_modules)

        frequency_score = float((custom_result.get('frequency') or {}).get('ai_score', 0.0))

        if non_frequency_support or forensics_only_ai_prob >= 0.30:
            return final_ai_prob, False, ""

        # Keep very strong ViT+artifact evidence untouched.
        if vit_prob >= 0.90 and frequency_score >= 0.45:
            return final_ai_prob, False, ""

        if negative_modules and forensics_only_ai_prob <= 0.18:
            guard_real_prob = float(np.clip(
                (0.58 * res_prob) +
                (0.22 * legacy_ai_prob) +
                (0.20 * forensics_only_ai_prob),
                0.0,
                1.0,
            ))
            if guard_real_prob < final_ai_prob:
                return guard_real_prob, True, 'Low-support cross-model disagreement guard applied.'

        return final_ai_prob, False, ""

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

    def _resolve_ai_threshold(
        self,
        res_prob: float,
        vit_prob: Optional[float],
        legacy_ai_prob: float,
        custom_result: dict,
        final_ai_prob: float,
    ) -> float:
        support_count = len(custom_result.get('supporting_modules', []))
        negative_count = len(custom_result.get('negative_modules', []))
        forensics_only_ai_prob = float(custom_result.get('forensics_only_ai_prob', 0.0))
        frequency_score = float((custom_result.get('frequency') or {}).get('ai_score', 0.0))
        entropy_score = float((custom_result.get('entropy') or {}).get('ai_score', 0.0))
        anatomy_available = bool((custom_result.get('anatomy') or {}).get('available', False))

        threshold: float
        if support_count >= 2:
            threshold = 0.54 if vit_prob is not None else 0.57
        elif support_count == 1:
            if forensics_only_ai_prob >= 0.55:
                threshold = 0.58 if vit_prob is not None else 0.62
            else:
                threshold = 0.62 if vit_prob is not None else 0.66
        else:
            # No positive forensic support.
            # Keep thresholds conservative, but avoid forcing nearly-all fallback cases to "Real".
            if negative_count >= 2 or forensics_only_ai_prob <= 0.12:
                threshold = 0.76 if vit_prob is not None else 0.80
            elif negative_count == 1 or forensics_only_ai_prob <= 0.22:
                threshold = 0.72 if vit_prob is not None else 0.76
            elif forensics_only_ai_prob <= 0.35:
                threshold = 0.68 if vit_prob is not None else 0.72
            elif final_ai_prob >= 0.90:
                threshold = 0.60 if vit_prob is not None else 0.64
            else:
                threshold = 0.64 if vit_prob is not None else 0.68

        if vit_prob is not None:
            disagreement = abs(float(vit_prob) - float(res_prob))
            if disagreement >= 0.55 and support_count == 0 and forensics_only_ai_prob <= 0.25:
                threshold = max(threshold, 0.72)
            elif disagreement >= 0.45 and support_count == 0 and forensics_only_ai_prob <= 0.20:
                threshold = max(threshold, 0.70)

            # If anatomy validation is unavailable and corroborating forensic support is weak,
            # require a materially stronger model score before calling the image AI-generated.
            low_corroboration_guard_active = (
                support_count == 0 and
                not anatomy_available and
                negative_count >= 1 and
                forensics_only_ai_prob <= 0.35 and
                frequency_score < 0.45 and
                entropy_score <= 0.20
            )

            if (
                low_corroboration_guard_active
            ):
                threshold = max(threshold, 0.86)

            ai_consensus, _, score_margin = self._compute_score_consensus(
                res_prob=res_prob,
                vit_prob=vit_prob,
                legacy_ai_prob=legacy_ai_prob,
            )

            if support_count == 0 and forensics_only_ai_prob <= 0.22:
                if ai_consensus <= 0.40:
                    threshold = max(threshold, 0.74)
                elif ai_consensus >= 0.76 and negative_count >= 1:
                    threshold = max(threshold, 0.74)

            if score_margin >= 0.32 and float(vit_prob) >= 0.85:
                # Lowering threshold is only safe when there is corroborating forensic support.
                if low_corroboration_guard_active:
                    threshold = max(threshold, 0.86)
                elif support_count >= 1 or forensics_only_ai_prob >= 0.30:
                    threshold = min(threshold, 0.66)
                elif support_count == 0 and (negative_count >= 1 or forensics_only_ai_prob <= 0.22):
                    threshold = max(threshold, 0.74)
            elif score_margin <= -0.24 and negative_count >= 1:
                threshold = max(threshold, 0.74)

        return float(threshold)

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

        final_ai_prob, _, _ = self._apply_vit_disagreement_guard(
            res_prob=res_prob,
            vit_prob=vit_prob,
            legacy_ai_prob=legacy_ai_prob,
            custom_result=custom_result,
            final_ai_prob=final_ai_prob,
        )

        final_ai_prob, _, _ = self._apply_real_disagreement_guard(
            res_prob=res_prob,
            vit_prob=vit_prob,
            legacy_ai_prob=legacy_ai_prob,
            custom_result=custom_result,
            final_ai_prob=final_ai_prob,
        )

        final_ai_prob, _, _, _, _ = self._refine_with_score_consensus(
            res_prob=res_prob,
            vit_prob=vit_prob,
            legacy_ai_prob=legacy_ai_prob,
            custom_result=custom_result,
            final_ai_prob=final_ai_prob,
        )

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

            final_ai_prob, vit_guard_applied, vit_guard_note = self._apply_vit_disagreement_guard(
                res_prob=res_prob,
                vit_prob=vit_prob,
                legacy_ai_prob=legacy_ai_prob,
                custom_result=custom_result,
                final_ai_prob=final_ai_prob,
            )
            final_ai_prob, real_guard_applied, real_guard_note = self._apply_real_disagreement_guard(
                res_prob=res_prob,
                vit_prob=vit_prob,
                legacy_ai_prob=legacy_ai_prob,
                custom_result=custom_result,
                final_ai_prob=final_ai_prob,
            )
            final_ai_prob, score_consensus_ai, score_consensus_real, score_margin, score_refinement_applied = self._refine_with_score_consensus(
                res_prob=res_prob,
                vit_prob=vit_prob,
                legacy_ai_prob=legacy_ai_prob,
                custom_result=custom_result,
                final_ai_prob=final_ai_prob,
            )
            if vit_guard_applied:
                decision_source = f"{decision_source} + ViT disagreement guard"
                fusion_info['fusion_note'] = f"{fusion_info.get('fusion_note', 'Model fusion applied.')} {vit_guard_note}"
            if real_guard_applied:
                decision_source = f"{decision_source} + real disagreement guard"
                fusion_info['fusion_note'] = f"{fusion_info.get('fusion_note', 'Model fusion applied.')} {real_guard_note}"
            if score_refinement_applied:
                decision_source = f"{decision_source} + score-consensus calibration"
                fusion_info['fusion_note'] = (
                    f"{fusion_info.get('fusion_note', 'Model fusion applied.')} "
                    "Score-consensus calibration adjusted the final probability."
                )
        except Exception as exc:
            if return_details:
                return {'status': 'error', 'error': f"Model inference failed: {exc}"}
            return "Error", f"Model inference failed: {exc}"

        ai_threshold = self._resolve_ai_threshold(
            res_prob=res_prob,
            vit_prob=vit_prob,
            legacy_ai_prob=legacy_ai_prob,
            custom_result=custom_result,
            final_ai_prob=final_ai_prob,
        )
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
                    'vit_guard_applied': bool(vit_guard_applied),
                    'real_guard_applied': bool(real_guard_applied),
                    'score_consensus_ai_score': score_consensus_ai * 100.0,
                    'score_consensus_real_score': score_consensus_real * 100.0,
                    'score_consensus_margin': score_margin * 100.0,
                    'score_refinement_applied': bool(score_refinement_applied),
                    'resnet_vit_disagreement': None if vit_prob is None else abs(vit_prob - res_prob) * 100.0,
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
import math
import os
from .robust_extractor import RobustSynthIDExtractor


class SynthIDService:
    def __init__(self, codebook_path: str):
        self.codebook_path = codebook_path
        self.extractor = None
        self.available = False
        self.error = None
        self.min_confidence = self._env_float("SYNTHID_MIN_CONFIDENCE", 0.58)
        self.min_phase_match = self._env_float("SYNTHID_MIN_PHASE_MATCH", 0.48)
        self.min_correlation_margin = self._env_float("SYNTHID_MIN_CORRELATION_MARGIN", 0.0018)
        self.min_carrier_match_ratio = self._env_float("SYNTHID_MIN_CARRIER_MATCH_RATIO", 0.30)
        self.min_signal_score = self._env_float("SYNTHID_MIN_SIGNAL_SCORE", 0.56)
        self.min_structure_ratio = self._env_float("SYNTHID_MIN_STRUCTURE_RATIO", 0.78)
        self.max_structure_ratio = self._env_float("SYNTHID_MAX_STRUCTURE_RATIO", 1.85)
        self.max_multi_scale_std = self._env_float("SYNTHID_MAX_MULTI_SCALE_STD", 0.18)

        try:
            if not os.path.exists(codebook_path):
                raise FileNotFoundError(f"SynthID codebook not found: {codebook_path}")

            self.extractor = RobustSynthIDExtractor(codebook_path=codebook_path)
            self.available = True

        except Exception as exc:
            self.available = False
            self.error = str(exc)

    def analyze(self, image_path: str) -> dict:
        if not self.available:
            return {
                "status": "unavailable",
                "available": False,
                "message": "SynthID detector unavailable",
                "error": self.error,
            }

        if not os.path.exists(image_path):
            return {
                "status": "error",
                "available": True,
                "message": "Image path could not be analyzed",
                "error": f"Image not found: {image_path}",
            }

        try:
            result = self.extractor.detect(image_path)
            details = getattr(result, "details", {}) or {}
            raw_confidence = self._safe_float(result.confidence)
            confidence = raw_confidence * 100.0
            correlation = self._safe_float(result.correlation)
            phase_match = self._safe_float(result.phase_match)
            structure_ratio = self._safe_float(result.structure_ratio)
            multi_scale_consistency = self._safe_float(result.multi_scale_consistency)
            carrier_match_ratio = self._safe_float(details.get("carrier_match_ratio"), 0.0)
            threshold = self._safe_float(details.get("threshold"), float("nan"))
            correlation_margin = correlation - threshold if math.isfinite(threshold) else 0.0
            signal_score = self._compose_signal_score(
                raw_confidence=raw_confidence,
                phase_match=phase_match,
                correlation_margin=correlation_margin,
                structure_ratio=structure_ratio,
                multi_scale_consistency=multi_scale_consistency,
                carrier_match_ratio=carrier_match_ratio,
            )

            raw_is_watermarked = bool(result.is_watermarked)
            calibrated_is_watermarked = self._is_calibrated_positive(
                raw_is_watermarked=raw_is_watermarked,
                raw_confidence=raw_confidence,
                phase_match=phase_match,
                structure_ratio=structure_ratio,
                correlation_margin=correlation_margin,
                multi_scale_consistency=multi_scale_consistency,
                carrier_match_ratio=carrier_match_ratio,
                signal_score=signal_score,
            )

            if calibrated_is_watermarked:
                signal_band = "strong"
            elif signal_score >= (self.min_signal_score * 0.75):
                signal_band = "weak"
            else:
                signal_band = "negative"

            if calibrated_is_watermarked:
                message = "SynthID watermark detected (calibrated high-confidence match)"
            elif raw_is_watermarked or signal_band == "weak":
                message = "Watermark-like SynthID signal detected, but below calibrated confidence threshold"
            else:
                message = "No SynthID watermark detected"

            return {
                "status": "complete",
                "available": True,
                "is_watermarked": calibrated_is_watermarked,
                "is_watermarked_raw": raw_is_watermarked,
                "message": message,
                "confidence": confidence,
                "raw_confidence": raw_confidence,
                "correlation": correlation,
                "phase_match": phase_match,
                "structure_ratio": structure_ratio,
                "carrier_strength": self._safe_float(result.carrier_strength),
                "carrier_match_ratio": carrier_match_ratio,
                "multi_scale_consistency": multi_scale_consistency,
                "correlation_margin": correlation_margin,
                "signal_score": signal_score,
                "signal_band": signal_band,
                "threshold": None if not math.isfinite(threshold) else threshold,
                "calibration": {
                    "applied": True,
                    "min_confidence": self.min_confidence,
                    "min_phase_match": self.min_phase_match,
                    "min_correlation_margin": self.min_correlation_margin,
                    "min_carrier_match_ratio": self.min_carrier_match_ratio,
                    "min_signal_score": self.min_signal_score,
                    "structure_ratio_range": [self.min_structure_ratio, self.max_structure_ratio],
                    "max_multi_scale_std": self.max_multi_scale_std,
                },
                "source": "reverse-SynthID robust extractor",
            }

        except Exception as exc:
            return {
                "status": "error",
                "available": True,
                "message": "SynthID analysis failed",
                "error": str(exc),
            }

    def health(self) -> dict:
        return {
            "available": self.available,
            "codebook_path": self.codebook_path,
            "error": self.error,
        }

    def _is_calibrated_positive(
        self,
        *,
        raw_is_watermarked: bool,
        raw_confidence: float,
        phase_match: float,
        structure_ratio: float,
        correlation_margin: float,
        multi_scale_consistency: float,
        carrier_match_ratio: float,
        signal_score: float,
    ) -> bool:
        if not (self.min_structure_ratio <= structure_ratio <= self.max_structure_ratio):
            return False

        # If the raw detector fired, allow a slightly relaxed path so we do not
        # suppress true positives due to calibration drift across environments.
        if raw_is_watermarked:
            if signal_score < max(0.52, self.min_signal_score * 0.90):
                return False
            if phase_match < max(0.42, self.min_phase_match * 0.92):
                return False
            if correlation_margin < (self.min_correlation_margin * 0.70):
                return False
            if carrier_match_ratio < (self.min_carrier_match_ratio * 0.85):
                return False
            if multi_scale_consistency > (self.max_multi_scale_std * 1.15):
                return False
            return True

        if signal_score < self.min_signal_score:
            return False
        if raw_confidence < self.min_confidence:
            return False
        if phase_match < self.min_phase_match:
            return False
        if correlation_margin < self.min_correlation_margin:
            return False
        if carrier_match_ratio < self.min_carrier_match_ratio:
            return False
        if multi_scale_consistency > self.max_multi_scale_std:
            return False

        return True

    @staticmethod
    def _compose_signal_score(
        *,
        raw_confidence: float,
        phase_match: float,
        correlation_margin: float,
        structure_ratio: float,
        multi_scale_consistency: float,
        carrier_match_ratio: float,
    ) -> float:
        def clip01(value: float) -> float:
            return float(max(0.0, min(1.0, value)))

        confidence_score = clip01((raw_confidence - 0.45) / 0.50)
        phase_score = clip01((phase_match - 0.42) / 0.33)
        corr_score = clip01((correlation_margin + 0.0025) / 0.015)
        carrier_score = clip01((carrier_match_ratio - 0.20) / 0.45)
        structure_score = clip01(1.0 - abs(structure_ratio - 1.32) / 0.75)
        consistency_score = clip01(1.0 - (multi_scale_consistency / 0.20))

        return clip01(
            (0.26 * confidence_score) +
            (0.22 * phase_score) +
            (0.20 * corr_score) +
            (0.16 * carrier_score) +
            (0.10 * structure_score) +
            (0.06 * consistency_score)
        )

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        value = os.environ.get(name)
        if value is None:
            return float(default)

        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _safe_float(value: float, default: float = 0.0) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default

        if not math.isfinite(value):
            return default

        return value
import math
import os
from .robust_extractor import RobustSynthIDExtractor


class SynthIDService:
    def __init__(self, codebook_path: str):
        self.codebook_path = codebook_path
        self.extractor = None
        self.available = False
        self.error = None

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
            confidence = self._safe_float(result.confidence) * 100.0

            return {
                "status": "complete",
                "available": True,
                "is_watermarked": bool(result.is_watermarked),
                "message": "SynthID watermark detected" if result.is_watermarked else "No SynthID watermark detected",
                "confidence": confidence,
                "raw_confidence": self._safe_float(result.confidence),
                "correlation": self._safe_float(result.correlation),
                "phase_match": self._safe_float(result.phase_match),
                "structure_ratio": self._safe_float(result.structure_ratio),
                "carrier_strength": self._safe_float(result.carrier_strength),
                "multi_scale_consistency": self._safe_float(result.multi_scale_consistency),
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

    @staticmethod
    def _safe_float(value: float, default: float = 0.0) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default

        if not math.isfinite(value):
            return default

        return value

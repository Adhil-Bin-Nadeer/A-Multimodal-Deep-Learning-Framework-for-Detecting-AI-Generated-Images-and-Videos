"""
SynthID Presence Checker

A standalone module to detect if SynthID watermark is present in images.
This module can be independently copied to other projects.

Features:
- Fast SynthID detection using pre-extracted codebook
- Confidence score for detection
- No heavy model dependencies
- Works with various image formats (PNG, JPG, JPEG, WEBP)

Usage:
    from synthid_checker import SynthIDChecker
    
    checker = SynthIDChecker(codebook_path="path/to/robust_codebook.pkl")
    result = checker.check_image("image.png")
    
    if result['is_synthid_present']:
        print(f"SynthID detected with {result['confidence']:.2%} confidence")
"""

import os
import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift
from scipy import ndimage
from typing import Dict, Optional
import pickle
import pywt


class SynthIDChecker:
    """
    Standalone SynthID watermark detector.
    
    Checks if an image contains Google's SynthID watermark using
    frequency domain analysis and multi-scale consistency checks.
    """
    
    def __init__(self, codebook_path: str):
        """
        Initialize the SynthID checker with a codebook.
        
        Args:
            codebook_path: Path to the pre-extracted codebook pickle file
            
        Raises:
            FileNotFoundError: If codebook doesn't exist
            ValueError: If codebook is invalid
        """
        if not os.path.exists(codebook_path):
            raise FileNotFoundError(f"Codebook not found: {codebook_path}")
        
        self.codebook_path = codebook_path
        self.codebook = None
        self.available = False
        self.error = None
        
        try:
            with open(codebook_path, 'rb') as f:
                self.codebook = pickle.load(f)
            self.available = True
        except Exception as e:
            self.error = str(e)
            self.available = False
    
    def check_image(self, image_path: str) -> Dict:
        """
        Check if an image contains SynthID watermark.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with detection results:
            {
                'is_synthid_present': bool,
                'confidence': float (0.0 to 1.0),
                'correlation': float,
                'phase_match': float,
                'carrier_strength': float,
                'structure_ratio': float,
                'status': 'detected' | 'not_detected' | 'error',
                'message': str
            }
        """
        if not self.available:
            return {
                'is_synthid_present': False,
                'confidence': 0.0,
                'status': 'error',
                'message': f'SynthID checker unavailable: {self.error}',
                'error': self.error
            }
        
        if not os.path.exists(image_path):
            return {
                'is_synthid_present': False,
                'confidence': 0.0,
                'status': 'error',
                'message': f'Image file not found: {image_path}',
                'error': 'File not found'
            }
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {
                    'is_synthid_present': False,
                    'confidence': 0.0,
                    'status': 'error',
                    'message': f'Could not load image: {image_path}',
                    'error': 'Invalid image format'
                }
            
            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return self._detect_watermark(img)
        
        except Exception as e:
            return {
                'is_synthid_present': False,
                'confidence': 0.0,
                'status': 'error',
                'message': f'Detection error: {str(e)}',
                'error': str(e)
            }
    
    def check_array(self, image: np.ndarray) -> Dict:
        """
        Check if a numpy array image contains SynthID watermark.
        
        Args:
            image: Numpy array (H, W, 3) in RGB format, values 0-255
            
        Returns:
            Dictionary with detection results (same format as check_image)
        """
        if not self.available:
            return {
                'is_synthid_present': False,
                'confidence': 0.0,
                'status': 'error',
                'message': f'SynthID checker unavailable: {self.error}',
                'error': self.error
            }
        
        try:
            return self._detect_watermark(image)
        except Exception as e:
            return {
                'is_synthid_present': False,
                'confidence': 0.0,
                'status': 'error',
                'message': f'Detection error: {str(e)}',
                'error': str(e)
            }
    
    def _detect_watermark(self, image: np.ndarray) -> Dict:
        """
        Internal method to perform watermark detection.
        
        Uses multi-method analysis:
        1. Correlation with reference noise pattern
        2. Carrier frequency analysis
        3. Noise structure ratio
        4. Multi-scale consistency
        """
        target_size = self.codebook['image_size']
        
        # Resize image
        img_resized = cv2.resize(image, (target_size, target_size))
        
        # Extract noise signature
        noise = self._extract_noise_fused(img_resized)
        
        # Method 1: Correlation with reference noise
        ref_noise = self.codebook['reference_noise']
        correlation = float(np.corrcoef(noise.ravel(), ref_noise.ravel())[0, 1])
        
        # Method 2: Carrier frequency analysis
        gray = np.mean(img_resized, axis=2) if len(img_resized.shape) == 3 else img_resized
        gray = gray.astype(np.float32)
        f = fftshift(fft2(gray))
        magnitude = np.abs(f)
        phase = np.angle(f)
        
        center = target_size // 2
        carrier_scores = []
        carrier_strengths = []
        
        # Use carriers from codebook
        carriers_to_check = self.codebook.get('carriers', [])[:30]
        known_carriers = self.codebook.get('known_carriers', [])
        
        ref_phase = self.codebook.get('reference_phase')
        
        for carrier in carriers_to_check:
            freq = carrier['frequency']
            y = freq[0] + center
            x = freq[1] + center
            
            if 0 <= y < target_size and 0 <= x < target_size:
                actual_phase = phase[y, x]
                expected_phase = ref_phase[y, x] if ref_phase is not None else carrier.get('phase', 0)
                
                phase_diff = np.abs(np.angle(np.exp(1j * (actual_phase - expected_phase))))
                phase_match = 1 - phase_diff / np.pi
                carrier_scores.append(phase_match)
                carrier_strengths.append(magnitude[y, x])
        
        # Also check known carriers
        for freq in known_carriers:
            y = freq[0] + center
            x = freq[1] + center
            
            if 0 <= y < target_size and 0 <= x < target_size:
                actual_phase = phase[y, x]
                expected_phase = ref_phase[y, x] if ref_phase is not None else 0
                
                phase_diff = np.abs(np.angle(np.exp(1j * (actual_phase - expected_phase))))
                phase_match = 1 - phase_diff / np.pi
                carrier_scores.append(phase_match)
                carrier_strengths.append(magnitude[y, x])
        
        avg_phase_match = float(np.mean(carrier_scores)) if carrier_scores else 0
        avg_carrier_strength = float(np.mean(carrier_strengths)) if carrier_strengths else 0
        
        # Method 3: Noise structure ratio
        noise_gray = np.mean(noise, axis=2) if len(noise.shape) == 3 else noise
        structure_ratio = float(np.std(noise_gray) / (np.mean(np.abs(noise_gray)) + 1e-10))
        
        # Method 4: Multi-scale consistency
        scale_scores = []
        scales = self.codebook.get('scales_used', [256, 512, 1024])
        
        for scale in scales:
            img_scaled = cv2.resize(image, (scale, scale))
            noise_scaled = self._extract_noise_single(img_scaled, 'wavelet')
            ref_scaled = cv2.resize(ref_noise, (scale, scale))
            
            # Handle NaN values
            scale_corr = np.corrcoef(noise_scaled.ravel(), ref_scaled.ravel())[0, 1]
            if not np.isnan(scale_corr):
                scale_scores.append(scale_corr)
        
        multi_scale_consistency = float(np.std(scale_scores)) if scale_scores else 0
        
        # Detection decision
        threshold = self.codebook.get('detection_threshold', 0.21)
        is_watermarked = (
            correlation > threshold and
            avg_phase_match > 0.45 and
            0.7 < structure_ratio < 2.0
        )
        
        # Confidence score calculation
        corr_mean = self.codebook.get('correlation_mean', 0.25)
        corr_score = max(0, (correlation - threshold) / (corr_mean - threshold + 1e-10))
        phase_score = avg_phase_match
        structure_score = max(0, 1 - abs(structure_ratio - 1.32) / 0.6)
        consistency_score = max(0, 1 - multi_scale_consistency * 5)
        
        confidence = min(1.0, (
            0.35 * corr_score +
            0.35 * phase_score +
            0.15 * structure_score +
            0.15 * consistency_score
        ))
        
        return {
            'is_synthid_present': bool(is_watermarked),
            'confidence': float(confidence),
            'correlation': correlation,
            'phase_match': avg_phase_match,
            'carrier_strength': avg_carrier_strength,
            'structure_ratio': structure_ratio,
            'status': 'detected' if is_watermarked else 'not_detected',
            'message': 'SynthID watermark detected' if is_watermarked else 'No SynthID watermark detected'
        }
    
    def _extract_noise_single(self, image: np.ndarray, method: str = 'wavelet') -> np.ndarray:
        """Extract noise using a single denoising method."""
        img_f = image.astype(np.float32)
        if img_f.max() > 1:
            img_f = img_f / 255.0
        
        if method == 'wavelet':
            if len(img_f.shape) == 2:
                denoised = self._wavelet_denoise(img_f)
            else:
                denoised = np.zeros_like(img_f)
                for c in range(img_f.shape[2]):
                    denoised[:, :, c] = self._wavelet_denoise(img_f[:, :, c])
        
        elif method == 'bilateral':
            denoised = self._bilateral_denoise(img_f)
        
        elif method == 'nlm':
            denoised = self._nlm_denoise(img_f)
        
        elif method == 'wiener':
            if len(img_f.shape) == 2:
                denoised = self._wiener_filter(img_f)
            else:
                denoised = np.zeros_like(img_f)
                for c in range(img_f.shape[2]):
                    denoised[:, :, c] = self._wiener_filter(img_f[:, :, c])
        else:
            raise ValueError(f"Unknown denoising method: {method}")
        
        return img_f - denoised
    
    def _extract_noise_fused(self, image: np.ndarray) -> np.ndarray:
        """Extract noise using multiple methods and fuse results."""
        noises = []
        weights = []
        
        # Wavelet denoising with multiple families
        for wavelet in ['db4', 'sym8', 'coif3']:
            try:
                img_f = image.astype(np.float32)
                if img_f.max() > 1:
                    img_f = img_f / 255.0
                
                if len(img_f.shape) == 2:
                    denoised = self._wavelet_denoise(img_f, wavelet)
                else:
                    denoised = np.zeros_like(img_f)
                    for c in range(img_f.shape[2]):
                        denoised[:, :, c] = self._wavelet_denoise(img_f[:, :, c], wavelet)
                
                noise = img_f - denoised
                noises.append(noise)
                weights.append(1.0)
            except:
                pass
        
        # Bilateral filter
        try:
            noise = self._extract_noise_single(image, 'bilateral')
            noises.append(noise)
            weights.append(0.8)
        except:
            pass
        
        # Non-local means
        try:
            noise = self._extract_noise_single(image, 'nlm')
            noises.append(noise)
            weights.append(0.7)
        except:
            pass
        
        # Wiener filter
        try:
            noise = self._extract_noise_single(image, 'wiener')
            noises.append(noise)
            weights.append(0.6)
        except:
            pass
        
        if not noises:
            # Fallback: simple noise extraction
            return self._extract_noise_single(image, 'wavelet')
        
        noises = np.array(noises)
        weights = np.array(weights) / sum(weights)
        fused = np.tensordot(weights, noises, axes=([0], [0]))
        
        return fused
    
    def _wavelet_denoise(self, channel: np.ndarray, wavelet: str = 'db4', level: int = 3) -> np.ndarray:
        """Wavelet-based denoising using soft thresholding."""
        try:
            coeffs = pywt.wavedec2(channel, wavelet, level=level)
            detail = coeffs[-1][0]
            sigma = np.median(np.abs(detail)) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(channel.size))
            
            new_coeffs = [coeffs[0]]
            for details in coeffs[1:]:
                new_details = tuple(
                    pywt.threshold(d, threshold, mode='soft') for d in details
                )
                new_coeffs.append(new_details)
            
            denoised = pywt.waverec2(new_coeffs, wavelet)
            result = denoised[:channel.shape[0], :channel.shape[1]]
            
            return result if result is not None else channel
        except:
            return channel
    
    def _bilateral_denoise(self, image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """Bilateral filter denoising (edge-preserving)."""
        try:
            if len(image.shape) == 2:
                return cv2.bilateralFilter(image.astype(np.float32), d, sigma_color, sigma_space)
            else:
                result = np.zeros_like(image)
                for c in range(image.shape[2]):
                    result[:, :, c] = cv2.bilateralFilter(
                        image[:, :, c].astype(np.float32), d, sigma_color, sigma_space
                    )
                return result
        except:
            return image
    
    def _nlm_denoise(self, image: np.ndarray, h: float = 10, template_size: int = 7, search_size: int = 21) -> np.ndarray:
        """Non-local means denoising."""
        try:
            img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
            
            if len(image.shape) == 2:
                denoised = cv2.fastNlMeansDenoising(
                    img_uint8, None, int(h), template_size, search_size
                )
            else:
                denoised = cv2.fastNlMeansDenoisingColored(
                    img_uint8, None, int(h), int(h), template_size, search_size
                )
            
            return denoised.astype(np.float32) / 255.0
        except:
            return image
    
    def _wiener_filter(self, image: np.ndarray, noise_variance: Optional[float] = None) -> np.ndarray:
        """Wiener filter for optimal noise estimation."""
        try:
            if noise_variance is None:
                noise_variance = np.var(image - ndimage.gaussian_filter(image, sigma=2))
            
            f = fft2(image)
            power = np.abs(f) ** 2
            signal_power = np.maximum(power - noise_variance, 0)
            wiener_ratio = signal_power / (signal_power + noise_variance + 1e-10)
            
            denoised = np.real(ifft2(f * wiener_ratio))
            return denoised
        except:
            return image


def is_synthid_present(image_path: str, codebook_path: str) -> bool:
    """
    Quick utility function to check if SynthID is present in an image.
    
    Args:
        image_path: Path to image file
        codebook_path: Path to codebook pickle
        
    Returns:
        True if SynthID detected, False otherwise
    """
    checker = SynthIDChecker(codebook_path)
    result = checker.check_image(image_path)
    return result['is_synthid_present']


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='SynthID Watermark Checker')
    parser.add_argument('image', type=str, help='Image to check')
    parser.add_argument('--codebook', type=str, required=True, help='Path to codebook')
    parser.add_argument('--verbose', action='store_true', help='Print detailed results')
    
    args = parser.parse_args()
    
    checker = SynthIDChecker(args.codebook)
    result = checker.check_image(args.image)
    
    if args.verbose:
        print("\n" + "=" * 50)
        print("SYNTHID WATERMARK CHECK")
        print("=" * 50)
        for key, value in result.items():
            print(f"  {key}: {value}")
        print("=" * 50)
    else:
        print(f"SynthID Present: {result['is_synthid_present']}")
        print(f"Confidence: {result['confidence']:.2%}")

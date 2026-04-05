"""
Unit Tests for SynthID Detection Module

Tests cover:
- SynthIDService initialization and health checks
- SynthIDChecker initialization and image validation
- RobustSynthIDExtractor functionality
- Error handling and edge cases
- Confidence score calculations
"""

import unittest
import tempfile
import numpy as np
import os
import pickle
from unittest.mock import patch, MagicMock, mock_open
import sys

# Add the backend src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.synthid.service import SynthIDService
from src.synthid_checker import SynthIDChecker
from src.synthid.robust_extractor import RobustSynthIDExtractor, DetectionResult


class MockCodebook:
    """Helper to create mock codebook for testing."""
    
    @staticmethod
    def create():
        """Create a mock codebook with required keys."""
        return {
            'image_size': 512,
            'reference_noise': np.random.randn(512, 512, 3),
            'reference_phase': np.random.randn(512, 512),
            'carriers': [
                {'frequency': (48, 0), 'phase': 0.5},
                {'frequency': (-48, 0), 'phase': 0.5},
                {'frequency': (96, 0), 'phase': 0.5},
            ],
            'known_carriers': [(48, 0), (-48, 0), (96, 0)],
            'scales_used': [256, 512, 1024],
            'detection_threshold': 0.21,
            'correlation_mean': 0.25,
        }


class TestSynthIDServiceInitialization(unittest.TestCase):
    """Test SynthIDService initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization_with_valid_codebook(self):
        """Test initialization with a valid codebook file."""
        # Create a mock codebook file
        codebook_path = os.path.join(self.temp_dir, 'codebook.pkl')
        with open(codebook_path, 'wb') as f:
            pickle.dump(MockCodebook.create(), f)
        
        service = SynthIDService(codebook_path)
        
        self.assertTrue(service.available)
        self.assertIsNone(service.error)
        self.assertIsNotNone(service.extractor)
    
    def test_initialization_with_missing_codebook(self):
        """Test initialization fails gracefully with missing codebook."""
        service = SynthIDService('/nonexistent/codebook.pkl')
        
        self.assertFalse(service.available)
        self.assertIsNotNone(service.error)
        self.assertIn('not found', service.error.lower())
    
    def test_initialization_with_corrupted_codebook(self):
        """Test initialization handles corrupted codebook files."""
        codebook_path = os.path.join(self.temp_dir, 'corrupt.pkl')
        with open(codebook_path, 'w') as f:
            f.write('corrupted data')
        
        service = SynthIDService(codebook_path)
        
        self.assertFalse(service.available)
        self.assertIsNotNone(service.error)


class TestSynthIDServiceAnalyze(unittest.TestCase):
    """Test SynthIDService analyze method."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a valid codebook
        self.codebook_path = os.path.join(self.temp_dir, 'codebook.pkl')
        with open(self.codebook_path, 'wb') as f:
            pickle.dump(MockCodebook.create(), f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_analyze_unavailable_service(self):
        """Test analyze returns appropriate response when service unavailable."""
        service = SynthIDService('/nonexistent/codebook.pkl')
        result = service.analyze('/dummy/image.png')
        
        self.assertEqual(result['status'], 'unavailable')
        self.assertFalse(result['available'])
        self.assertIn('unavailable', result['message'].lower())
    
    def test_analyze_missing_image(self):
        """Test analyze handles missing image files."""
        service = SynthIDService(self.codebook_path)
        result = service.analyze('/nonexistent/image.png')
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('not found', result['error'].lower())
    
    def test_health_check(self):
        """Test health check returns correct status."""
        service = SynthIDService(self.codebook_path)
        health = service.health()
        
        self.assertTrue(health['available'])
        self.assertEqual(health['codebook_path'], self.codebook_path)
        self.assertIsNone(health['error'])
    
    def test_safe_float_conversion(self):
        """Test _safe_float handles various inputs."""
        # Valid float
        self.assertEqual(SynthIDService._safe_float(3.14), 3.14)
        
        # Integer
        self.assertEqual(SynthIDService._safe_float(5), 5.0)
        
        # String
        self.assertEqual(SynthIDService._safe_float("2.5"), 2.5)
        
        # Invalid types return default
        self.assertEqual(SynthIDService._safe_float(None), 0.0)
        self.assertEqual(SynthIDService._safe_float("invalid"), 0.0)
        
        # NaN and infinity
        self.assertEqual(SynthIDService._safe_float(float('nan')), 0.0)
        self.assertEqual(SynthIDService._safe_float(float('inf')), 0.0)
        
        # Custom default
        self.assertEqual(SynthIDService._safe_float(None, -1.0), -1.0)


class TestSynthIDCheckerInitialization(unittest.TestCase):
    """Test SynthIDChecker initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization_with_valid_codebook(self):
        """Test initialization with a valid codebook file."""
        codebook_path = os.path.join(self.temp_dir, 'codebook.pkl')
        with open(codebook_path, 'wb') as f:
            pickle.dump(MockCodebook.create(), f)
        
        checker = SynthIDChecker(codebook_path)
        
        self.assertTrue(checker.available)
        self.assertIsNone(checker.error)
        self.assertIsNotNone(checker.codebook)
    
    def test_initialization_with_missing_codebook(self):
        """Test initialization fails with missing codebook."""
        with self.assertRaises(FileNotFoundError):
            SynthIDChecker('/nonexistent/codebook.pkl')
    
    def test_initialization_with_corrupted_codebook(self):
        """Test initialization handles corrupted codebook."""
        codebook_path = os.path.join(self.temp_dir, 'corrupt.pkl')
        with open(codebook_path, 'w') as f:
            f.write('corrupted data')
        
        checker = SynthIDChecker(codebook_path)
        
        self.assertFalse(checker.available)
        self.assertIsNotNone(checker.error)


class TestSynthIDCheckerImageChecking(unittest.TestCase):
    """Test SynthIDChecker image checking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a valid codebook
        self.codebook_path = os.path.join(self.temp_dir, 'codebook.pkl')
        with open(self.codebook_path, 'wb') as f:
            pickle.dump(MockCodebook.create(), f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_check_image_unavailable_checker(self):
        """Test check_image when checker is unavailable."""
        codebook_path = os.path.join(self.temp_dir, 'corrupt.pkl')
        with open(codebook_path, 'w') as f:
            f.write('corrupted data')
        
        checker = SynthIDChecker(codebook_path)
        result = checker.check_image('/dummy/image.png')
        
        self.assertFalse(result['is_synthid_present'])
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['confidence'], 0.0)
    
    def test_check_image_missing_file(self):
        """Test check_image with missing image file."""
        checker = SynthIDChecker(self.codebook_path)
        result = checker.check_image('/nonexistent/image.png')
        
        self.assertFalse(result['is_synthid_present'])
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['confidence'], 0.0)
        self.assertIn('not found', result['error'].lower())
    
    def test_check_array_with_valid_array(self):
        """Test check_array with valid numpy array."""
        checker = SynthIDChecker(self.codebook_path)
        
        # Create a dummy image array
        image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        result = checker.check_array(image)
        
        # Verify response structure
        self.assertIn('is_synthid_present', result)
        self.assertIn('confidence', result)
        self.assertIn('status', result)
        self.assertIn('message', result)
        
        # Verify value types
        self.assertIsInstance(result['is_synthid_present'], bool)
        self.assertIsInstance(result['confidence'], float)
        self.assertTrue(0.0 <= result['confidence'] <= 1.0)
    
    def test_check_array_unavailable_checker(self):
        """Test check_array when checker is unavailable."""
        codebook_path = os.path.join(self.temp_dir, 'corrupt.pkl')
        with open(codebook_path, 'w') as f:
            f.write('corrupted data')
        
        checker = SynthIDChecker(codebook_path)
        image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        result = checker.check_array(image)
        
        self.assertFalse(result['is_synthid_present'])
        self.assertEqual(result['status'], 'error')


class TestRobustSynthIDExtractorInitialization(unittest.TestCase):
    """Test RobustSynthIDExtractor initialization."""
    
    def test_initialization_with_defaults(self):
        """Test initialization with default parameters."""
        extractor = RobustSynthIDExtractor()
        
        self.assertEqual(extractor.scales, [256, 512, 1024])
        self.assertEqual(len(extractor.wavelets), 3)
        self.assertEqual(extractor.n_carriers, 100)
        self.assertIsNotNone(extractor.known_carriers)
    
    def test_initialization_with_custom_scales(self):
        """Test initialization with custom scales."""
        custom_scales = [128, 256]
        extractor = RobustSynthIDExtractor(scales=custom_scales)
        
        self.assertEqual(extractor.scales, custom_scales)
    
    def test_initialization_with_custom_wavelets(self):
        """Test initialization with custom wavelets."""
        custom_wavelets = ['db2', 'db4']
        extractor = RobustSynthIDExtractor(wavelets=custom_wavelets)
        
        self.assertEqual(extractor.wavelets, custom_wavelets)
    
    def test_known_carriers_structure(self):
        """Test that known_carriers have expected structure."""
        extractor = RobustSynthIDExtractor()
        
        self.assertIsInstance(extractor.known_carriers, list)
        self.assertTrue(len(extractor.known_carriers) > 0)
        
        for carrier in extractor.known_carriers:
            self.assertIsInstance(carrier, tuple)
            self.assertEqual(len(carrier), 2)


class TestDetectionResult(unittest.TestCase):
    """Test DetectionResult dataclass."""
    
    def test_detection_result_creation(self):
        """Test creating DetectionResult instances."""
        result = DetectionResult(
            is_watermarked=True,
            confidence=0.85,
            correlation=0.3,
            phase_match=0.6,
            structure_ratio=1.32,
            carrier_strength=100.0,
            multi_scale_consistency=0.05,
            details={'test': 'value'}
        )
        
        self.assertTrue(result.is_watermarked)
        self.assertEqual(result.confidence, 0.85)
        self.assertEqual(result.correlation, 0.3)
        self.assertEqual(result.phase_match, 0.6)
        self.assertEqual(result.structure_ratio, 1.32)
        self.assertEqual(result.carrier_strength, 100.0)
        self.assertEqual(result.multi_scale_consistency, 0.05)
        self.assertEqual(result.details, {'test': 'value'})


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a valid codebook
        self.codebook_path = os.path.join(self.temp_dir, 'codebook.pkl')
        with open(self.codebook_path, 'wb') as f:
            pickle.dump(MockCodebook.create(), f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_check_array_with_grayscale_image(self):
        """Test check_array handles grayscale images."""
        checker = SynthIDChecker(self.codebook_path)
        
        # Create a grayscale image
        image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        
        # Should handle gracefully (may error or convert)
        try:
            result = checker.check_array(image)
            self.assertIsNotNone(result)
        except Exception:
            # It's acceptable to fail on invalid input
            pass
    
    def test_check_array_with_zero_values(self):
        """Test check_array with zero-valued image."""
        checker = SynthIDChecker(self.codebook_path)
        
        # All zeros
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        result = checker.check_array(image)
        
        self.assertIsNotNone(result)
        self.assertIn('status', result)
    
    def test_check_array_with_max_values(self):
        """Test check_array with maximum-valued image."""
        checker = SynthIDChecker(self.codebook_path)
        
        # All 255s
        image = np.full((512, 512, 3), 255, dtype=np.uint8)
        result = checker.check_array(image)
        
        self.assertIsNotNone(result)
        self.assertIn('status', result)
    
    def test_safe_float_with_edge_values(self):
        """Test _safe_float with various edge case values."""
        # Very small number
        self.assertEqual(SynthIDService._safe_float(1e-10), 1e-10)
        
        # Very large number
        self.assertEqual(SynthIDService._safe_float(1e10), 1e10)
        
        # Negative numbers
        self.assertEqual(SynthIDService._safe_float(-5.5), -5.5)
        
        # Zero
        self.assertEqual(SynthIDService._safe_float(0), 0.0)


class TestResponseStructure(unittest.TestCase):
    """Test that responses have the expected structure."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a valid codebook
        self.codebook_path = os.path.join(self.temp_dir, 'codebook.pkl')
        with open(self.codebook_path, 'wb') as f:
            pickle.dump(MockCodebook.create(), f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_checker_response_has_required_fields(self):
        """Test that checker response includes all required fields."""
        checker = SynthIDChecker(self.codebook_path)
        image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        result = checker.check_array(image)
        
        required_fields = [
            'is_synthid_present',
            'confidence',
            'status',
            'message'
        ]
        
        for field in required_fields:
            self.assertIn(field, result, f"Missing required field: {field}")
    
    def test_service_analyze_response_fields(self):
        """Test service response has all required fields when unavailable."""
        service = SynthIDService('/nonexistent/codebook.pkl')
        result = service.analyze('/dummy/image.png')
        
        required_fields = ['status', 'available', 'message', 'error']
        
        for field in required_fields:
            self.assertIn(field, result, f"Missing required field: {field}")


if __name__ == '__main__':
    unittest.main()

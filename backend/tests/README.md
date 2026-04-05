# SynthID Module Unit Tests

This directory contains comprehensive unit tests for the SynthID watermark detection module.

## Test Coverage

### `test_synthid.py`

Tests for the complete SynthID detection pipeline:

#### **SynthIDService Tests**

- ✅ Initialization with valid/invalid codebooks
- ✅ Graceful handling of missing/corrupted codebooks
- ✅ Image analysis with missing files
- ✅ Health checks and status reporting
- ✅ Safe float conversion with edge cases

#### **SynthIDChecker Tests**

- ✅ Initialization and codebook loading
- ✅ Image file checking (missing files, invalid formats)
- ✅ Numpy array checking with various image types
- ✅ Error handling and graceful degradation
- ✅ Response structure validation

#### **RobustSynthIDExtractor Tests**

- ✅ Initialization with default/custom parameters
- ✅ Custom scales and wavelets configuration
- ✅ Known carriers structure validation
- ✅ Detection result dataclass creation

#### **Edge Cases and Error Handling**

- ✅ Grayscale vs RGB image handling
- ✅ Zero-valued and max-valued images
- ✅ NaN and infinity value handling
- ✅ Corrupted file handling
- ✅ Response structure validation

## Running the Tests

### Run All Tests

```bash
cd backend
python -m unittest tests.test_synthid
```

### Run Specific Test Class

```bash
python -m unittest tests.test_synthid.TestSynthIDServiceInitialization
```

### Run Specific Test Method

```bash
python -m unittest tests.test_synthid.TestSynthIDServiceInitialization.test_initialization_with_valid_codebook
```

### Run with Verbose Output

```bash
python -m unittest tests.test_synthid -v
```

### Run with Coverage Report (requires coverage package)

```bash
pip install coverage
coverage run -m unittest tests.test_synthid
coverage report
coverage html
```

## Test Structure

Each test class focuses on a specific component or functionality:

- **TestSynthIDServiceInitialization**: Service setup and initialization
- **TestSynthIDServiceAnalyze**: Image analysis functionality
- **TestSynthIDCheckerInitialization**: Checker setup
- **TestSynthIDCheckerImageChecking**: Image checking methods
- **TestRobustSynthIDExtractorInitialization**: Extractor configuration
- **TestDetectionResult**: Result dataclass behavior
- **TestEdgeCases**: Boundary conditions and error scenarios
- **TestResponseStructure**: Output validation

## Requirements

The tests require:

- `numpy`
- `opencv-python` (cv2)
- `scipy`
- `scikit-learn`
- `PyWavelets`
- Standard library modules: `unittest`, `tempfile`, `pickle`, `os`

All these are already in `requirements.txt`.

## Mocking Strategy

Tests use:

- **tempfile**: Create temporary directories for test files
- **unittest.mock**: Mock external dependencies
- **MockCodebook**: Custom fixture for consistent test codebooks

## Notes

- Tests create temporary files and directories that are cleaned up automatically
- Mock codebooks are minimal but contain all required keys
- Tests validate both successful operations and error conditions
- Response structure tests ensure consistent API contracts
- Float handling tests cover NaN, infinity, and extreme values

## Extending the Tests

To add tests for new functionality:

1. Create a new test class inheriting from `unittest.TestCase`
2. Use `setUp()` for test fixture initialization
3. Use `tearDown()` for cleanup
4. Follow naming convention: `test_<what_is_being_tested>`
5. Use descriptive docstrings for each test

Example:

```python
class TestNewFeature(unittest.TestCase):
    def setUp(self):
        # Initialize fixtures
        pass

    def tearDown(self):
        # Clean up resources
        pass

    def test_specific_behavior(self):
        """Test that describes what is being tested."""
        # Arrange
        # Act
        # Assert
```

## Continuous Integration

These tests are designed to be CI/CD friendly:

- No external files required (uses temp directories)
- No network dependencies
- Deterministic pass/fail results
- Fast execution (< 5 seconds typical)
- Clear error messages for debugging

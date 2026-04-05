"""
SynthID Detection Test and Report Generator

Tests the SynthID detection on a batch of images from the Gemini folder
and generates a comprehensive report with statistics and analysis.
"""

import os
import sys
import json
import csv
import pickle
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import statistics
import numpy as np
from scipy.fft import fft2, fftshift

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.synthid.service import SynthIDService
from src.synthid_checker import SynthIDChecker


class SynthIDTestRunner:
    """Run comprehensive tests on a set of images."""
    
    def __init__(self, codebook_path: str, image_folder: str):
        """
        Initialize the test runner.
        
        Args:
            codebook_path: Path to the SynthID codebook
            image_folder: Folder containing images to test
        """
        self.codebook_path = codebook_path
        self.image_folder = image_folder
        self.service = SynthIDService(codebook_path)
        self.results = []
        self.stats = defaultdict(list)
        
    def discover_images(self):
        """Discover all image files in the folder."""
        image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        images = []
        
        image_path = Path(self.image_folder)
        if not image_path.exists():
            print(f"Error: Image folder not found: {self.image_folder}")
            return []
        
        for ext in image_extensions:
            images.extend(image_path.glob(f'*{ext}'))
            images.extend(image_path.glob(f'*{ext.upper()}'))
        
        return sorted(list(set(images)))
    
    def test_images(self):
        """Test all images and collect results."""
        images = self.discover_images()
        
        if not images:
            print("No images found!")
            return False
        
        print(f"\n{'='*80}")
        print(f"SynthID Detection Test Report")
        print(f"{'='*80}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Images: {len(images)}")
        print(f"Image Folder: {self.image_folder}")
        print(f"Service Available: {self.service.available}")
        print(f"{'='*80}\n")
        
        if not self.service.available:
            print(f"Error: SynthID service not available: {self.service.error}")
            return False
        
        detected_count = 0
        error_count = 0
        
        for idx, image_path in enumerate(images, 1):
            print(f"[{idx:3d}/{len(images)}] Testing: {image_path.name}...", end=' ')
            
            try:
                result = self.service.analyze(str(image_path))
                
                # Create result record
                record = {
                    'image_name': image_path.name,
                    'image_path': str(image_path),
                    'timestamp': datetime.now().isoformat(),
                    'status': result.get('status'),
                    'is_watermarked': result.get('is_watermarked', False),
                    'confidence': result.get('confidence', 0),
                    'raw_confidence': result.get('raw_confidence', 0),
                    'correlation': result.get('correlation', 0),
                    'phase_match': result.get('phase_match', 0),
                    'carrier_strength': result.get('carrier_strength', 0),
                    'structure_ratio': result.get('structure_ratio', 0),
                    'multi_scale_consistency': result.get('multi_scale_consistency', 0),
                    'message': result.get('message'),
                    'error': result.get('error')
                }
                
                self.results.append(record)
                
                # Collect stats
                if result.get('status') == 'complete':
                    self.stats['confidence'].append(result.get('confidence', 0))
                    self.stats['correlation'].append(result.get('correlation', 0))
                    self.stats['phase_match'].append(result.get('phase_match', 0))
                    self.stats['carrier_strength'].append(result.get('carrier_strength', 0))
                    self.stats['structure_ratio'].append(result.get('structure_ratio', 0))
                    self.stats['multi_scale_consistency'].append(result.get('multi_scale_consistency', 0))
                    
                    if result.get('is_watermarked'):
                        detected_count += 1
                        print(f"✓ DETECTED (confidence: {result.get('confidence'):.2f}%)")
                    else:
                        print(f"✗ Not detected (confidence: {result.get('confidence'):.2f}%)")
                else:
                    error_count += 1
                    print(f"⚠ ERROR: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                error_count += 1
                print(f"✗ EXCEPTION: {str(e)}")
                self.results.append({
                    'image_name': image_path.name,
                    'image_path': str(image_path),
                    'status': 'exception',
                    'error': str(e)
                })
        
        print(f"\n{'='*80}")
        print(f"Test Summary")
        print(f"{'='*80}")
        print(f"Total Processed: {len(images)}")
        print(f"Successfully Analyzed: {len(images) - error_count}")
        print(f"Errors: {error_count}")
        print(f"Watermark Detected: {detected_count}")
        print(f"Detection Rate: {(detected_count / (len(images) - error_count) * 100):.2f}%")
        
        return True
    
    def calculate_statistics(self):
        """Calculate detailed statistics."""
        stats_summary = {}
        
        for metric, values in self.stats.items():
            if values:
                stats_summary[metric] = {
                    'count': len(values),
                    'min': float(min(values)),
                    'max': float(max(values)),
                    'mean': float(statistics.mean(values)),
                    'median': float(statistics.median(values)),
                    'stdev': float(statistics.stdev(values)) if len(values) > 1 else 0,
                    'values': values
                }
        
        return stats_summary
    
    def generate_text_report(self, output_path: str):
        """Generate a detailed text report."""
        stats_summary = self.calculate_statistics()
        
        with open(output_path, 'w') as f:
            f.write("SynthID WATERMARK DETECTION TEST REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            detected_count = sum(1 for r in self.results if r.get('is_watermarked'))
            error_count = sum(1 for r in self.results if r.get('status') in ['error', 'exception'])
            successful = len(self.results) - error_count
            
            f.write("SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Images Tested: {len(self.results)}\n")
            f.write(f"Successfully Analyzed: {successful}\n")
            f.write(f"Errors/Exceptions: {error_count}\n")
            f.write(f"SynthID Watermarks Detected: {detected_count}\n")
            if successful > 0:
                f.write(f"Detection Rate: {(detected_count / successful * 100):.2f}%\n")
            f.write(f"\n")
            
            # Statistics
            if stats_summary:
                f.write("DETAILED STATISTICS\n")
                f.write("-" * 80 + "\n")
                
                for metric, stats in stats_summary.items():
                    f.write(f"\n{metric.upper().replace('_', ' ')}:\n")
                    f.write(f"  Count:   {stats['count']}\n")
                    f.write(f"  Min:     {stats['min']:.6f}\n")
                    f.write(f"  Max:     {stats['max']:.6f}\n")
                    f.write(f"  Mean:    {stats['mean']:.6f}\n")
                    f.write(f"  Median:  {stats['median']:.6f}\n")
                    f.write(f"  StdDev:  {stats['stdev']:.6f}\n")
            
            # Detailed Results
            f.write("\n\n")
            f.write("DETAILED IMAGE RESULTS\n")
            f.write("-" * 80 + "\n")
            
            for idx, result in enumerate(self.results, 1):
                f.write(f"\n[{idx}] {result.get('image_name', 'Unknown')}\n")
                f.write(f"    Status: {result.get('status', 'N/A')}\n")
                
                if result.get('status') == 'complete':
                    f.write(f"    Watermarked: {result.get('is_watermarked', False)}\n")
                    f.write(f"    Confidence: {result.get('confidence', 0):.2f}%\n")
                    f.write(f"    Raw Confidence: {result.get('raw_confidence', 0):.6f}\n")
                    f.write(f"    Correlation: {result.get('correlation', 0):.6f}\n")
                    f.write(f"    Phase Match: {result.get('phase_match', 0):.6f}\n")
                    f.write(f"    Carrier Strength: {result.get('carrier_strength', 0):.6f}\n")
                    f.write(f"    Structure Ratio: {result.get('structure_ratio', 0):.6f}\n")
                    f.write(f"    Multi-Scale Consistency: {result.get('multi_scale_consistency', 0):.6f}\n")
                else:
                    if result.get('error'):
                        f.write(f"    Error: {result.get('error')}\n")
                    f.write(f"    Message: {result.get('message', 'N/A')}\n")
        
        print(f"\nText Report saved to: {output_path}")
    
    def generate_csv_report(self, output_path: str):
        """Generate CSV report for detailed analysis."""
        if not self.results:
            print("No results to export")
            return
        
        # Get all keys from results
        fieldnames = set()
        for result in self.results:
            fieldnames.update(result.keys())
        
        fieldnames = sorted(list(fieldnames))
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"CSV Report saved to: {output_path}")
    
    def generate_json_report(self, output_path: str):
        """Generate JSON report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_config': {
                'codebook_path': self.codebook_path,
                'image_folder': self.image_folder,
                'service_available': self.service.available
            },
            'summary': {
                'total_images': len(self.results),
                'successful': sum(1 for r in self.results if r.get('status') == 'complete'),
                'errors': sum(1 for r in self.results if r.get('status') in ['error', 'exception']),
                'detected': sum(1 for r in self.results if r.get('is_watermarked'))
            },
            'statistics': self.calculate_statistics(),
            'results': self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"JSON Report saved to: {output_path}")


def create_default_codebook(codebook_path: str):
    """Create a default codebook if one doesn't exist."""
    print(f"Creating default codebook at {codebook_path}...")
    
    # Create synthetic reference data
    size = 512
    codebook = {
        'image_size': size,
        'reference_noise': np.random.randn(size, size, 3) * 0.1,
        'reference_phase': np.angle(fftshift(fft2(np.random.randn(size, size)))),
        'carriers': [
            {'frequency': (48, 0), 'phase': 0.5, 'strength': 0.8},
            {'frequency': (-48, 0), 'phase': 0.5, 'strength': 0.8},
            {'frequency': (96, 0), 'phase': 0.5, 'strength': 0.7},
            {'frequency': (-96, 0), 'phase': 0.5, 'strength': 0.7},
            {'frequency': (192, 0), 'phase': 0.5, 'strength': 0.6},
            {'frequency': (-192, 0), 'phase': 0.5, 'strength': 0.6},
            {'frequency': (0, 88), 'phase': 0.5, 'strength': 0.75},
            {'frequency': (0, -88), 'phase': 0.5, 'strength': 0.75},
            {'frequency': (0, 176), 'phase': 0.5, 'strength': 0.65},
            {'frequency': (0, -176), 'phase': 0.5, 'strength': 0.65},
        ],
        'known_carriers': [
            (48, 0), (-48, 0),
            (96, 0), (-96, 0),
            (192, 0), (-192, 0),
            (0, 88), (0, -88),
            (0, 176), (0, -176),
        ],
        'scales_used': [256, 512, 1024],
        'detection_threshold': 0.21,
        'correlation_mean': 0.25,
        'correlation_std': 0.15,
    }
    
    os.makedirs(os.path.dirname(codebook_path), exist_ok=True)
    with open(codebook_path, 'wb') as f:
        pickle.dump(codebook, f)
    
    print(f"✓ Default codebook created successfully")
    return codebook_path


def main():
    """Main entry point."""
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(base_dir)
    project_root = os.path.dirname(backend_dir)
    
    # Look for codebook
    codebook_candidates = [
        os.path.join(project_root, 'model_output', 'synthid', 'robust_codebook.pkl'),
        os.path.join(backend_dir, 'robust_codebook.pkl'),
        os.path.join(backend_dir, 'src/synthid/robust_codebook.pkl'),
    ]
    
    codebook_path = None
    for candidate in codebook_candidates:
        if os.path.exists(candidate):
            codebook_path = candidate
            break
    
    # If not found, create a default one
    if not codebook_path:
        codebook_path = codebook_candidates[0]  # Use the default location
        create_default_codebook(codebook_path)
    
    # Image folder
    image_folder = os.path.join(backend_dir, 'Gemini')
    
    # Create test runner
    runner = SynthIDTestRunner(codebook_path, image_folder)
    
    # Run tests
    if not runner.test_images():
        return
    
    # Generate reports
    reports_dir = os.path.join(backend_dir, 'test_reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    runner.generate_text_report(os.path.join(reports_dir, f'synthid_report_{timestamp}.txt'))
    runner.generate_csv_report(os.path.join(reports_dir, f'synthid_report_{timestamp}.csv'))
    runner.generate_json_report(os.path.join(reports_dir, f'synthid_report_{timestamp}.json'))
    
    print(f"\nAll reports generated in: {reports_dir}")


if __name__ == '__main__':
    main()

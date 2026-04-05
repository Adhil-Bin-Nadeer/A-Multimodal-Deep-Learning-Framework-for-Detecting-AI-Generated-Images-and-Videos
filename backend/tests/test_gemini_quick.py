"""
Quick SynthID Batch Test - Fast Version

Tests the SynthID detection on Gemini images and generates a report.
Optimized for speed with reduced sample size.
"""

import os
import sys
import json
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


class QuickSynthIDTest:
    """Quick test runner for SynthID detection."""
    
    def __init__(self, codebook_path: str, image_folder: str, limit: int = None):
        """
        Initialize the test runner.
        
        Args:
            codebook_path: Path to the SynthID codebook
            image_folder: Folder containing images to test
            limit: Max number of images to test (None for all)
        """
        self.codebook_path = codebook_path
        self.image_folder = image_folder
        self.limit = limit
        self.service = SynthIDService(codebook_path)
        self.results = []
        self.stats = defaultdict(list)
        
    def discover_images(self):
        """Discover image files in the folder."""
        image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        images = []
        
        image_path = Path(self.image_folder)
        if not image_path.exists():
            print(f"Error: Image folder not found: {self.image_folder}")
            return []
        
        for ext in image_extensions:
            images.extend(image_path.glob(f'*{ext}'))
            images.extend(image_path.glob(f'*{ext.upper()}'))
        
        images = sorted(list(set(images)))
        if self.limit:
            images = images[:self.limit]
        
        return images
    
    def run(self):
        """Run tests and generate report."""
        images = self.discover_images()
        
        if not images:
            print("No images found!")
            return False
        
        print(f"\n{'='*80}")
        print(f"SynthID Batch Test - {len(images)} Images")
        print(f"{'='*80}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Image Folder: {self.image_folder}")
        print(f"Service Available: {self.service.available}")
        
        if not self.service.available:
            print(f"ERROR: Service not available: {self.service.error}")
            return False
        
        detected_count = 0
        error_count = 0
        
        for idx, image_path in enumerate(images, 1):
            rel_path = image_path.name
            print(f"[{idx:3d}/{len(images)}] {rel_path:<50}", end=' ', flush=True)
            
            try:
                result = self.service.analyze(str(image_path))
                
                record = {
                    'name': image_path.name,
                    'path': str(image_path),
                    'status': result.get('status'),
                    'watermarked': result.get('is_watermarked', False),
                    'confidence': result.get('confidence', 0),
                    'correlation': result.get('correlation', 0),
                    'phase_match': result.get('phase_match', 0),
                    'structure_ratio': result.get('structure_ratio', 0),
                    'error': result.get('error')
                }
                
                self.results.append(record)
                
                if result.get('status') == 'complete':
                    self.stats['confidence'].append(result.get('confidence', 0))
                    self.stats['correlation'].append(result.get('correlation', 0))
                    self.stats['phase_match'].append(result.get('phase_match', 0))
                    self.stats['structure_ratio'].append(result.get('structure_ratio', 0))
                    
                    if result.get('is_watermarked'):
                        detected_count += 1
                        print(f"✓ DETECTED ({result.get('confidence'):.1f}%)")
                    else:
                        print(f"✗ Not detected ({result.get('confidence'):.1f}%)")
                else:
                    error_count += 1
                    print(f"⚠ ERROR: {result.get('error', 'Unknown')[:40]}")
                    
            except Exception as e:
                error_count += 1
                print(f"✗ EXCEPTION: {str(e)[:40]}")
                self.results.append({
                    'name': image_path.name,
                    'status': 'exception',
                    'error': str(e)
                })
        
        self.print_summary(len(images), detected_count, error_count)
        return True
    
    def print_summary(self, total, detected, errors):
        """Print summary statistics."""
        successful = total - errors
        
        print(f"\n{'='*80}")
        print(f"SUMMARY STATISTICS")
        print(f"{'='*80}")
        print(f"Total Images:          {total}")
        print(f"Successfully Analyzed: {successful}")
        print(f"Errors:                {errors}")
        print(f"Watermarks Detected:   {detected}")
        if successful > 0:
            print(f"Detection Rate:        {(detected / successful * 100):.1f}%")
        
        # Metric statistics
        print(f"\n{'METRIC STATISTICS':-^80}")
        for metric, values in self.stats.items():
            if values:
                print(f"\n{metric.upper().replace('_', ' ')}:")
                print(f"  Min:    {min(values):>8.4f}  |  Max:    {max(values):>8.4f}")
                print(f"  Mean:   {statistics.mean(values):>8.4f}  |  Median: {statistics.median(values):>8.4f}")
                if len(values) > 1:
                    print(f"  StdDev: {statistics.stdev(values):>8.4f}")
        
        self.save_results()
    
    def save_results(self):
        """Save JSON report."""
        report_dir = Path(self.image_folder).parent / 'test_reports'
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = report_dir / f'gemini_synthid_report_{timestamp}.json'
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_config': {
                'image_folder': self.image_folder,
                'total_images': len(self.results),
                'service_available': self.service.available
            },
            'summary': {
                'total': len(self.results),
                'successful': sum(1 for r in self.results if r.get('status') == 'complete'),
                'detected': sum(1 for r in self.results if r.get('watermarked', False)),
                'errors': sum(1 for r in self.results if r.get('status') in ['error', 'exception'])
            },
            'statistics': {
                metric: {
                    'count': len(vals),
                    'min': float(min(vals)),
                    'max': float(max(vals)),
                    'mean': float(statistics.mean(vals)),
                    'median': float(statistics.median(vals)),
                    'stdev': float(statistics.stdev(vals)) if len(vals) > 1 else 0
                }
                for metric, vals in self.stats.items() if vals
            },
            'results': self.results
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"Report saved to: {report_path}")
        print(f"{'='*80}\n")


def create_default_codebook(codebook_path: str):
    """Create a default codebook if one doesn't exist."""
    print(f"Creating default codebook...")
    
    size = 512
    codebook = {
        'image_size': size,
        'reference_noise': np.random.randn(size, size, 3) * 0.1,
        'reference_phase': np.angle(fftshift(fft2(np.random.randn(size, size)))),
        'carriers': [
            {'frequency': (f, 0), 'phase': 0.5} for f in [48, -48, 96, -96, 192, -192]
        ] + [
            {'frequency': (0, f), 'phase': 0.5} for f in [88, -88, 176, -176]
        ],
        'known_carriers': [
            (48, 0), (-48, 0), (96, 0), (-96, 0), (192, 0), (-192, 0),
            (0, 88), (0, -88), (0, 176), (0, -176)
        ],
        'scales_used': [256, 512, 1024],
        'detection_threshold': 0.21,
        'correlation_mean': 0.25,
    }
    
    os.makedirs(os.path.dirname(codebook_path), exist_ok=True)
    with open(codebook_path, 'wb') as f:
        pickle.dump(codebook, f)
    
    print(f"✓ Codebook created")
    return codebook_path


def main():
    """Main entry point."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(base_dir)
    project_root = os.path.dirname(backend_dir)
    
    codebook_path = os.path.join(project_root, 'model_output', 'synthid', 'robust_codebook.pkl')
    
    if not os.path.exists(codebook_path):
        create_default_codebook(codebook_path)
    
    image_folder = os.path.join(backend_dir, 'Gemini')
    
    # Test with all images (or limit if needed for testing)
    test = QuickSynthIDTest(codebook_path, image_folder, limit=None)
    test.run()


if __name__ == '__main__':
    main()

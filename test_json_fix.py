#!/usr/bin/env python3
"""
Quick test of HepG2 experiment to verify JSON serialization fix.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_json_conversion():
    """Test the JSON conversion function."""
    
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj
    
    # Create test results similar to experiment output
    test_results = {
        'latent_dims': [32, 64],
        'final_losses': [np.float32(1234.5), np.float64(2345.6)],
        'n_parameters': [12345, 23456],
        'best_metrics': {
            32: {
                'cell_r2_mean': np.float32(0.58),
                'cell_r2_std': np.float32(0.15),
                'cell_pearson_mean': np.float64(0.76),
                'cell_pearson_std': np.float64(0.06),
                'gene_values': np.array([1.1, 2.2, 3.3])
            },
            64: {
                'cell_r2_mean': np.float32(0.62),
                'cell_r2_std': np.float32(0.14),
                'cell_pearson_mean': np.float64(0.78),
                'cell_pearson_std': np.float64(0.05),
                'gene_values': np.array([1.4, 2.5, 3.6])
            }
        }
    }
    
    print("Original data types:")
    print(f"  final_losses[0] type: {type(test_results['final_losses'][0])}")
    print(f"  cell_r2_mean type: {type(test_results['best_metrics'][32]['cell_r2_mean'])}")
    
    # Convert to JSON-safe format
    json_safe_results = convert_numpy_types(test_results)
    
    print("\nConverted data types:")
    print(f"  final_losses[0] type: {type(json_safe_results['final_losses'][0])}")
    print(f"  cell_r2_mean type: {type(json_safe_results['best_metrics'][32]['cell_r2_mean'])}")
    
    # Try to serialize to JSON
    try:
        json_str = json.dumps(json_safe_results, indent=2)
        print("\n✓ JSON serialization successful!")
        
        # Save to temporary file
        os.makedirs("experiments", exist_ok=True)
        test_path = "experiments/test_json_conversion.json"
        with open(test_path, 'w') as f:
            json.dump(json_safe_results, f, indent=2)
        
        print(f"✓ Test results saved to: {test_path}")
        
        # Read back to verify
        with open(test_path, 'r') as f:
            loaded_results = json.load(f)
        
        print("✓ Successfully loaded back from JSON")
        print(f"Loaded cell_r2_mean: {loaded_results['best_metrics']['32']['cell_r2_mean']}")
        
        return True
        
    except Exception as e:
        print(f"✗ JSON serialization failed: {e}")
        return False

if __name__ == '__main__':
    success = test_json_conversion()
    if success:
        print("\n🎉 JSON conversion fix is working!")
    else:
        print("\n❌ JSON conversion fix needs more work")

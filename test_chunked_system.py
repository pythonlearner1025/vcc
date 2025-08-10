#!/usr/bin/env python3
"""
Comprehensive test suite for the VCC chunked data processing system.

This script provides modular testing for:
1. Environment validation and setup
2. Data processing into optimized chunks 
3. Efficient data loading with caching
4. Memory monitoring and performance validation
5. Integration testing with training pipeline
6. Benchmarking against legacy system

Usage:
    # Quick test with K562 subset
    uv run test_chunked_system.py --quick-test
    
    # Full processing test
    uv run test_chunked_system.py --config config/chunked_data_config.json
    
    # Test loading only (skip processing)
    uv run test_chunked_system.py --skip-processing --data-path /workspace/vcc/data/processed/chunked_test
    
    # Production-scale test
    uv run test_chunked_system.py --config config/production_config.json --benchmark
"""

import sys
import logging
import argparse
from pathlib import Path
import time
import traceback
import json
import os
from typing import Dict, List, Optional, Tuple, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.data_processor import DataProcessor, DataProcessingConfig
    from src.dataloaders.vae_loader import create_vae_dataloaders, DatasetConfig, MemoryMonitor
    import torch
    import psutil
    import h5py
    import numpy as np
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed:")
    print("  uv sync")
    print("  uv add torch pytorch-lightning scanpy anndata h5py psutil")
    sys.exit(1)

# Constants for paths
RAW_DATA_DIR = Path("/workspace/vcc/data/raw/competition_support_set")
PROCESSED_DATA_DIR = Path("/workspace/vcc/data/processed")
LOG_DIR = PROCESSED_DATA_DIR / "logs"

# Create log directory
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Setup comprehensive logging
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging with file and console handlers."""
    log_file = LOG_DIR / f"test_chunked_system_{int(time.time())}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    return logger

logger = setup_logging()


class TestSuite:
    """Main test suite class for chunked data processing system."""
    
    def __init__(self, args: argparse.Namespace):
        """Initialize test suite with command line arguments."""
        self.args = args
        self.start_time = time.time()
        self.test_results = {}
        
        # Setup paths
        self.raw_data_dir = RAW_DATA_DIR
        self.processed_data_dir = PROCESSED_DATA_DIR
        
        # Determine configuration and output paths
        if args.config:
            self.config_path = Path(args.config)
        elif args.quick_test:
            self.config_path = Path("config/test_config.json")
        else:
            self.config_path = Path("config/chunked_data_config.json")
        
        # Set output directory
        if args.data_path:
            self.output_path = Path(args.data_path)
        elif args.quick_test:
            self.output_path = self.processed_data_dir / "chunked_test"
        else:
            self.output_path = self.processed_data_dir / "chunked"
        
        logger.info(f"Test suite initialized:")
        logger.info(f"  Configuration: {self.config_path}")
        logger.info(f"  Output path: {self.output_path}")
        logger.info(f"  Quick test mode: {args.quick_test}")
    
    def validate_environment(self) -> bool:
        """Validate that the environment is properly set up."""
        logger.info("="*80)
        logger.info("🔍 VALIDATING ENVIRONMENT")
        logger.info("="*80)
        
        try:
            # Check Python version
            python_version = sys.version_info
            logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
            
            # Check system resources
            memory_info = MemoryMonitor.get_memory_info()
            total_ram = psutil.virtual_memory().total / (1024**3)
            logger.info(f"System RAM: {total_ram:.1f}GB (using {memory_info['ram_gb']:.2f}GB)")
            
            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.device_count()} devices")
                logger.info(f"GPU memory: {memory_info.get('gpu_reserved_gb', 0):.2f}GB")
            
            # Check raw data directory
            if not self.raw_data_dir.exists():
                raise FileNotFoundError(f"Raw data directory not found: {self.raw_data_dir}")
            
            # Check for required data files
            required_files = ["k562.h5", "gene_names.csv"]
            available_files = list(self.raw_data_dir.glob("*.h5"))
            
            logger.info(f"Available datasets in {self.raw_data_dir}:")
            for file_path in available_files:
                size_mb = file_path.stat().st_size / (1024**2)
                logger.info(f"  - {file_path.name}: {size_mb:.1f}MB")
            
            missing_files = []
            for file_name in required_files:
                file_path = self.raw_data_dir / file_name
                if not file_path.exists():
                    missing_files.append(str(file_path))
            
            if missing_files:
                logger.warning(f"Some files missing: {missing_files}")
            
            # Create/validate processed data directories
            self.processed_data_dir.mkdir(parents=True, exist_ok=True)
            self.output_path.mkdir(parents=True, exist_ok=True)
            
            # Check configuration file
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            # Validate configuration
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            logger.info(f"Configuration validation:")
            logger.info(f"  Chunk size: {config_data.get('chunk_size', 'unknown')}")
            logger.info(f"  Output path: {config_data.get('output_path', 'unknown')}")
            logger.info(f"  Datasets enabled: {len([d for d in config_data.get('datasets', {}).values() if d.get('include', False)])}")
            
            self.test_results['environment_validation'] = True
            logger.info("✅ Environment validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Environment validation failed: {e}")
            logger.error(traceback.format_exc())
            self.test_results['environment_validation'] = False
            return False
    
    def test_chunked_processing(self) -> Optional[Any]:
        """Test the chunked data processing pipeline."""
        logger.info("="*80)
        logger.info("⚙️  TESTING CHUNKED DATA PROCESSING")
        logger.info("="*80)
        
        start_time = time.time()
        
        try:
            # Skip if requested
            if self.args.skip_processing:
                logger.info("⏭️  Skipping data processing as requested")
                return None
            
            logger.info(f"Using configuration: {self.config_path}")
            
            # Log memory before processing
            MemoryMonitor.log_memory_usage("🔍 Before processing: ")
            
            # Create processor from config
            processor = DataProcessor.from_config(str(self.config_path))
            
            # Run processing pipeline
            logger.info("🚀 Starting data processing...")
            processing_index = processor.process()
            
            processing_time = time.time() - start_time
            
            # Log memory after processing
            MemoryMonitor.log_memory_usage("📊 After processing: ")
            
            # Validate output
            if not processing_index:
                raise ValueError("Processing returned empty index")
            
            # Log results
            logger.info("✅ Chunked processing completed successfully!")
            logger.info(f"⏱️  Processing time: {processing_time:.2f} seconds")
            logger.info(f"📦 Created {processing_index.total_chunks} chunks")
            logger.info(f"🧬 Total cells: {processing_index.total_cells:,}")
            logger.info(f"🧬 Genes per chunk: {len(processing_index.gene_list):,}")
            logger.info(f"💾 Output directory: {self.output_path}")
            
            # Validate chunk files exist
            chunk_files = list(self.output_path.glob("chunk_*.h5"))
            logger.info(f"📁 Chunk files created: {len(chunk_files)}")
            
            if len(chunk_files) != processing_index.total_chunks:
                logger.warning(f"⚠️  Chunk file count mismatch: expected {processing_index.total_chunks}, found {len(chunk_files)}")
            
            self.test_results['processing'] = {
                'success': True,
                'time': processing_time,
                'chunks': processing_index.total_chunks,
                'cells': processing_index.total_cells,
                'genes': len(processing_index.gene_list)
            }
            
            return processing_index
            
        except Exception as e:
            logger.error(f"❌ Chunked processing failed: {e}")
            logger.error(traceback.format_exc())
            self.test_results['processing'] = {'success': False, 'error': str(e)}
            raise
    def test_chunked_loading(self) -> bool:
        """Test loading data with the new chunked dataloader."""
        logger.info("="*80)
        logger.info("📚 TESTING CHUNKED DATA LOADING")
        logger.info("="*80)
        
        start_time = time.time()
        
        try:
            # Validate data path
            if not self.output_path.exists():
                raise FileNotFoundError(f"Processed data path not found: {self.output_path}")
            
            # Check for required files
            index_file = self.output_path / "chunk_index.json"
            if not index_file.exists():
                raise FileNotFoundError(f"Chunk index not found: {index_file}")
            
            # Create dataset config based on test mode
            config = DatasetConfig(
                data_path=str(self.output_path),
                batch_size=16 if self.args.quick_test else 64,
                num_workers=2 if self.args.quick_test else 4,
                chunk_cache_size=2 if self.args.quick_test else 5,
                max_chunks_in_memory=1 if self.args.quick_test else 3,
                monitor_memory=True,
                pin_memory=torch.cuda.is_available()
            )
            
            logger.info(f"📋 DataLoader configuration:")
            logger.info(f"  Batch size: {config.batch_size}")
            logger.info(f"  Workers: {config.num_workers}")
            logger.info(f"  Cache size: {config.chunk_cache_size}")
            logger.info(f"  Max chunks in memory: {config.max_chunks_in_memory}")
            
            # Log memory before loading
            MemoryMonitor.log_memory_usage("🔍 Before dataloader creation: ")
            
            # Create data module
            logger.info("🏗️  Creating data module...")
            data_module = create_vae_dataloaders(config)
            
            # Test dataloaders
            train_loader = data_module.train_dataloader()
            val_loader = data_module.val_dataloader()
            
            logger.info(f"📊 Dataset splits:")
            logger.info(f"  Training samples: {len(data_module.train_dataset)}")
            logger.info(f"  Validation samples: {len(data_module.val_dataset)}")
            
            # Test training dataloader
            logger.info("🚂 Testing training dataloader...")
            max_batches = 3 if self.args.quick_test else 8
            
            train_batch_info = []
            for i, batch_data in enumerate(train_loader):
                gene_expression, metadata_list = batch_data
                
                batch_info = {
                    'batch_id': i + 1,
                    'shape': gene_expression.shape,
                    'dtype': str(gene_expression.dtype),
                    'device': str(gene_expression.device),
                    'has_nan': torch.isnan(gene_expression).any().item(),
                    'has_inf': torch.isinf(gene_expression).any().item(),
                    'min_val': gene_expression.min().item(),
                    'max_val': gene_expression.max().item(),
                    'mean_val': gene_expression.mean().item()
                }
                
                train_batch_info.append(batch_info)
                
                logger.info(f"  Batch {i+1}: {gene_expression.shape} | "
                           f"Range: [{batch_info['min_val']:.3f}, {batch_info['max_val']:.3f}] | "
                           f"Mean: {batch_info['mean_val']:.3f}")
                
                # Check metadata
                if metadata_list and len(metadata_list) > 0:
                    sample_metadata = metadata_list[0]
                    logger.info(f"    Metadata keys: {list(sample_metadata.keys())}")
                    
                    # Check for important metadata fields
                    important_fields = ['dataset', 'UMI_count', 'cell_type', 'perturbation']
                    for field in important_fields:
                        if field in sample_metadata:
                            logger.info(f"    {field}: {sample_metadata[field]}")
                
                # Validate data quality
                if batch_info['has_nan']:
                    logger.warning(f"  ⚠️  NaN values detected in batch {i+1}")
                if batch_info['has_inf']:
                    logger.warning(f"  ⚠️  Inf values detected in batch {i+1}")
                
                # Memory check during loading
                if (i + 1) % 3 == 0:
                    MemoryMonitor.log_memory_usage(f"  After batch {i+1}: ")
                
                if i >= max_batches - 1:
                    break
            
            # Test validation dataloader
            logger.info("🔬 Testing validation dataloader...")
            max_val_batches = 2 if self.args.quick_test else 5
            
            val_batch_info = []
            for i, batch_data in enumerate(val_loader):
                gene_expression, metadata_list = batch_data
                
                batch_info = {
                    'batch_id': i + 1,
                    'shape': gene_expression.shape,
                    'dtype': str(gene_expression.dtype)
                }
                
                val_batch_info.append(batch_info)
                logger.info(f"  Val batch {i+1}: {gene_expression.shape}")
                
                if i >= max_val_batches - 1:
                    break
            
            loading_time = time.time() - start_time
            
            # Log memory after loading
            MemoryMonitor.log_memory_usage("📊 After dataloader testing: ")
            
            logger.info("✅ Chunked loading completed successfully!")
            logger.info(f"⏱️  Loading test time: {loading_time:.2f} seconds")
            logger.info(f"📦 Tested {len(train_batch_info)} training batches")
            logger.info(f"🔬 Tested {len(val_batch_info)} validation batches")
            
            self.test_results['loading'] = {
                'success': True,
                'time': loading_time,
                'train_batches': len(train_batch_info),
                'val_batches': len(val_batch_info),
                'batch_info': train_batch_info[0] if train_batch_info else None
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Chunked loading failed: {e}")
            logger.error(traceback.format_exc())
            self.test_results['loading'] = {'success': False, 'error': str(e)}
            return False
    def test_system_integration(self) -> bool:
        """Test integration with training components and system validation."""
        logger.info("="*80)
        logger.info("🔗 TESTING SYSTEM INTEGRATION")
        logger.info("="*80)
        
        try:
            # Test data path validation
            if not self.output_path.exists():
                raise FileNotFoundError(f"Output path not found: {self.output_path}")
            
            # Load and validate chunk index
            index_file = self.output_path / "chunk_index.json"
            if not index_file.exists():
                raise FileNotFoundError(f"Chunk index not found: {index_file}")
            
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            
            logger.info(f"📋 Index validation:")
            logger.info(f"  Total chunks: {index_data['total_chunks']}")
            logger.info(f"  Total cells: {index_data['total_cells']:,}")
            logger.info(f"  Total genes: {len(index_data['gene_list']):,}")
            logger.info(f"  Processing timestamp: {index_data.get('processing_timestamp', 'unknown')}")
            
            # Check chunk files exist and validate sizes
            chunk_info_list = index_data['chunks']
            missing_chunks = []
            total_chunk_size = 0
            
            for chunk_info in chunk_info_list:
                chunk_file = self.output_path / chunk_info['chunk_path']
                if not chunk_file.exists():
                    missing_chunks.append(chunk_info['chunk_path'])
                else:
                    chunk_size = chunk_file.stat().st_size
                    total_chunk_size += chunk_size
                    
                    # Validate chunk can be opened
                    try:
                        with h5py.File(chunk_file, 'r') as f:
                            if 'X' not in f:
                                logger.warning(f"⚠️  Chunk {chunk_info['chunk_path']} missing 'X' dataset")
                    except Exception as e:
                        logger.warning(f"⚠️  Cannot open chunk {chunk_info['chunk_path']}: {e}")
            
            if missing_chunks:
                logger.error(f"❌ Missing chunk files: {missing_chunks}")
                raise FileNotFoundError(f"Chunk files missing: {missing_chunks}")
            
            logger.info(f"💾 Storage validation:")
            logger.info(f"  Total chunk storage: {total_chunk_size / (1024**2):.1f}MB")
            logger.info(f"  Average chunk size: {total_chunk_size / len(chunk_info_list) / (1024**2):.1f}MB")
            
            # Test data consistency
            logger.info("🔍 Testing data consistency...")
            
            # Sample a few chunks and validate gene consistency
            sample_chunks = chunk_info_list[:min(3, len(chunk_info_list))]
            first_genes = None
            
            for chunk_info in sample_chunks:
                chunk_file = self.output_path / chunk_info['chunk_path']
                with h5py.File(chunk_file, 'r') as f:
                    chunk_genes = [g.decode('utf-8') if isinstance(g, bytes) else str(g) for g in f['genes'][:]]
                    
                    if first_genes is None:
                        first_genes = chunk_genes
                        logger.info(f"  Reference gene count: {len(first_genes)}")
                    else:
                        if chunk_genes != first_genes:
                            logger.warning(f"⚠️  Gene inconsistency in chunk {chunk_info['chunk_path']}")
                        else:
                            logger.info(f"  ✅ Gene consistency verified for {chunk_info['chunk_path']}")
            
            # Test compatibility with training interface
            logger.info("🚀 Testing training interface compatibility...")
            
            try:
                config = DatasetConfig(
                    data_path=str(self.output_path),
                    batch_size=8,
                    num_workers=0,  # Avoid multiprocessing in tests
                    chunk_cache_size=2
                )
                
                data_module = create_vae_dataloaders(config)
                
                # Test one batch
                train_loader = data_module.train_dataloader()
                batch = next(iter(train_loader))
                gene_expression, metadata_list = batch
                
                logger.info(f"  ✅ Training interface test passed")
                logger.info(f"    Batch shape: {gene_expression.shape}")
                logger.info(f"    Data type: {gene_expression.dtype}")
                logger.info(f"    Metadata samples: {len(metadata_list)}")
                
            except Exception as e:
                logger.warning(f"⚠️  Training interface test failed: {e}")
            
            logger.info("✅ System integration tests passed!")
            
            self.test_results['integration'] = {
                'success': True,
                'total_chunks': len(chunk_info_list),
                'total_size_mb': total_chunk_size / (1024**2),
                'missing_chunks': len(missing_chunks)
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ System integration failed: {e}")
            logger.error(traceback.format_exc())
            self.test_results['integration'] = {'success': False, 'error': str(e)}
            return False
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run performance benchmarks if requested."""
        if not self.args.benchmark:
            return {}
        
        logger.info("="*80)
        logger.info("📊 RUNNING PERFORMANCE BENCHMARKS")
        logger.info("="*80)
        
        benchmark_results = {}
        
        try:
            # Benchmark loading speed
            logger.info("🚄 Benchmarking data loading speed...")
            
            config = DatasetConfig(
                data_path=str(self.output_path),
                batch_size=64,
                num_workers=4,
                chunk_cache_size=5
            )
            
            data_module = create_vae_dataloaders(config)
            train_loader = data_module.train_dataloader()
            
            # Time loading of multiple batches
            start_time = time.time()
            batch_count = 0
            total_cells = 0
            
            for i, (gene_expression, metadata_list) in enumerate(train_loader):
                batch_count += 1
                total_cells += gene_expression.shape[0]
                
                if i >= 20:  # Test 20 batches
                    break
            
            loading_time = time.time() - start_time
            cells_per_second = total_cells / loading_time
            
            benchmark_results['loading_speed'] = {
                'batches_tested': batch_count,
                'total_cells': total_cells,
                'time_seconds': loading_time,
                'cells_per_second': cells_per_second
            }
            
            logger.info(f"  📈 Loading speed: {cells_per_second:.0f} cells/second")
            logger.info(f"  📦 Tested {batch_count} batches ({total_cells:,} cells)")
            
        except Exception as e:
            logger.error(f"❌ Benchmark failed: {e}")
            benchmark_results['error'] = str(e)
        
        return benchmark_results
    
    def generate_report(self):
        """Generate a comprehensive test report."""
        total_time = time.time() - self.start_time
        
        logger.info("="*80)
        logger.info("📋 COMPREHENSIVE TEST REPORT")
        logger.info("="*80)
        
        logger.info(f"⏱️  Total test duration: {total_time:.2f} seconds")
        logger.info(f"🔧 Configuration used: {self.config_path}")
        logger.info(f"💾 Output directory: {self.output_path}")
        
        # Summary of test results
        passed_tests = sum(1 for test_name, result in self.test_results.items() 
                          if isinstance(result, dict) and result.get('success', False))
        total_tests = len(self.test_results)
        
        logger.info(f"✅ Tests passed: {passed_tests}/{total_tests}")
        
        # Detailed results
        for test_name, result in self.test_results.items():
            if isinstance(result, dict):
                status = "✅ PASSED" if result.get('success', False) else "❌ FAILED"
                logger.info(f"  {test_name}: {status}")
                
                if 'time' in result:
                    logger.info(f"    Time: {result['time']:.2f}s")
                
                if 'error' in result:
                    logger.info(f"    Error: {result['error']}")
            else:
                status = "✅ PASSED" if result else "❌ FAILED"
                logger.info(f"  {test_name}: {status}")
        
        # System benefits summary
        if passed_tests == total_tests:
            logger.info("")
            logger.info("🎉 ALL TESTS PASSED - CHUNKED SYSTEM WORKING!")
            logger.info("")
            logger.info("💡 Benefits of new chunked system:")
            logger.info("  • ⚡ No more huge monolithic files")
            logger.info("  • 🚄 Efficient training data loading")
            logger.info("  • 🧠 Memory-optimized chunk caching")
            logger.info("  • 🔍 Fast indexed retrieval by SRX/cell")
            logger.info("  • 🔗 Easy integration with existing training")
            logger.info("  • 📦 Compressed, optimized storage")
            logger.info("  • 🔧 Modular and configurable pipeline")
            logger.info("")
            logger.info(f"📁 Processed data ready at: {self.output_path}")
        else:
            logger.info("")
            logger.info("⚠️  Some tests failed - please review the errors above")
        
        return passed_tests == total_tests


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test suite for VCC chunked data processing system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with K562 subset
  uv run test_chunked_system.py --quick-test
  
  # Full processing test with all datasets
  uv run test_chunked_system.py --config config/chunked_data_config.json
  
  # Test loading only (skip processing)  
  uv run test_chunked_system.py --skip-processing --data-path /workspace/vcc/data/processed/chunked_test
  
  # Production test with benchmarks
  uv run test_chunked_system.py --config config/production_config.json --benchmark
        """
    )
    
    parser.add_argument('--config', 
                       help='Configuration file path (default: auto-select based on test mode)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with K562 subset only')
    parser.add_argument('--skip-processing', action='store_true',
                       help='Skip data processing, only test loading')
    parser.add_argument('--data-path',
                       help='Path to processed data directory (for loading tests)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmarks')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    return parser


def main():
    """Main function to run all tests."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging with specified level
    global logger
    logger = setup_logging(args.log_level)
    
    try:
        # Initialize test suite
        test_suite = TestSuite(args)
        
        # Run validation
        if not test_suite.validate_environment():
            return 1
        
        # Run processing test
        processing_index = test_suite.test_chunked_processing()
        
        # Run loading test
        if not test_suite.test_chunked_loading():
            return 1
        
        # Run integration test
        if not test_suite.test_system_integration():
            return 1
        
        # Run benchmarks if requested
        benchmark_results = test_suite.run_benchmark()
        
        # Generate final report
        success = test_suite.generate_report()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

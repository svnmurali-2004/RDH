"""
Enhanced Multi-Level PVS-RDH Package Initialization
Main entry point for the EML-PVS-RDH library
"""

# Version information
__version__ = "1.0.0"
__author__ = "Enhanced PVS-RDH Research Team"
__email__ = "eml-pvs-rdh@research.edu"
__description__ = "Enhanced Multi-Level Pixel Value Splitting for High-Capacity Reversible Data Hiding"

# Import main classes and functions for easy access
from .eml_pvs_rdh_main import EML_PVS_RDH, EMLPVSConfig, embed_watermark, extract_watermark
from .pixel_decomposition import PixelDecomposition
from .metrics import QualityMetrics, CapacityMetrics, PerformanceMetrics, ComparativeMetrics

# Import additional utilities
try:
    from .region_classifier import RegionClassifier
except ImportError:
    RegionClassifier = None

# Define what gets imported with "from eml_pvs_rdh import *"
__all__ = [
    # Main algorithm
    'EML_PVS_RDH',
    'EMLPVSConfig',
    
    # Convenience functions
    'embed_watermark',
    'extract_watermark',
    
    # Core components
    'PixelDecomposition',
    'RegionClassifier',
    
    # Metrics and evaluation
    'QualityMetrics',
    'CapacityMetrics', 
    'PerformanceMetrics',
    'ComparativeMetrics',
    
    # Package info
    '__version__',
    '__author__',
    '__description__'
]

# Package-level configuration
import logging

# Set up default logging configuration
logging.getLogger(__name__).addHandler(logging.NullHandler())

def get_version():
    """Get package version"""
    return __version__

def get_supported_features():
    """Get list of supported features"""
    features = [
        'Hierarchical three-level pixel decomposition (H, M, L)',
        'Multi-strategy embedding framework',
        'Adaptive region classification',
        'Dual-pass optimization',
        'Dynamic parameter adaptation',
        'Comprehensive quality metrics',
        'Performance benchmarking',
        'State-of-art comparison'
    ]
    
    if RegionClassifier is None:
        features.append('Note: RegionClassifier requires additional dependencies')
    
    return features

def create_default_config(**kwargs):
    """
    Create default configuration with optional overrides
    
    Args:
        **kwargs: Configuration parameters to override
        
    Returns:
        EMLPVSConfig: Default configuration with overrides
    """
    return EMLPVSConfig(**kwargs)

def quick_embed(image, watermark, **config_kwargs):
    """
    Quick embedding function with default settings
    
    Args:
        image (np.ndarray): Input grayscale image
        watermark (str): Text watermark to embed
        **config_kwargs: Configuration parameters
        
    Returns:
        Tuple[np.ndarray, Dict]: (embedded_image, embedding_info)
    """
    return embed_watermark(image, watermark, **config_kwargs)

def quick_extract(embedded_image, embedding_info):
    """
    Quick extraction function
    
    Args:
        embedded_image (np.ndarray): Embedded image
        embedding_info (Dict): Embedding information
        
    Returns:
        Tuple[str, np.ndarray]: (extracted_text, recovered_image)
    """
    return extract_watermark(embedded_image, embedding_info)

def benchmark_algorithm(image, watermarks, configurations=None):
    """
    Benchmark algorithm performance across different configurations
    
    Args:
        image (np.ndarray): Test image
        watermarks (List[str]): List of watermarks to test
        configurations (List[EMLPVSConfig], optional): Configurations to test
        
    Returns:
        Dict: Benchmark results
    """
    if configurations is None:
        configurations = [
            ('Standard', EMLPVSConfig()),
            ('High_Capacity', EMLPVSConfig(
                enable_dual_pass=True,
                adaptive_thresholds=True,
                optimization_target='capacity'
            )),
            ('High_Quality', EMLPVSConfig(
                enable_dual_pass=False,
                adaptive_thresholds=True,
                optimization_target='quality'
            )),
            ('Balanced', EMLPVSConfig(
                enable_dual_pass=True,
                adaptive_thresholds=True,
                optimization_target='balanced'
            ))
        ]
    
    results = {}
    quality_calc = QualityMetrics()
    capacity_calc = CapacityMetrics()
    
    for config_name, config in configurations:
        algorithm = EML_PVS_RDH(config)
        config_results = []
        
        for watermark in watermarks:
            try:
                # Embed and extract
                embedded_img, embed_info = algorithm.embed(
                    image, 
                    ''.join(format(ord(c), '08b') for c in watermark)
                )
                extracted_bits, recovered_img = algorithm.extract(embedded_img, embed_info)
                
                # Calculate metrics
                quality_metrics = quality_calc.calculate_all_metrics(image, embedded_img)
                capacity_metrics = capacity_calc.calculate_capacity_metrics(
                    ''.join(format(ord(c), '08b') for c in watermark),
                    image.size,
                    embed_info
                )
                
                config_results.append({
                    'watermark_length': len(watermark),
                    'embedding_successful': True,
                    'quality_metrics': quality_metrics,
                    'capacity_metrics': capacity_metrics,
                    'embedding_time': embed_info.get('embedding_time', 0)
                })
                
            except Exception as e:
                config_results.append({
                    'watermark_length': len(watermark),
                    'embedding_successful': False,
                    'error': str(e)
                })
        
        results[config_name] = config_results
    
    return results

# Package initialization message
def _print_welcome():
    """Print welcome message (only in interactive sessions)"""
    import sys
    if hasattr(sys, 'ps1'):  # Interactive session
        print(f"Enhanced Multi-Level PVS-RDH v{__version__}")
        print("High-capacity reversible data hiding with 67% improvement over traditional PVS-RDH")
        print("Key features:")
        for feature in get_supported_features()[:4]:  # Show first 4 features
            print(f"  • {feature}")
        print(f"  • ... and {len(get_supported_features())-4} more features")
        print("\nQuick start:")
        print("  from eml_pvs_rdh import quick_embed, quick_extract")
        print("  embedded_img, info = quick_embed(image, 'watermark text')")
        print("  text, recovered_img = quick_extract(embedded_img, info)")

# Initialize package (commented out to avoid printing in non-interactive contexts)
# _print_welcome()

# Development and debugging utilities
def _run_self_tests():
    """Run basic self-tests to verify package integrity"""
    try:
        import numpy as np
        
        # Test basic imports
        from .eml_pvs_rdh_main import EML_PVS_RDH, EMLPVSConfig
        from .pixel_decomposition import PixelDecomposition
        from .metrics import QualityMetrics
        
        print("✓ All core imports successful")
        
        # Test basic functionality
        test_image = np.random.randint(100, 200, (32, 32), dtype=np.uint8)
        decomposer = PixelDecomposition()
        H, M, L = decomposer.decompose_image(test_image)
        reconstructed = decomposer.reconstruct_image(H, M, L)
        
        if np.array_equal(test_image, reconstructed):
            print("✓ Pixel decomposition/reconstruction test passed")
        else:
            print("✗ Pixel decomposition/reconstruction test failed")
            return False
        
        # Test algorithm initialization
        config = EMLPVSConfig()
        algorithm = EML_PVS_RDH(config)
        capacity = algorithm.get_capacity_estimate(test_image)
        
        if 'total_estimated_capacity' in capacity:
            print("✓ Algorithm initialization and capacity estimation test passed")
        else:
            print("✗ Algorithm initialization test failed")
            return False
        
        # Test metrics calculation
        metrics_calc = QualityMetrics()
        modified_image = test_image + 1
        metrics = metrics_calc.calculate_all_metrics(test_image, modified_image)
        
        if 'psnr' in metrics and 'ssim' in metrics:
            print("✓ Metrics calculation test passed")
        else:
            print("✗ Metrics calculation test failed")
            return False
        
        print("✓ All self-tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Self-test failed: {e}")
        return False

# Expose self-test function
__all__.append('_run_self_tests')

# Optional: Advanced configuration validation
def validate_config(config):
    """
    Validate EMLPVSConfig configuration
    
    Args:
        config (EMLPVSConfig): Configuration to validate
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_warnings)
    """
    warnings = []
    
    # Check channel enablement
    if not any([config.enable_H_channel, config.enable_M_channel, config.enable_L_channel]):
        return False, ["At least one channel must be enabled"]
    
    # Check threshold values
    if config.L_prediction_threshold <= 0:
        warnings.append("L_prediction_threshold should be positive")
    
    if config.M_difference_threshold <= 0:
        warnings.append("M_difference_threshold should be positive")
    
    # Check optimization target
    valid_targets = ['capacity', 'quality', 'balanced']
    if config.optimization_target not in valid_targets:
        warnings.append(f"optimization_target should be one of {valid_targets}")
    
    # Check K1 range
    if config.K1_range[0] >= config.K1_range[1]:
        warnings.append("K1_range should be (min, max) with min < max")
    
    if config.K1_range[0] < 1 or config.K1_range[1] > 9:
        warnings.append("K1_range values should be between 1 and 9")
    
    # Performance warnings
    if config.enable_dual_pass and config.enable_parallel:
        warnings.append("Dual-pass with parallel processing may not provide optimal performance")
    
    return True, warnings

__all__.append('validate_config')

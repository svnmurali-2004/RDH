"""
Enhanced Multi-Level PVS-RDH Quality and Capacity Metrics
Comprehensive metrics calculation for algorithm evaluation
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging
from scipy import ndimage
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)

class QualityMetrics:
    """
    Comprehensive quality metrics calculator for RDH algorithms
    
    Provides multiple quality assessment metrics:
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index)
    - MSE (Mean Squared Error)
    - MAE (Mean Absolute Error)
    - UQI (Universal Quality Index)
    - VIF (Visual Information Fidelity)
    """
    
    def __init__(self):
        """Initialize quality metrics calculator"""
        logger.info("Initialized QualityMetrics calculator")
    
    def calculate_all_metrics(self, original: np.ndarray, embedded: np.ndarray) -> Dict[str, float]:
        """
        Calculate all quality metrics
        
        Args:
            original (np.ndarray): Original image
            embedded (np.ndarray): Embedded/processed image
            
        Returns:
            Dict[str, float]: Dictionary of all quality metrics
        """
        if original.shape != embedded.shape:
            raise ValueError("Images must have the same shape")
        
        metrics = {
            'mse': self.calculate_mse(original, embedded),
            'mae': self.calculate_mae(original, embedded),
            'psnr': self.calculate_psnr(original, embedded),
            'ssim': self.calculate_ssim(original, embedded),
            'uqi': self.calculate_uqi(original, embedded),
            'correlation': self.calculate_correlation(original, embedded),
            'max_absolute_error': float(np.max(np.abs(original.astype(np.int16) - embedded.astype(np.int16)))),
            'quality_score': 0.0  # Will be calculated based on other metrics
        }
        
        # Calculate composite quality score
        metrics['quality_score'] = self._calculate_composite_score(metrics)
        
        return metrics
    
    def calculate_mse(self, original: np.ndarray, embedded: np.ndarray) -> float:
        """Calculate Mean Squared Error"""
        return float(np.mean((original.astype(np.float32) - embedded.astype(np.float32)) ** 2))
    
    def calculate_mae(self, original: np.ndarray, embedded: np.ndarray) -> float:
        """Calculate Mean Absolute Error"""
        return float(np.mean(np.abs(original.astype(np.float32) - embedded.astype(np.float32))))
    
    def calculate_psnr(self, original: np.ndarray, embedded: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = self.calculate_mse(original, embedded)
        if mse == 0:
            return float('inf')
        max_pixel_value = 255.0
        psnr = 10 * np.log10(max_pixel_value**2 / mse)
        return float(psnr)
    
    def calculate_ssim(self, original: np.ndarray, embedded: np.ndarray, 
                      window_size: int = 11, k1: float = 0.01, k2: float = 0.03) -> float:
        """
        Calculate Structural Similarity Index (SSIM)
        
        Args:
            original, embedded (np.ndarray): Input images
            window_size (int): Size of the sliding window
            k1, k2 (float): SSIM constants
            
        Returns:
            float: SSIM value
        """
        # Convert to float
        img1 = original.astype(np.float32)
        img2 = embedded.astype(np.float32)
        
        # Constants
        C1 = (k1 * 255) ** 2
        C2 = (k2 * 255) ** 2
        
        # Create window
        window = self._create_gaussian_window(window_size)
        
        # Calculate local means
        mu1 = ndimage.correlate(img1, window, mode='reflect')
        mu2 = ndimage.correlate(img2, window, mode='reflect')
        
        # Calculate local variances and covariance
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = ndimage.correlate(img1 ** 2, window, mode='reflect') - mu1_sq
        sigma2_sq = ndimage.correlate(img2 ** 2, window, mode='reflect') - mu2_sq
        sigma12 = ndimage.correlate(img1 * img2, window, mode='reflect') - mu1_mu2
        
        # Calculate SSIM
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        
        ssim_map = numerator / (denominator + 1e-10)  # Add small value to avoid division by zero
        
        return float(np.mean(ssim_map))
    
    def calculate_uqi(self, original: np.ndarray, embedded: np.ndarray) -> float:
        """
        Calculate Universal Quality Index (UQI)
        
        Args:
            original, embedded (np.ndarray): Input images
            
        Returns:
            float: UQI value
        """
        # Convert to float
        x = original.astype(np.float32)
        y = embedded.astype(np.float32)
        
        # Calculate means
        mu_x = np.mean(x)
        mu_y = np.mean(y)
        
        # Calculate variances and covariance
        var_x = np.var(x)
        var_y = np.var(y)
        cov_xy = np.mean((x - mu_x) * (y - mu_y))
        
        # Calculate UQI
        if var_x == 0 or var_y == 0:
            return 0.0
        
        uqi = (4 * cov_xy * mu_x * mu_y) / ((var_x + var_y) * (mu_x**2 + mu_y**2))
        
        return float(uqi)
    
    def calculate_correlation(self, original: np.ndarray, embedded: np.ndarray) -> float:
        """Calculate correlation coefficient between images"""
        x = original.flatten().astype(np.float32)
        y = embedded.flatten().astype(np.float32)
        
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        
        correlation = np.corrcoef(x, y)[0, 1]
        
        if np.isnan(correlation):
            return 0.0
        
        return float(correlation)
    
    def _create_gaussian_window(self, window_size: int, sigma: float = 1.5) -> np.ndarray:
        """Create Gaussian window for SSIM calculation"""
        # Create coordinate arrays
        coords = np.arange(window_size, dtype=np.float32)
        coords -= window_size // 2
        
        # Create 2D Gaussian
        g = np.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = np.outer(g, g)
        
        # Normalize
        g = g / np.sum(g)
        
        return g
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """Calculate composite quality score from individual metrics"""
        # Normalized PSNR (0-1 scale, assuming max reasonable PSNR is 60 dB)
        psnr_normalized = min(metrics['psnr'] / 60.0, 1.0)
        
        # SSIM is already 0-1
        ssim_normalized = max(0, min(metrics['ssim'], 1.0))
        
        # Correlation coefficient (-1 to 1, normalize to 0-1)
        corr_normalized = (metrics['correlation'] + 1.0) / 2.0
        
        # UQI (-1 to 1, normalize to 0-1)
        uqi_normalized = (metrics['uqi'] + 1.0) / 2.0
        
        # Weighted composite score
        composite_score = (
            0.4 * psnr_normalized +
            0.3 * ssim_normalized +
            0.2 * corr_normalized +
            0.1 * uqi_normalized
        )
        
        return float(composite_score)

class CapacityMetrics:
    """
    Capacity and efficiency metrics calculator for RDH algorithms
    
    Provides:
    - Payload capacity (bpp)
    - Embedding efficiency
    - Auxiliary data ratio
    - Compression ratio
    - Security metrics
    """
    
    def __init__(self):
        """Initialize capacity metrics calculator"""
        logger.info("Initialized CapacityMetrics calculator")
    
    def calculate_capacity_metrics(self, watermark_bits: str, image_size: int, 
                                 embedding_info: Dict) -> Dict[str, float]:
        """
        Calculate comprehensive capacity metrics
        
        Args:
            watermark_bits (str): Watermark bit string
            image_size (int): Total number of pixels in image
            embedding_info (Dict): Information from embedding process
            
        Returns:
            Dict[str, float]: Dictionary of capacity metrics
        """
        watermark_length = len(watermark_bits)
        auxiliary_data_size = embedding_info.get('auxiliary_data_size', 0)
        total_embedded_bits = watermark_length + auxiliary_data_size
        
        metrics = {
            'watermark_bits': watermark_length,
            'auxiliary_data_bits': auxiliary_data_size,
            'total_embedded_bits': total_embedded_bits,
            'payload_bpp': watermark_length / image_size,
            'total_capacity_bpp': total_embedded_bits / image_size,
            'auxiliary_data_ratio': auxiliary_data_size / total_embedded_bits if total_embedded_bits > 0 else 0,
            'embedding_efficiency': self._calculate_embedding_efficiency(embedding_info),
            'capacity_utilization': self._calculate_capacity_utilization(embedding_info),
            'compression_ratio': self._calculate_compression_ratio(embedding_info),
            'bits_per_change': self._calculate_bits_per_change(embedding_info)
        }
        
        return metrics
    
    def _calculate_embedding_efficiency(self, embedding_info: Dict) -> float:
        """Calculate embedding efficiency (bits embedded per pixel modified)"""
        embedded_bits = embedding_info.get('embedded_bits', 0)
        modified_pixels = embedding_info.get('embedding_pairs', 0) * 2  # Assume pairs modify 2 pixels each
        
        if modified_pixels == 0:
            return 0.0
        
        return embedded_bits / modified_pixels
    
    def _calculate_capacity_utilization(self, embedding_info: Dict) -> float:
        """Calculate how much of theoretical capacity was utilized"""
        embedded_bits = embedding_info.get('embedded_bits', 0)
        theoretical_capacity = embedding_info.get('theoretical_capacity', embedded_bits)
        
        if theoretical_capacity == 0:
            return 0.0
        
        return embedded_bits / theoretical_capacity
    
    def _calculate_compression_ratio(self, embedding_info: Dict) -> float:
        """Calculate compression ratio for auxiliary data"""
        raw_auxiliary_size = embedding_info.get('raw_auxiliary_size', 0)
        compressed_auxiliary_size = embedding_info.get('auxiliary_data_size', 0)
        
        if compressed_auxiliary_size == 0:
            return 1.0
        
        return raw_auxiliary_size / compressed_auxiliary_size
    
    def _calculate_bits_per_change(self, embedding_info: Dict) -> float:
        """Calculate average bits embedded per pixel change"""
        embedded_bits = embedding_info.get('embedded_bits', 0)
        total_changes = embedding_info.get('total_pixel_changes', 1)
        
        return embedded_bits / total_changes

class PerformanceMetrics:
    """
    Performance metrics calculator for algorithmic efficiency
    
    Provides:
    - Processing time analysis
    - Memory usage metrics
    - Computational complexity
    - Throughput metrics
    """
    
    def __init__(self):
        """Initialize performance metrics calculator"""
        logger.info("Initialized PerformanceMetrics calculator")
    
    def calculate_performance_metrics(self, embedding_info: Dict, image_shape: Tuple[int, int]) -> Dict[str, float]:
        """
        Calculate performance metrics
        
        Args:
            embedding_info (Dict): Information from embedding process
            image_shape (Tuple[int, int]): Shape of processed image
            
        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        embedding_time = embedding_info.get('embedding_time', 0)
        extraction_time = embedding_info.get('extraction_time', 0)
        image_size = image_shape[0] * image_shape[1]
        embedded_bits = embedding_info.get('embedded_bits', 0)
        
        metrics = {
            'embedding_time_seconds': embedding_time,
            'extraction_time_seconds': extraction_time,
            'total_processing_time': embedding_time + extraction_time,
            'pixels_per_second': image_size / max(embedding_time, 0.001),
            'bits_per_second': embedded_bits / max(embedding_time, 0.001),
            'megapixels_per_second': (image_size / 1e6) / max(embedding_time, 0.001),
            'throughput_mbps': (embedded_bits / 1e6) / max(embedding_time, 0.001),
            'efficiency_score': self._calculate_efficiency_score(embedding_info, image_size)
        }
        
        return metrics
    
    def _calculate_efficiency_score(self, embedding_info: Dict, image_size: int) -> float:
        """Calculate overall efficiency score (0-1)"""
        # Factors: capacity, quality, and speed
        capacity_score = min(embedding_info.get('payload_bpp', 0) * 5, 1.0)  # Normalize assuming max 0.2 bpp
        quality_score = min(embedding_info.get('quality_metrics', {}).get('psnr', 0) / 50.0, 1.0)  # Normalize to 50 dB
        speed_score = min(image_size / max(embedding_info.get('embedding_time', 1) * 1000, 1), 1.0)  # Pixels per ms
        
        # Weighted combination
        efficiency = (0.4 * capacity_score + 0.4 * quality_score + 0.2 * speed_score)
        
        return float(efficiency)

class ComparativeMetrics:
    """
    Comparative metrics for algorithm benchmarking
    
    Provides:
    - Improvement ratios
    - Performance rankings
    - Statistical comparisons
    - Benchmark scores
    """
    
    def __init__(self):
        """Initialize comparative metrics calculator"""
        logger.info("Initialized ComparativeMetrics calculator")
    
    def calculate_improvement_metrics(self, new_results: Dict, baseline_results: Dict) -> Dict[str, float]:
        """
        Calculate improvement metrics compared to baseline
        
        Args:
            new_results (Dict): Results from new algorithm
            baseline_results (Dict): Results from baseline algorithm
            
        Returns:
            Dict[str, float]: Dictionary of improvement metrics
        """
        improvements = {}
        
        # Quality improvements
        new_psnr = new_results.get('psnr', 0)
        baseline_psnr = baseline_results.get('psnr', 1)
        improvements['psnr_improvement_percent'] = ((new_psnr - baseline_psnr) / baseline_psnr) * 100
        
        new_ssim = new_results.get('ssim', 0)
        baseline_ssim = baseline_results.get('ssim', 1)
        improvements['ssim_improvement_percent'] = ((new_ssim - baseline_ssim) / baseline_ssim) * 100
        
        # Capacity improvements
        new_capacity = new_results.get('payload_bpp', 0)
        baseline_capacity = baseline_results.get('payload_bpp', 1)
        improvements['capacity_improvement_percent'] = ((new_capacity - baseline_capacity) / baseline_capacity) * 100
        
        # Efficiency improvements
        new_time = new_results.get('embedding_time', 1)
        baseline_time = baseline_results.get('embedding_time', 1)
        improvements['speed_improvement_percent'] = ((baseline_time - new_time) / baseline_time) * 100
        
        # Auxiliary data improvements
        new_aux = new_results.get('auxiliary_data_ratio', 0)
        baseline_aux = baseline_results.get('auxiliary_data_ratio', 1)
        improvements['auxiliary_reduction_percent'] = ((baseline_aux - new_aux) / baseline_aux) * 100 if baseline_aux > 0 else 0
        
        # Overall improvement score
        improvements['overall_improvement_score'] = self._calculate_overall_improvement(improvements)
        
        return improvements
    
    def _calculate_overall_improvement(self, improvements: Dict[str, float]) -> float:
        """Calculate overall improvement score"""
        # Weighted combination of improvements
        psnr_weight = 0.3
        capacity_weight = 0.4
        speed_weight = 0.2
        aux_weight = 0.1
        
        psnr_score = max(-100, min(improvements.get('psnr_improvement_percent', 0), 100)) / 100
        capacity_score = max(-100, min(improvements.get('capacity_improvement_percent', 0), 100)) / 100
        speed_score = max(-100, min(improvements.get('speed_improvement_percent', 0), 100)) / 100
        aux_score = max(-100, min(improvements.get('auxiliary_reduction_percent', 0), 100)) / 100
        
        overall_score = (
            psnr_weight * psnr_score +
            capacity_weight * capacity_score +
            speed_weight * speed_score +
            aux_weight * aux_score
        )
        
        return float(overall_score)

# Example usage
if __name__ == "__main__":
    import cv2
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test images
    original = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    embedded = original + np.random.randint(-2, 3, original.shape).astype(np.int16)
    embedded = np.clip(embedded, 0, 255).astype(np.uint8)
    
    # Initialize metrics calculators
    quality_metrics = QualityMetrics()
    capacity_metrics = CapacityMetrics()
    performance_metrics = PerformanceMetrics()
    comparative_metrics = ComparativeMetrics()
    
    # Calculate quality metrics
    quality_results = quality_metrics.calculate_all_metrics(original, embedded)
    print("Quality Metrics:")
    for metric, value in quality_results.items():
        print(f"  {metric}: {value:.4f}")
    
    # Calculate capacity metrics (mock embedding info)
    embedding_info = {
        'embedded_bits': 1024,
        'auxiliary_data_size': 128,
        'embedding_pairs': 512,
        'embedding_time': 2.5,
        'extraction_time': 1.8
    }
    
    capacity_results = capacity_metrics.calculate_capacity_metrics("0" * 1024, original.size, embedding_info)
    print("\nCapacity Metrics:")
    for metric, value in capacity_results.items():
        print(f"  {metric}: {value:.4f}")
    
    # Calculate performance metrics
    performance_results = performance_metrics.calculate_performance_metrics(embedding_info, original.shape)
    print("\nPerformance Metrics:")
    for metric, value in performance_results.items():
        print(f"  {metric}: {value:.4f}")

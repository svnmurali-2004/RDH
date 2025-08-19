"""
Enhanced Multi-Level Pixel Value Splitting RDH (EML-PVS-RDH)
Region classifier for intelligent image region analysis
"""

import numpy as np
from typing import Tuple, Optional
import logging
from scipy.ndimage import generic_filter, gaussian_filter, sobel

logger = logging.getLogger(__name__)

class RegionClassifier:
    """
    Intelligent region classifier for adaptive embedding strategies
    
    Classifies image regions into:
    - SMOOTH (0): Low variance, high correlation - use all channels (H, M, L)
    - EDGE (1): High gradient, medium variance - use M and L channels
    - TEXTURE (2): High variance, low correlation - use L channel only
    """
    
    def __init__(self, smooth_threshold: float = 100.0, edge_threshold: float = 400.0, window_size: int = 5):
        """
        Initialize region classifier
        
        Args:
            smooth_threshold (float): Variance threshold for smooth regions
            edge_threshold (float): Variance threshold for edge regions
            window_size (int): Window size for local analysis
        """
        self.smooth_threshold = smooth_threshold
        self.edge_threshold = edge_threshold
        self.window_size = window_size
        
        logger.info(f"Initialized RegionClassifier: smooth_th={smooth_threshold}, edge_th={edge_threshold}, window={window_size}")
    
    def classify_regions(self, image: np.ndarray) -> np.ndarray:
        """
        Classify image regions based on local statistics
        
        Args:
            image (np.ndarray): Input grayscale image
            
        Returns:
            np.ndarray: Region map (0=smooth, 1=edge, 2=texture)
        """
        if image.dtype != np.uint8:
            raise ValueError("Input image must be uint8 type")
        
        if len(image.shape) != 2:
            raise ValueError("Input must be grayscale (2D) image")
        
        height, width = image.shape
        region_map = np.zeros((height, width), dtype=np.uint8)
        
        # Calculate local variance
        local_variance = self._calculate_local_variance(image)
        
        # Calculate gradient magnitude
        gradient_magnitude = self._calculate_gradient_magnitude(image)
        
        # Calculate local correlation
        local_correlation = self._calculate_local_correlation(image)
        
        # Classify regions based on combined criteria
        for i in range(height):
            for j in range(width):
                variance = local_variance[i, j]
                gradient = gradient_magnitude[i, j]
                correlation = local_correlation[i, j]
                
                if variance < self.smooth_threshold and correlation > 0.8:
                    # Smooth region: low variance, high correlation
                    region_map[i, j] = 0  # SMOOTH
                elif gradient > 50 or (variance < self.edge_threshold and correlation > 0.5):
                    # Edge region: high gradient or medium variance with good correlation
                    region_map[i, j] = 1  # EDGE
                else:
                    # Texture region: high variance, low correlation
                    region_map[i, j] = 2  # TEXTURE
        
        # Post-processing: smooth region map to reduce noise
        region_map = self._smooth_region_map(region_map)
        
        # Log region statistics
        self._log_region_statistics(region_map)
        
        return region_map
    
    def _calculate_local_variance(self, image: np.ndarray) -> np.ndarray:
        """Calculate local variance using sliding window"""
        def variance_filter(values):
            return np.var(values)
        
        # Apply variance filter with padding
        local_variance = generic_filter(
            image.astype(np.float32),
            variance_filter,
            size=self.window_size,
            mode='reflect'
        )
        
        return local_variance
    
    def _calculate_gradient_magnitude(self, image: np.ndarray) -> np.ndarray:
        """Calculate gradient magnitude using Sobel operators"""
        # Calculate gradients in x and y directions
        grad_x = sobel(image.astype(np.float32), axis=1)
        grad_y = sobel(image.astype(np.float32), axis=0)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return gradient_magnitude
    
    def _calculate_local_correlation(self, image: np.ndarray) -> np.ndarray:
        """Calculate local correlation coefficient"""
        height, width = image.shape
        correlation_map = np.zeros((height, width), dtype=np.float32)
        
        # Half window size
        half_win = self.window_size // 2
        
        for i in range(half_win, height - half_win):
            for j in range(half_win, width - half_win):
                # Extract local window
                window = image[i-half_win:i+half_win+1, j-half_win:j+half_win+1].astype(np.float32)
                
                # Calculate correlation between adjacent pixels
                corr_h = self._pixel_correlation(window[:, :-1], window[:, 1:])  # Horizontal
                corr_v = self._pixel_correlation(window[:-1, :], window[1:, :])  # Vertical
                
                # Average correlation
                correlation_map[i, j] = (corr_h + corr_v) / 2
        
        # Fill borders with nearest values
        correlation_map = self._fill_borders(correlation_map, half_win)
        
        return correlation_map
    
    def _pixel_correlation(self, arr1: np.ndarray, arr2: np.ndarray) -> float:
        """Calculate correlation coefficient between two arrays"""
        if arr1.size == 0 or arr2.size == 0:
            return 0.0
        
        # Flatten arrays
        x = arr1.flatten()
        y = arr2.flatten()
        
        # Calculate correlation coefficient
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        
        correlation = np.corrcoef(x, y)[0, 1]
        
        # Handle NaN values
        if np.isnan(correlation):
            return 0.0
        
        return float(correlation)
    
    def _fill_borders(self, array: np.ndarray, border_size: int) -> np.ndarray:
        """Fill border regions with nearest valid values"""
        height, width = array.shape
        
        # Top border
        for i in range(border_size):
            array[i, :] = array[border_size, :]
        
        # Bottom border
        for i in range(height - border_size, height):
            array[i, :] = array[height - border_size - 1, :]
        
        # Left border
        for j in range(border_size):
            array[:, j] = array[:, border_size]
        
        # Right border
        for j in range(width - border_size, width):
            array[:, j] = array[:, width - border_size - 1]
        
        return array
    
    def _smooth_region_map(self, region_map: np.ndarray) -> np.ndarray:
        """Apply smoothing to reduce noise in region classification"""
        # Apply median filter to reduce isolated pixels
        def median_filter_func(values):
            return np.median(values)
        
        smoothed = generic_filter(
            region_map.astype(np.float32),
            median_filter_func,
            size=3,
            mode='nearest'
        ).astype(np.uint8)
        
        return smoothed
    
    def _log_region_statistics(self, region_map: np.ndarray):
        """Log statistics about region classification"""
        unique, counts = np.unique(region_map, return_counts=True)
        total_pixels = region_map.size
        
        region_names = {0: 'SMOOTH', 1: 'EDGE', 2: 'TEXTURE'}
        
        logger.info("Region classification statistics:")
        for region_type, count in zip(unique, counts):
            percentage = (count / total_pixels) * 100
            name = region_names.get(region_type, f'UNKNOWN_{region_type}')
            logger.info(f"  {name}: {count} pixels ({percentage:.1f}%)")
    
    def get_region_embedding_capacity(self, region_map: np.ndarray, H: np.ndarray, M: np.ndarray, L: np.ndarray) -> dict:
        """
        Estimate embedding capacity for each region type
        
        Args:
            region_map (np.ndarray): Region classification map
            H, M, L (np.ndarray): Decomposed image channels
            
        Returns:
            dict: Capacity estimates by region type
        """
        capacity_estimates = {
            'SMOOTH': {'pixels': 0, 'estimated_capacity': 0, 'channels_used': ['H', 'M', 'L']},
            'EDGE': {'pixels': 0, 'estimated_capacity': 0, 'channels_used': ['M', 'L']},
            'TEXTURE': {'pixels': 0, 'estimated_capacity': 0, 'channels_used': ['L']}
        }
        
        # Count pixels by region type
        smooth_mask = (region_map == 0)
        edge_mask = (region_map == 1)
        texture_mask = (region_map == 2)
        
        capacity_estimates['SMOOTH']['pixels'] = np.sum(smooth_mask)
        capacity_estimates['EDGE']['pixels'] = np.sum(edge_mask)
        capacity_estimates['TEXTURE']['pixels'] = np.sum(texture_mask)
        
        # Estimate capacity for each region type
        # Smooth regions: can use all channels effectively (high capacity)
        smooth_capacity = self._estimate_smooth_capacity(H, M, L, smooth_mask)
        capacity_estimates['SMOOTH']['estimated_capacity'] = smooth_capacity
        
        # Edge regions: use M and L channels (medium capacity)
        edge_capacity = self._estimate_edge_capacity(M, L, edge_mask)
        capacity_estimates['EDGE']['estimated_capacity'] = edge_capacity
        
        # Texture regions: L channel only (low capacity)
        texture_capacity = self._estimate_texture_capacity(L, texture_mask)
        capacity_estimates['TEXTURE']['estimated_capacity'] = texture_capacity
        
        # Calculate total capacity
        total_capacity = smooth_capacity + edge_capacity + texture_capacity
        total_pixels = region_map.size
        
        capacity_estimates['TOTAL'] = {
            'estimated_capacity_bits': total_capacity,
            'estimated_capacity_bpp': total_capacity / total_pixels if total_pixels > 0 else 0
        }
        
        return capacity_estimates
    
    def _estimate_smooth_capacity(self, H: np.ndarray, M: np.ndarray, L: np.ndarray, mask: np.ndarray) -> int:
        """Estimate capacity for smooth regions using all channels"""
        if not np.any(mask):
            return 0
        
        # Count similar adjacent pairs in each channel within smooth regions
        h_pairs = self._count_masked_similar_pairs(H, mask)
        m_pairs = self._count_masked_similar_pairs(M, mask)
        l_predictable = self._count_masked_predictable_pixels(L, mask)
        
        # Smooth regions can utilize multiple channels effectively
        estimated_capacity = int(h_pairs * 0.3 + m_pairs * 0.5 + l_predictable * 0.8)
        
        return estimated_capacity
    
    def _estimate_edge_capacity(self, M: np.ndarray, L: np.ndarray, mask: np.ndarray) -> int:
        """Estimate capacity for edge regions using M and L channels"""
        if not np.any(mask):
            return 0
        
        # Edge regions: use M and L channels with edge-preserving techniques
        m_pairs = self._count_masked_similar_pairs(M, mask)
        l_predictable = self._count_masked_predictable_pixels(L, mask)
        
        estimated_capacity = int(m_pairs * 0.3 + l_predictable * 0.6)
        
        return estimated_capacity
    
    def _estimate_texture_capacity(self, L: np.ndarray, mask: np.ndarray) -> int:
        """Estimate capacity for texture regions using L channel only"""
        if not np.any(mask):
            return 0
        
        # Texture regions: L channel only with advanced prediction
        l_predictable = self._count_masked_predictable_pixels(L, mask)
        
        estimated_capacity = int(l_predictable * 0.4)  # Lower capacity due to complexity
        
        return estimated_capacity
    
    def _count_masked_similar_pairs(self, channel: np.ndarray, mask: np.ndarray) -> int:
        """Count similar adjacent pairs within masked region"""
        height, width = channel.shape
        count = 0
        
        # Horizontal pairs
        for i in range(height):
            for j in range(width - 1):
                if mask[i, j] and mask[i, j + 1]:
                    if channel[i, j] == channel[i, j + 1]:
                        count += 1
        
        # Vertical pairs
        for i in range(height - 1):
            for j in range(width):
                if mask[i, j] and mask[i + 1, j]:
                    if channel[i, j] == channel[i + 1, j]:
                        count += 1
        
        return count
    
    def _count_masked_predictable_pixels(self, channel: np.ndarray, mask: np.ndarray) -> int:
        """Count predictable pixels within masked region"""
        height, width = channel.shape
        count = 0
        
        for i in range(1, height):
            for j in range(1, width):
                if mask[i, j] and mask[i-1, j] and mask[i, j-1]:
                    # Simple predictor: average of left and top neighbors
                    predicted = (int(channel[i, j-1]) + int(channel[i-1, j])) // 2
                    if abs(int(channel[i, j]) - predicted) <= 1:
                        count += 1
        
        return count
    
    def get_adaptive_thresholds(self, region_map: np.ndarray, H: np.ndarray, M: np.ndarray, L: np.ndarray) -> dict:
        """
        Calculate adaptive thresholds for each region type
        
        Args:
            region_map (np.ndarray): Region classification map
            H, M, L (np.ndarray): Decomposed image channels
            
        Returns:
            dict: Adaptive thresholds for each region
        """
        thresholds = {
            'SMOOTH': {'L_threshold': 1.0, 'M_threshold': 1.5, 'H_threshold': 0.0},
            'EDGE': {'L_threshold': 1.5, 'M_threshold': 2.0, 'H_threshold': 0.0},
            'TEXTURE': {'L_threshold': 2.0, 'M_threshold': 3.0, 'H_threshold': 0.0}
        }
        
        # Calculate region-specific statistics
        for region_type in [0, 1, 2]:  # SMOOTH, EDGE, TEXTURE
            mask = (region_map == region_type)
            if np.any(mask):
                # Calculate local variance for this region
                region_variance = np.var(L[mask])
                
                # Adapt thresholds based on local characteristics
                if region_type == 0:  # SMOOTH
                    thresholds['SMOOTH']['L_threshold'] = max(0.5, min(1.5, region_variance * 0.1))
                elif region_type == 1:  # EDGE
                    thresholds['EDGE']['L_threshold'] = max(1.0, min(2.0, region_variance * 0.15))
                else:  # TEXTURE
                    thresholds['TEXTURE']['L_threshold'] = max(1.5, min(3.0, region_variance * 0.2))
        
        logger.info(f"Adaptive thresholds calculated: {thresholds}")
        return thresholds

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test image with different regions
    test_image = np.zeros((128, 128), dtype=np.uint8)
    
    # Smooth region (top-left)
    test_image[:64, :64] = 150 + np.random.normal(0, 5, (64, 64)).astype(int)
    test_image[:64, :64] = np.clip(test_image[:64, :64], 0, 255).astype(np.uint8)
    
    # Edge region (top-right) - create vertical edges
    for i in range(64):
        for j in range(64, 128):
            if (j - 64) % 10 < 5:
                test_image[i, j] = 100
            else:
                test_image[i, j] = 200
    
    # Texture region (bottom half)
    test_image[64:, :] = np.random.randint(50, 200, (64, 128), dtype=np.uint8)
    
    # Initialize classifier
    classifier = RegionClassifier(smooth_threshold=100.0, edge_threshold=400.0)
    
    # Classify regions
    region_map = classifier.classify_regions(test_image)
    
    print(f"Region map shape: {region_map.shape}")
    print(f"Region types found: {np.unique(region_map)}")
    
    # Test with decomposed channels (dummy decomposition for testing)
    from pixel_decomposition import PixelDecomposition
    decomposer = PixelDecomposition()
    H, M, L = decomposer.decompose_image(test_image)
    
    # Get capacity estimates
    capacity_estimates = classifier.get_region_embedding_capacity(region_map, H, M, L)
    print(f"Total estimated capacity: {capacity_estimates['TOTAL']['estimated_capacity_bpp']:.4f} bpp")
    
    # Get adaptive thresholds
    adaptive_thresholds = classifier.get_adaptive_thresholds(region_map, H, M, L)
    print(f"Adaptive thresholds: {adaptive_thresholds}")

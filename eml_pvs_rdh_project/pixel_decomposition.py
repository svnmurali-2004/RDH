"""
Enhanced Multi-Level Pixel Value Splitting RDH (EML-PVS-RDH)
Core pixel decomposition module for hierarchical three-level splitting
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PixelDecomposition:
    """
    Hierarchical three-level pixel decomposition for enhanced RDH
    
    Splits 8-bit pixels into:
    - H (Hundreds place): Most significant, high distortion impact
    - M (Tens place): Medium significance, moderate distortion  
    - L (Units place): Least significant, minimal distortion
    """
    
    def __init__(self):
        """Initialize pixel decomposition with validation"""
        self._validate_pixel_range()
    
    def _validate_pixel_range(self):
        """Validate that 8-bit pixel range [0-255] works with decomposition"""
        test_pixels = [0, 127, 255]
        for pixel in test_pixels:
            h, m, l = self._decompose_single_pixel(pixel)
            reconstructed = self._reconstruct_single_pixel(h, m, l)
            if reconstructed != pixel:
                raise ValueError(f"Decomposition validation failed for pixel {pixel}")
        logger.debug("Pixel decomposition validation passed")
    
    def _decompose_single_pixel(self, pixel: int) -> Tuple[int, int, int]:
        """
        Decompose single pixel into H, M, L components
        
        Args:
            pixel (int): 8-bit pixel value [0-255]
            
        Returns:
            Tuple[int, int, int]: (H, M, L) components
            
        Mathematical formulation:
            H(p,q) = ⌊I(p,q)/100⌋     (range: 0-2)
            M(p,q) = ⌊(I(p,q) mod 100)/10⌋  (range: 0-9)
            L(p,q) = I(p,q) mod 10     (range: 0-9)
        """
        if not (0 <= pixel <= 255):
            raise ValueError(f"Pixel value {pixel} out of range [0-255]")
            
        h = pixel // 100          # Hundreds place: 0, 1, 2
        remainder = pixel % 100   # 0-99
        m = remainder // 10       # Tens place: 0-9
        l = remainder % 10        # Units place: 0-9
        
        return h, m, l
    
    def _reconstruct_single_pixel(self, h: int, m: int, l: int) -> int:
        """
        Reconstruct pixel from H, M, L components
        
        Args:
            h (int): Hundreds component [0-2]
            m (int): Tens component [0-9]
            l (int): Units component [0-9]
            
        Returns:
            int: Reconstructed pixel value
            
        Mathematical formulation:
            I(p,q) = 100×H(p,q) + 10×M(p,q) + L(p,q)
        """
        # Validate component ranges
        if not (0 <= h <= 2):
            raise ValueError(f"H component {h} out of range [0-2]")
        if not (0 <= m <= 9):
            raise ValueError(f"M component {m} out of range [0-9]")
        if not (0 <= l <= 9):
            raise ValueError(f"L component {l} out of range [0-9]")
            
        pixel = 100 * h + 10 * m + l
        
        # Additional validation
        if not (0 <= pixel <= 255):
            raise ValueError(f"Reconstructed pixel {pixel} out of valid range")
            
        return pixel
    
    def decompose_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decompose entire image into H, M, L channel matrices
        
        Args:
            image (np.ndarray): Input grayscale image [0-255]
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (H, M, L) channel matrices
        """
        if image.dtype != np.uint8:
            raise ValueError("Input image must be uint8 type")
        
        if len(image.shape) != 2:
            raise ValueError("Input must be grayscale (2D) image")
        
        height, width = image.shape
        
        # Initialize output arrays with appropriate data types
        H = np.zeros((height, width), dtype=np.uint8)  # Range: 0-2
        M = np.zeros((height, width), dtype=np.uint8)  # Range: 0-9  
        L = np.zeros((height, width), dtype=np.uint8)  # Range: 0-9
        
        # Vectorized decomposition for efficiency
        H = image // 100
        remainder = image % 100
        M = remainder // 10
        L = remainder % 10
        
        logger.info(f"Decomposed image of size {image.shape}")
        logger.debug(f"H channel range: [{H.min()}, {H.max()}]")
        logger.debug(f"M channel range: [{M.min()}, {M.max()}]")
        logger.debug(f"L channel range: [{L.min()}, {L.max()}]")
        
        return H, M, L
    
    def reconstruct_image(self, H: np.ndarray, M: np.ndarray, L: np.ndarray) -> np.ndarray:
        """
        Reconstruct image from H, M, L channel matrices
        
        Args:
            H (np.ndarray): Hundreds channel matrix
            M (np.ndarray): Tens channel matrix  
            L (np.ndarray): Units channel matrix
            
        Returns:
            np.ndarray: Reconstructed grayscale image
        """
        # Validate input shapes match
        if not (H.shape == M.shape == L.shape):
            raise ValueError("All channel matrices must have same shape")
        
        # Validate channel value ranges
        if not (H.min() >= 0 and H.max() <= 2):
            raise ValueError(f"H channel values out of range [0-2]: [{H.min()}, {H.max()}]")
        if not (M.min() >= 0 and M.max() <= 9):
            raise ValueError(f"M channel values out of range [0-9]: [{M.min()}, {M.max()}]")
        if not (L.min() >= 0 and L.max() <= 9):
            raise ValueError(f"L channel values out of range [0-9]: [{L.min()}, {L.max()}]")
        
        # Reconstruct using vectorized operations
        reconstructed = (100 * H + 10 * M + L).astype(np.uint8)
        
        # Validate final image range
        if not (reconstructed.min() >= 0 and reconstructed.max() <= 255):
            raise ValueError(f"Reconstructed image out of range [0-255]: [{reconstructed.min()}, {reconstructed.max()}]")
        
        logger.info(f"Reconstructed image of size {reconstructed.shape}")
        logger.debug(f"Pixel value range: [{reconstructed.min()}, {reconstructed.max()}]")
        
        return reconstructed
    
    def analyze_channel_statistics(self, H: np.ndarray, M: np.ndarray, L: np.ndarray) -> dict:
        """
        Analyze statistical properties of each channel for optimization
        
        Args:
            H, M, L (np.ndarray): Channel matrices
            
        Returns:
            dict: Statistical analysis results
        """
        stats = {
            'H_channel': {
                'mean': float(H.mean()),
                'std': float(H.std()),
                'entropy': self._calculate_entropy(H),
                'unique_values': len(np.unique(H)),
                'range': [int(H.min()), int(H.max())]
            },
            'M_channel': {
                'mean': float(M.mean()),
                'std': float(M.std()),
                'entropy': self._calculate_entropy(M),
                'unique_values': len(np.unique(M)),
                'range': [int(M.min()), int(M.max())]
            },
            'L_channel': {
                'mean': float(L.mean()),
                'std': float(L.std()),
                'entropy': self._calculate_entropy(L),
                'unique_values': len(np.unique(L)),
                'range': [int(L.min()), int(L.max())]
            }
        }
        
        # Calculate embedding potential for each channel
        stats['embedding_potential'] = {
            'H_pairs': self._count_similar_pairs(H),
            'M_pairs': self._count_similar_pairs(M), 
            'L_predictable': self._count_predictable_pixels(L)
        }
        
        return stats
    
    def _calculate_entropy(self, channel: np.ndarray) -> float:
        """Calculate Shannon entropy for a channel"""
        unique, counts = np.unique(channel, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return float(entropy)
    
    def _count_similar_pairs(self, channel: np.ndarray) -> int:
        """Count adjacent pairs with identical values"""
        # Horizontal pairs
        h_pairs = np.sum(channel[:, :-1] == channel[:, 1:])
        # Vertical pairs  
        v_pairs = np.sum(channel[:-1, :] == channel[1:, :])
        return int(h_pairs + v_pairs)
    
    def _count_predictable_pixels(self, channel: np.ndarray) -> int:
        """Count pixels that are highly predictable"""
        height, width = channel.shape
        predictable = 0
        
        for i in range(1, height):
            for j in range(1, width):
                # Simple predictor: average of left and top neighbors
                predicted = (int(channel[i, j-1]) + int(channel[i-1, j])) // 2
                if abs(int(channel[i, j]) - predicted) <= 1:
                    predictable += 1
                    
        return predictable
    
    def get_channel_capacity_estimate(self, image: np.ndarray) -> dict:
        """
        Estimate embedding capacity for each channel
        
        Args:
            image (np.ndarray): Input grayscale image
            
        Returns:
            dict: Capacity estimates for each channel
        """
        H, M, L = self.decompose_image(image)
        stats = self.analyze_channel_statistics(H, M, L)
        
        total_pixels = image.size
        
        # Estimate capacity based on statistical properties
        capacity_estimates = {
            'H_channel': {
                'similar_pairs': stats['embedding_potential']['H_pairs'],
                'estimated_capacity_bpp': stats['embedding_potential']['H_pairs'] / total_pixels,
                'distortion_impact': 'HIGH',  # ±100 pixel change
                'recommendation': 'Use sparingly for critical data'
            },
            'M_channel': {
                'similar_pairs': stats['embedding_potential']['M_pairs'], 
                'estimated_capacity_bpp': stats['embedding_potential']['M_pairs'] / total_pixels,
                'distortion_impact': 'MEDIUM',  # ±10 pixel change
                'recommendation': 'Good for moderate capacity needs'
            },
            'L_channel': {
                'predictable_pixels': stats['embedding_potential']['L_predictable'],
                'estimated_capacity_bpp': stats['embedding_potential']['L_predictable'] / total_pixels,
                'distortion_impact': 'LOW',  # ±1 pixel change
                'recommendation': 'Primary embedding channel'
            },
            'total_estimated_capacity': (
                stats['embedding_potential']['H_pairs'] +
                stats['embedding_potential']['M_pairs'] +
                stats['embedding_potential']['L_predictable']
            ) / total_pixels
        }
        
        logger.info(f"Total estimated capacity: {capacity_estimates['total_estimated_capacity']:.4f} bpp")
        
        return capacity_estimates

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test image
    test_image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    
    # Initialize decomposer
    decomposer = PixelDecomposition()
    
    # Decompose image
    H, M, L = decomposer.decompose_image(test_image)
    
    # Reconstruct and verify
    reconstructed = decomposer.reconstruct_image(H, M, L)
    
    print(f"Perfect reconstruction: {np.array_equal(test_image, reconstructed)}")
    
    # Analyze capacity
    capacity = decomposer.get_channel_capacity_estimate(test_image)
    print(f"Estimated total capacity: {capacity['total_estimated_capacity']:.4f} bpp")
    
    # Channel statistics
    stats = decomposer.analyze_channel_statistics(H, M, L)
    for channel in ['H_channel', 'M_channel', 'L_channel']:
        print(f"{channel}: entropy={stats[channel]['entropy']:.3f}, unique={stats[channel]['unique_values']}")

# PVS-RDH: Pixel Value Splitting based Reversible Data Hiding Implementation
# Based on the paper: "Pixel value splitting based reversible data embedding scheme"
# Authors: Ankita Meenpal, Saikat Majumder, Madhu Oruganti

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import json

class PVS_RDH:
    """
    Pixel Value Splitting based Reversible Data Hiding (PVS-RDH) Implementation
    """
    
    def __init__(self, K1=4, K2=6):
        """
        Initialize PVS-RDH with offset parameters
        
        Args:
            K1 (int): Positive offset for Group-B modification (default: 4)
            K2 (int): Negative offset for Group-B modification (default: 6)
                     Note: K1 + K2 must equal 10
        """
        if K1 + K2 != 10:
            raise ValueError("K1 + K2 must equal 10")
        
        self.K1 = K1
        self.K2 = K2
        self.embedding_capacity = 0
        self.auxiliary_data = []
        
    def pixel_value_splitting(self, image):
        """
        Algorithm 1: Split pixel values into Group-A and Group-B
        
        Args:
            image (numpy.ndarray): 8-bit grayscale image
            
        Returns:
            tuple: (group_A, group_B) arrays
        """
        # Group-A: hundreds and tens digits (floor division by 10)
        group_A = image // 10
        
        # Group-B: ones digit (modulo 10)
        group_B = image % 10
        
        return group_A, group_B
    
    def find_embedding_pairs(self, group_A):
        """
        Find embedding pairs where adjacent Group-A values are equal
        
        Args:
            group_A (numpy.ndarray): Group-A values
            
        Returns:
            list: List of embedding pair positions [(row, col1, col2), ...]
        """
        embedding_pairs = []
        rows, cols = group_A.shape
        
        # Check horizontal pairs (row-wise)
        for i in range(rows):
            for j in range(cols - 1):
                # Condition: A(p,q) = A(p,q+1) AND A(p,q) > 1
                if group_A[i, j] == group_A[i, j + 1] and group_A[i, j] > 1:
                    embedding_pairs.append((i, j, j + 1))
        
        return embedding_pairs
    
    def generate_auxiliary_data(self, group_A, embedding_pairs):
        """
        Generate auxiliary data for underflow handling
        
        Args:
            group_A (numpy.ndarray): Group-A values
            embedding_pairs (list): List of embedding pairs
            
        Returns:
            list: Auxiliary data (underflow locations)
        """
        auxiliary_data = []
        
        for pair in embedding_pairs:
            i, j1, j2 = pair
            # Check for potential underflow (A(p,q+1) = 0)
            if group_A[i, j1] > 1 and group_A[i, j2] == 0:
                auxiliary_data.append((i, j2))
        
        return auxiliary_data
    
    def perform_histogram_shifting(self, group_A, embedding_pairs):
        """
        Perform histogram shifting to avoid conflicts
        
        Args:
            group_A (numpy.ndarray): Group-A values
            embedding_pairs (list): Embedding pairs
            
        Returns:
            numpy.ndarray: Modified Group-A after shifting
        """
        group_A_shifted = group_A.copy()
        rows, cols = group_A.shape
        
        # Apply shifting for pairs with difference >= 1
        for i in range(rows):
            for j in range(cols - 1):
                difference = group_A[i, j] - group_A[i, j + 1]
                if difference >= 1:
                    # Check if this pair is not an embedding pair
                    if (i, j, j + 1) not in embedding_pairs:
                        group_A_shifted[i, j + 1] -= 1
        
        return group_A_shifted
    
    def embed_watermark(self, image, watermark_bits):
        """
        Main embedding function
        
        Args:
            image (numpy.ndarray): Original 8-bit grayscale image
            watermark_bits (str): Binary string of watermark data
            
        Returns:
            tuple: (embedded_image, embedding_info)
        """
        print("Starting PVS-RDH Embedding Process...")
        
        # Step 1: Pixel Value Splitting
        group_A, group_B = self.pixel_value_splitting(image)
        print(f"Image split into Group-A and Group-B")
        
        # Step 2: Find embedding pairs
        embedding_pairs = self.find_embedding_pairs(group_A)
        self.embedding_capacity = len(embedding_pairs)
        print(f"Found {self.embedding_capacity} embedding pairs")
        
        if self.embedding_capacity == 0:
            raise ValueError("No embedding pairs found. Image may be too textured.")
        
        # Step 3: Generate auxiliary data
        self.auxiliary_data = self.generate_auxiliary_data(group_A, embedding_pairs)
        auxiliary_bits = len(self.auxiliary_data) * 16  # 16 bits per location
        print(f"Auxiliary data size: {auxiliary_bits} bits")
        
        # Step 4: Check if watermark fits
        available_capacity = self.embedding_capacity - auxiliary_bits
        if len(watermark_bits) > available_capacity:
            print(f"Warning: Watermark ({len(watermark_bits)} bits) truncated to fit capacity ({available_capacity} bits)")
            watermark_bits = watermark_bits[:available_capacity]
        
        # Step 5: Perform histogram shifting
        group_A_shifted = self.perform_histogram_shifting(group_A, embedding_pairs)
        
        # Step 6: Embed watermark bits
        group_A_embedded = group_A_shifted.copy()
        group_B_embedded = group_B.copy()
        
        # Combine auxiliary data and watermark
        total_data = self._encode_auxiliary_data() + watermark_bits
        
        for idx, bit in enumerate(total_data[:len(embedding_pairs)]):
            if idx >= len(embedding_pairs):
                break
                
            i, j1, j2 = embedding_pairs[idx]
            
            # Skip if would cause underflow
            if (i, j2) in self.auxiliary_data:
                continue
            
            if bit == '1':
                # Embed bit 1: subtract 1 from A(p,q+1)
                group_A_embedded[i, j2] -= 1
                
                # Apply Group-B offset for quality enhancement
                if group_B[i, j2] < self.K2:
                    group_B_embedded[i, j2] = (group_B[i, j2] + self.K1) % 10
                else:
                    group_B_embedded[i, j2] = (group_B[i, j2] - self.K2) % 10
        
        # Step 7: Reconstruct embedded image
        embedded_image = group_A_embedded * 10 + group_B_embedded
        
        # Calculate quality metrics
        mse = np.mean((image - embedded_image) ** 2)
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
        
        embedding_info = {
            'embedding_pairs': len(embedding_pairs),
            'auxiliary_data_size': auxiliary_bits,
            'watermark_length': len(watermark_bits),
            'psnr': psnr,
            'mse': mse,
            'embedding_pairs_list': embedding_pairs,
            'auxiliary_data': self.auxiliary_data,
            'K1': self.K1,
            'K2': self.K2
        }
        
        print(f"Embedding completed successfully!")
        print(f"PSNR: {psnr:.2f} dB")
        print(f"Embedded {len(watermark_bits)} watermark bits")
        
        return embedded_image, embedding_info
    
    def extract_watermark(self, embedded_image, embedding_info):
        """
        Extract watermark and recover original image
        
        Args:
            embedded_image (numpy.ndarray): Embedded image
            embedding_info (dict): Information from embedding process
            
        Returns:
            tuple: (extracted_watermark, recovered_image)
        """
        print("Starting PVS-RDH Extraction Process...")
        
        # Step 1: Split embedded image
        group_A_embedded, group_B_embedded = self.pixel_value_splitting(embedded_image)
        
        # Step 2: Extract watermark bits
        embedding_pairs = embedding_info['embedding_pairs_list']
        auxiliary_data = embedding_info['auxiliary_data']
        K1, K2 = embedding_info['K1'], embedding_info['K2']
        
        extracted_bits = []
        group_A_recovered = group_A_embedded.copy()
        group_B_recovered = group_B_embedded.copy()
        
        for idx, pair in enumerate(embedding_pairs):
            i, j1, j2 = pair
            
            # Skip auxiliary data locations
            if (i, j2) in auxiliary_data:
                extracted_bits.append('0')  # Placeholder
                continue
            
            # Extract bit based on difference
            difference = group_A_embedded[i, j1] - group_A_embedded[i, j2]
            
            if difference == 1:
                extracted_bits.append('1')
                # Recover Group-A
                group_A_recovered[i, j2] += 1
                # Recover Group-B
                if group_B_embedded[i, j2] < K2:
                    group_B_recovered[i, j2] = (group_B_embedded[i, j2] - K1) % 10
                else:
                    group_B_recovered[i, j2] = (group_B_embedded[i, j2] + K2) % 10
            else:
                extracted_bits.append('0')
        
        # Step 3: Reverse histogram shifting
        rows, cols = group_A_recovered.shape
        for i in range(rows):
            for j in range(cols - 1):
                if (i, j, j + 1) not in embedding_pairs:
                    difference = group_A_recovered[i, j] - group_A_recovered[i, j + 1]
                    if difference >= 2:
                        group_A_recovered[i, j + 1] += 1
        
        # Step 4: Reconstruct original image
        recovered_image = group_A_recovered * 10 + group_B_recovered
        
        # Step 5: Separate auxiliary data from watermark
        auxiliary_bits = len(auxiliary_data) * 16
        watermark_bits = ''.join(extracted_bits[auxiliary_bits:])
        
        print(f"Extraction completed successfully!")
        print(f"Extracted {len(watermark_bits)} bits")
        
        return watermark_bits, recovered_image
    
    def _encode_auxiliary_data(self):
        """
        Encode auxiliary data as binary string
        
        Returns:
            str: Binary representation of auxiliary data
        """
        if not self.auxiliary_data:
            return '0' * 16  # 16 bits representing length 0
        
        # Encode length (16 bits) + locations
        length_bits = format(len(self.auxiliary_data), '016b')
        location_bits = ''
        
        for i, j in self.auxiliary_data:
            # Encode each location as 16 bits (8 bits each for i, j)
            location_bits += format(i, '08b') + format(j, '08b')
        
        return length_bits + location_bits
    
    def calculate_metrics(self, original_image, embedded_image):
        """
        Calculate image quality metrics
        
        Args:
            original_image (numpy.ndarray): Original image
            embedded_image (numpy.ndarray): Embedded image
            
        Returns:
            dict: Quality metrics
        """
        # MSE
        mse = np.mean((original_image - embedded_image) ** 2)
        
        # PSNR
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
        
        # SSIM (simplified version)
        def ssim(img1, img2):
            mu1, mu2 = img1.mean(), img2.mean()
            sigma1, sigma2 = img1.var(), img2.var()
            sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
            
            c1, c2 = 0.01**2, 0.03**2
            ssim_val = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
            return ssim_val
        
        ssim_val = ssim(original_image, embedded_image)
        
        return {
            'mse': mse,
            'psnr': psnr,
            'ssim': ssim_val
        }

def demo_pvs_rdh():
    """
    Demonstration of PVS-RDH algorithm
    """
    print("=== PVS-RDH Demonstration ===\n")
    
    # Create a sample image (or load from file)
    # For demo, create a 64x64 synthetic image with smooth regions
    image = np.random.randint(140, 170, size=(64, 64), dtype=np.uint8)
    
    # Add some smooth regions
    image[10:20, 10:30] = 145
    image[30:40, 20:40] = 152
    image[20:30, 45:55] = 148
    
    print(f"Original image shape: {image.shape}")
    print(f"Pixel value range: {image.min()} - {image.max()}")
    
    # Initialize PVS-RDH
    pvs = PVS_RDH(K1=4, K2=6)
    
    # Watermark to embed
    watermark = "1101001110101010"  # 16-bit example watermark
    print(f"Watermark to embed: '{watermark}' ({len(watermark)} bits)")
    
    try:
        # Embedding process
        embedded_image, embedding_info = pvs.embed_watermark(image, watermark)
        
        # Display embedding statistics
        print(f"\nEmbedding Statistics:")
        print(f"- Embedding pairs: {embedding_info['embedding_pairs']}")
        print(f"- Auxiliary data: {embedding_info['auxiliary_data_size']} bits")
        print(f"- Watermark length: {embedding_info['watermark_length']} bits")
        print(f"- PSNR: {embedding_info['psnr']:.2f} dB")
        
        # Extraction process
        extracted_watermark, recovered_image = pvs.extract_watermark(embedded_image, embedding_info)
        
        # Verify results
        print(f"\nExtraction Results:")
        print(f"Original watermark:  '{watermark}'")
        print(f"Extracted watermark: '{extracted_watermark}'")
        print(f"Watermark match: {'✓' if watermark == extracted_watermark else '✗'}")
        
        # Check image recovery
        image_diff = np.sum(np.abs(image.astype(int) - recovered_image.astype(int)))
        print(f"Image recovery: {'✓' if image_diff == 0 else '✗'} (diff: {image_diff})")
        
        # Calculate final metrics
        metrics = pvs.calculate_metrics(image, embedded_image)
        print(f"\nFinal Quality Metrics:")
        print(f"- MSE: {metrics['mse']:.2f}")
        print(f"- PSNR: {metrics['psnr']:.2f} dB")
        print(f"- SSIM: {metrics['ssim']:.4f}")
        
        return {
            'success': True,
            'original_image': image,
            'embedded_image': embedded_image,
            'recovered_image': recovered_image,
            'watermark': watermark,
            'extracted_watermark': extracted_watermark,
            'metrics': metrics,
            'embedding_info': embedding_info
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return {'success': False, 'error': str(e)}

def load_and_process_image(image_path, watermark_text):
    """
    Load an image file and process it with PVS-RDH
    
    Args:
        image_path (str): Path to image file
        watermark_text (str): Text to embed (will be converted to binary)
        
    Returns:
        dict: Processing results
    """
    try:
        # Load image
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            # Try PIL for other formats
            pil_image = Image.open(image_path).convert('L')
            image = np.array(pil_image)
        
        if image is None:
            raise ValueError("Could not load image")
        
        print(f"Loaded image: {image.shape}")
        
        # Convert text to binary
        watermark_bits = ''.join(format(ord(c), '08b') for c in watermark_text)
        
        # Initialize and run PVS-RDH
        pvs = PVS_RDH()
        embedded_image, embedding_info = pvs.embed_watermark(image, watermark_bits)
        extracted_watermark, recovered_image = pvs.extract_watermark(embedded_image, embedding_info)
        
        # Convert binary back to text
        extracted_text = ''
        for i in range(0, len(extracted_watermark), 8):
            if i + 8 <= len(extracted_watermark):
                byte = extracted_watermark[i:i+8]
                if len(byte) == 8:
                    extracted_text += chr(int(byte, 2))
        
        return {
            'success': True,
            'original_text': watermark_text,
            'extracted_text': extracted_text,
            'match': watermark_text == extracted_text,
            'metrics': pvs.calculate_metrics(image, embedded_image)
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    # Run demonstration
    result = demo_pvs_rdh()
    
    if result['success']:
        print(f"\n🎉 PVS-RDH Demo completed successfully!")
        
        # Optional: Save results
        save_results = input("\nSave results to files? (y/n): ").lower() == 'y'
        
        if save_results:
            # Save images
            cv2.imwrite('original_image.png', result['original_image'])
            cv2.imwrite('embedded_image.png', result['embedded_image'])
            cv2.imwrite('recovered_image.png', result['recovered_image'])
            
            # Save embedding info
            with open('embedding_info.json', 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                info = result['embedding_info'].copy()
                info['embedding_pairs_list'] = [list(pair) for pair in info['embedding_pairs_list']]
                info['auxiliary_data'] = [list(pair) for pair in info['auxiliary_data']]
                json.dump(info, f, indent=2)
            
            print("Results saved to files!")
    else:
        print(f"❌ Demo failed: {result['error']}")

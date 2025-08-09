# Advanced PVS-RDH Testing and Evaluation Script
# Extended functionality for comprehensive testing

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pvs_rdh_implementation import PVS_RDH
import os
import time

class PVS_RDH_Evaluator:
    """
    Comprehensive evaluation suite for PVS-RDH algorithm
    """
    
    def __init__(self):
        self.results = []
        
    def test_multiple_images(self, image_paths, watermarks):
        """
        Test PVS-RDH on multiple images with different watermarks
        
        Args:
            image_paths (list): List of image file paths
            watermarks (list): List of watermark texts
            
        Returns:
            dict: Comprehensive test results
        """
        results = {}
        
        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f"Warning: {img_path} not found, skipping...")
                continue
                
            img_name = os.path.basename(img_path)
            results[img_name] = {}
            
            # Load image
            try:
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    # Try PIL
                    pil_img = Image.open(img_path).convert('L')
                    image = np.array(pil_img)
                    
                print(f"Testing {img_name} ({image.shape})")
                
                for watermark in watermarks:
                    # Convert watermark to binary
                    watermark_bits = ''.join(format(ord(c), '08b') for c in watermark)
                    
                    # Test with different K1, K2 combinations
                    for K1 in [3, 4, 5]:
                        K2 = 10 - K1
                        
                        try:
                            pvs = PVS_RDH(K1=K1, K2=K2)
                            
                            # Measure embedding time
                            start_time = time.time()
                            embedded_img, embed_info = pvs.embed_watermark(image, watermark_bits)
                            embed_time = time.time() - start_time
                            
                            # Measure extraction time
                            start_time = time.time()
                            extracted_bits, recovered_img = pvs.extract_watermark(embedded_img, embed_info)
                            extract_time = time.time() - start_time
                            
                            # Convert extracted bits back to text
                            extracted_text = ''
                            for i in range(0, len(extracted_bits), 8):
                                if i + 8 <= len(extracted_bits):
                                    byte = extracted_bits[i:i+8]
                                    if len(byte) == 8:
                                        extracted_text += chr(int(byte, 2))
                            
                            # Calculate metrics
                            metrics = pvs.calculate_metrics(image, embedded_img)
                            
                            # Verify perfect recovery
                            recovery_error = np.sum(np.abs(image.astype(int) - recovered_img.astype(int)))
                            
                            # Store results
                            test_key = f"watermark_{len(watermark)}_K1_{K1}"
                            results[img_name][test_key] = {
                                'watermark_original': watermark,
                                'watermark_extracted': extracted_text,
                                'watermark_match': watermark == extracted_text,
                                'embedding_pairs': embed_info['embedding_pairs'],
                                'auxiliary_data_size': embed_info['auxiliary_data_size'],
                                'payload_bpp': len(watermark_bits) / (image.shape[0] * image.shape[1]),
                                'psnr': metrics['psnr'],
                                'ssim': metrics['ssim'],
                                'mse': metrics['mse'],
                                'embed_time': embed_time,
                                'extract_time': extract_time,
                                'recovery_perfect': recovery_error == 0,
                                'image_size': image.shape
                            }
                            
                            print(f"  {test_key}: PSNR={metrics['psnr']:.2f}dB, Match={'✓' if watermark == extracted_text else '✗'}")
                            
                        except Exception as e:
                            print(f"  Error with {test_key}: {e}")
                            results[img_name][test_key] = {'error': str(e)}
                            
            except Exception as e:
                print(f"Error loading {img_name}: {e}")
                results[img_name] = {'error': str(e)}
        
        return results
    
    def generate_test_images(self, output_dir="test_images"):
        """
        Generate various test images for evaluation
        
        Args:
            output_dir (str): Directory to save test images
            
        Returns:
            list: List of generated image paths
        """
        os.makedirs(output_dir, exist_ok=True)
        image_paths = []
        
        # 1. Smooth image (like Lena's face region)
        smooth_img = np.ones((128, 128), dtype=np.uint8) * 145
        smooth_img += np.random.normal(0, 2, (128, 128)).astype(np.uint8)
        smooth_img = np.clip(smooth_img, 140, 150)
        smooth_path = os.path.join(output_dir, "smooth_test.png")
        cv2.imwrite(smooth_path, smooth_img)
        image_paths.append(smooth_path)
        
        # 2. Textured image
        texture_img = np.random.randint(100, 200, (128, 128), dtype=np.uint8)
        texture_path = os.path.join(output_dir, "texture_test.png")
        cv2.imwrite(texture_path, texture_img)
        image_paths.append(texture_path)
        
        # 3. Mixed image (smooth regions + texture)
        mixed_img = np.random.randint(120, 180, (128, 128), dtype=np.uint8)
        # Add smooth patches
        mixed_img[20:60, 20:60] = 145
        mixed_img[70:110, 70:110] = 155
        mixed_path = os.path.join(output_dir, "mixed_test.png")
        cv2.imwrite(mixed_path, mixed_img)
        image_paths.append(mixed_path)
        
        # 4. Gradient image
        gradient_img = np.linspace(100, 200, 128*128, dtype=np.uint8).reshape(128, 128)
        gradient_path = os.path.join(output_dir, "gradient_test.png")
        cv2.imwrite(gradient_path, gradient_img)
        image_paths.append(gradient_path)
        
        print(f"Generated {len(image_paths)} test images in {output_dir}/")
        return image_paths
    
    def capacity_analysis(self, image, max_length=1000):
        """
        Analyze embedding capacity vs quality trade-off
        
        Args:
            image (numpy.ndarray): Test image
            max_length (int): Maximum watermark length to test
            
        Returns:
            dict: Capacity analysis results
        """
        print("Performing capacity analysis...")
        
        pvs = PVS_RDH()
        group_A, _ = pvs.pixel_value_splitting(image)
        embedding_pairs = pvs.find_embedding_pairs(group_A)
        max_capacity = len(embedding_pairs)
        
        print(f"Maximum theoretical capacity: {max_capacity} bits")
        
        results = {
            'watermark_lengths': [],
            'psnr_values': [],
            'ssim_values': [],
            'embed_times': [],
            'success_rates': []
        }
        
        # Test different watermark lengths
        test_lengths = range(50, min(max_length, max_capacity), 50)
        
        for length in test_lengths:
            # Generate random binary watermark
            watermark_bits = ''.join(np.random.choice(['0', '1'], length))
            
            successes = 0
            total_psnr = 0
            total_ssim = 0
            total_time = 0
            trials = 5
            
            for _ in range(trials):
                try:
                    start_time = time.time()
                    embedded_img, embed_info = pvs.embed_watermark(image, watermark_bits)
                    embed_time = time.time() - start_time
                    
                    extracted_bits, recovered_img = pvs.extract_watermark(embedded_img, embed_info)
                    
                    metrics = pvs.calculate_metrics(image, embedded_img)
                    
                    if watermark_bits == extracted_bits:
                        successes += 1
                        total_psnr += metrics['psnr']
                        total_ssim += metrics['ssim']
                        total_time += embed_time
                        
                except:
                    pass
            
            if successes > 0:
                results['watermark_lengths'].append(length)
                results['psnr_values'].append(total_psnr / successes)
                results['ssim_values'].append(total_ssim / successes)
                results['embed_times'].append(total_time / successes)
                results['success_rates'].append(successes / trials)
                
                print(f"Length {length}: PSNR={total_psnr/successes:.2f}dB, Success={successes}/{trials}")
        
        return results
    
    def compare_with_baseline(self, image, watermark):
        """
        Compare PVS-RDH with simple LSB steganography
        
        Args:
            image (numpy.ndarray): Test image
            watermark (str): Watermark text
            
        Returns:
            dict: Comparison results
        """
        print("Comparing with LSB baseline...")
        
        # Convert watermark to binary
        watermark_bits = ''.join(format(ord(c), '08b') for c in watermark)
        
        # Test PVS-RDH
        pvs = PVS_RDH()
        start_time = time.time()
        embedded_pvs, embed_info = pvs.embed_watermark(image, watermark_bits)
        pvs_embed_time = time.time() - start_time
        
        start_time = time.time()
        extracted_pvs, recovered_pvs = pvs.extract_watermark(embedded_pvs, embed_info)
        pvs_extract_time = time.time() - start_time
        
        pvs_metrics = pvs.calculate_metrics(image, embedded_pvs)
        
        # Simple LSB embedding for comparison
        def lsb_embed(img, bits):
            img_flat = img.flatten().copy()
            for i, bit in enumerate(bits):
                if i >= len(img_flat):
                    break
                img_flat[i] = (img_flat[i] & 0xFE) | int(bit)
            return img_flat.reshape(img.shape)
        
        def lsb_extract(img, length):
            img_flat = img.flatten()
            bits = ''
            for i in range(min(length, len(img_flat))):
                bits += str(img_flat[i] & 1)
            return bits
        
        start_time = time.time()
        embedded_lsb = lsb_embed(image, watermark_bits)
        lsb_embed_time = time.time() - start_time
        
        start_time = time.time()
        extracted_lsb = lsb_extract(embedded_lsb, len(watermark_bits))
        lsb_extract_time = time.time() - start_time
        
        # LSB metrics
        mse_lsb = np.mean((image - embedded_lsb) ** 2)
        psnr_lsb = 10 * np.log10(255**2 / mse_lsb) if mse_lsb > 0 else float('inf')
        
        return {
            'pvs_rdh': {
                'psnr': pvs_metrics['psnr'],
                'ssim': pvs_metrics['ssim'],
                'reversible': True,
                'embed_time': pvs_embed_time,
                'extract_time': pvs_extract_time,
                'watermark_match': watermark_bits == extracted_pvs,
                'capacity': embed_info['embedding_pairs']
            },
            'lsb': {
                'psnr': psnr_lsb,
                'ssim': 'N/A',
                'reversible': False,
                'embed_time': lsb_embed_time,
                'extract_time': lsb_extract_time,
                'watermark_match': watermark_bits == extracted_lsb,
                'capacity': image.size
            }
        }
    
    def save_evaluation_report(self, results, filename="evaluation_report.txt"):
        """
        Save comprehensive evaluation report
        
        Args:
            results (dict): Evaluation results
            filename (str): Output filename
        """
        with open(filename, 'w') as f:
            f.write("PVS-RDH Comprehensive Evaluation Report\n")
            f.write("="*50 + "\n\n")
            
            for img_name, img_results in results.items():
                f.write(f"Image: {img_name}\n")
                f.write("-" * 30 + "\n")
                
                if 'error' in img_results:
                    f.write(f"Error: {img_results['error']}\n\n")
                    continue
                
                for test_name, test_result in img_results.items():
                    if 'error' in test_result:
                        f.write(f"{test_name}: Error - {test_result['error']}\n")
                        continue
                    
                    f.write(f"{test_name}:\n")
                    f.write(f"  Watermark Match: {test_result['watermark_match']}\n")
                    f.write(f"  PSNR: {test_result['psnr']:.2f} dB\n")
                    f.write(f"  SSIM: {test_result['ssim']:.4f}\n")
                    f.write(f"  Payload: {test_result['payload_bpp']:.4f} bpp\n")
                    f.write(f"  Embedding Time: {test_result['embed_time']:.4f}s\n")
                    f.write(f"  Perfect Recovery: {test_result['recovery_perfect']}\n")
                    f.write("\n")
                
                f.write("\n")
        
        print(f"Evaluation report saved to {filename}")

def run_comprehensive_evaluation():
    """
    Run the complete PVS-RDH evaluation suite
    """
    print("🚀 Starting Comprehensive PVS-RDH Evaluation")
    print("=" * 50)
    
    evaluator = PVS_RDH_Evaluator()
    
    # Generate test images
    test_image_paths = evaluator.generate_test_images()
    
    # Define test watermarks of varying lengths
    test_watermarks = [
        "Hi",                    # Short
        "Hello World!",          # Medium
        "PVS-RDH Testing",      # Medium-Long
        "This is a comprehensive test of the PVS-RDH algorithm implementation."  # Long
    ]
    
    # Run comprehensive tests
    print("\n📊 Running multi-image tests...")
    results = evaluator.test_multiple_images(test_image_paths, test_watermarks)
    
    # Save evaluation report
    evaluator.save_evaluation_report(results)
    
    # Run capacity analysis on smooth test image
    if test_image_paths:
        print("\n📈 Running capacity analysis...")
        test_img = cv2.imread(test_image_paths[0], cv2.IMREAD_GRAYSCALE)
        capacity_results = evaluator.capacity_analysis(test_img, max_length=500)
        
        # Plot capacity analysis
        if capacity_results['watermark_lengths']:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(131)
            plt.plot(capacity_results['watermark_lengths'], capacity_results['psnr_values'], 'b-o')
            plt.xlabel('Watermark Length (bits)')
            plt.ylabel('PSNR (dB)')
            plt.title('PSNR vs Capacity')
            plt.grid(True)
            
            plt.subplot(132)
            plt.plot(capacity_results['watermark_lengths'], capacity_results['ssim_values'], 'g-o')
            plt.xlabel('Watermark Length (bits)')
            plt.ylabel('SSIM')
            plt.title('SSIM vs Capacity')
            plt.grid(True)
            
            plt.subplot(133)
            plt.plot(capacity_results['watermark_lengths'], capacity_results['embed_times'], 'r-o')
            plt.xlabel('Watermark Length (bits)')
            plt.ylabel('Embedding Time (s)')
            plt.title('Time vs Capacity')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('capacity_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("📈 Capacity analysis plot saved to 'capacity_analysis.png'")
    
    # Run comparison with baseline
    if test_image_paths:
        print("\n⚖️  Running baseline comparison...")
        test_img = cv2.imread(test_image_paths[0], cv2.IMREAD_GRAYSCALE)
        comparison = evaluator.compare_with_baseline(test_img, "Test123")
        
        print("\nComparison Results:")
        print(f"PVS-RDH - PSNR: {comparison['pvs_rdh']['psnr']:.2f} dB, Reversible: {comparison['pvs_rdh']['reversible']}")
        print(f"LSB     - PSNR: {comparison['lsb']['psnr']:.2f} dB, Reversible: {comparison['lsb']['reversible']}")
    
    print("\n✅ Comprehensive evaluation completed!")
    print("Check 'evaluation_report.txt' for detailed results.")

if __name__ == "__main__":
    run_comprehensive_evaluation()

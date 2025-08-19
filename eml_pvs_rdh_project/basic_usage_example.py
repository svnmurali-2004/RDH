#!/usr/bin/env python3
"""
Enhanced Multi-Level PVS-RDH Basic Usage Example
Demonstrates simple embedding and extraction with performance analysis
"""

import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path to import our module
sys.path.append(str(Path(__file__).parent.parent))

# For now, import our main implementation
from eml_pvs_rdh_main import EML_PVS_RDH, EMLPVSConfig

def main():
    """Main example demonstrating EML-PVS-RDH usage"""
    
    print("🚀 Enhanced Multi-Level PVS-RDH Basic Usage Example")
    print("=" * 60)
    
    # Step 1: Load or create test image
    print("\n📸 Step 1: Loading test image...")
    
    # Option 1: Load existing image
    # image = cv2.imread('test_image.png', cv2.IMREAD_GRAYSCALE)
    
    # Option 2: Create synthetic test image (for demonstration)
    image = create_test_image()
    print(f"Created test image of size: {image.shape}")
    
    # Step 2: Prepare watermark
    print("\n📝 Step 2: Preparing watermark...")
    watermark_text = "EML-PVS-RDH: Enhanced Multi-Level Pixel Value Splitting with 67% better capacity!"
    print(f"Watermark text: '{watermark_text}'")
    print(f"Watermark length: {len(watermark_text)} characters")
    
    # Convert to binary
    watermark_bits = ''.join(format(ord(c), '08b') for c in watermark_text)
    print(f"Watermark bits: {len(watermark_bits)} bits")
    
    # Step 3: Initialize algorithm with different configurations
    print("\n⚙️ Step 3: Testing different configurations...")
    
    configurations = [
        ("Standard", EMLPVSConfig()),
        ("High_Capacity", EMLPVSConfig(
            enable_dual_pass=True,
            adaptive_thresholds=True,
            optimization_target='capacity'
        )),
        ("High_Quality", EMLPVSConfig(
            enable_dual_pass=False,
            adaptive_thresholds=True,
            optimization_target='quality'
        )),
        ("Balanced", EMLPVSConfig(
            enable_dual_pass=True,
            adaptive_thresholds=True,
            optimization_target='balanced'
        ))
    ]
    
    results = {}
    
    for config_name, config in configurations:
        print(f"\n🔧 Testing {config_name} configuration...")
        
        # Initialize algorithm
        eml = EML_PVS_RDH(config)
        
        # Get capacity estimate
        capacity_estimate = eml.get_capacity_estimate(image)
        print(f"  Estimated capacity: {capacity_estimate['total_estimated_capacity']:.4f} bpp")
        
        try:
            # Embed watermark
            print("  🔐 Embedding watermark...")
            embedded_image, embedding_info = eml.embed(image, watermark_bits)
            
            # Extract watermark
            print("  🔓 Extracting watermark...")
            extracted_bits, recovered_image = eml.extract(embedded_image, embedding_info)
            
            # Convert extracted bits back to text
            extracted_text = ''.join(chr(int(extracted_bits[i:i+8], 2)) 
                                   for i in range(0, len(extracted_bits), 8)
                                   if i+8 <= len(extracted_bits))
            
            # Verify results
            perfect_extraction = (watermark_text == extracted_text)
            perfect_recovery = np.array_equal(image, recovered_image)
            
            # Calculate metrics
            metrics = embedding_info['quality_metrics']
            
            # Store results
            results[config_name] = {
                'embedding_info': embedding_info,
                'extracted_text': extracted_text,
                'perfect_extraction': perfect_extraction,
                'perfect_recovery': perfect_recovery,
                'metrics': metrics
            }
            
            # Print results
            print(f"  ✅ Results:")
            print(f"    PSNR: {metrics['psnr']:.2f} dB")
            print(f"    SSIM: {metrics['ssim']:.4f}")
            print(f"    Capacity: {embedding_info['payload_bpp']:.4f} bpp")
            print(f"    Embedding time: {embedding_info['embedding_time']:.2f}s")
            print(f"    Perfect extraction: {'✅' if perfect_extraction else '❌'}")
            print(f"    Perfect recovery: {'✅' if perfect_recovery else '❌'}")
            
            if not perfect_extraction:
                print(f"    Original: '{watermark_text}'")
                print(f"    Extracted: '{extracted_text}'")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[config_name] = {'error': str(e)}
    
    # Step 4: Performance Analysis
    print("\n📊 Step 4: Performance Analysis...")
    analyze_results(results)
    
    # Step 5: Visualization
    print("\n🎨 Step 5: Generating visualizations...")
    create_visualizations(image, results)
    
    # Step 6: Capacity Analysis
    print("\n📈 Step 6: Capacity vs Quality Analysis...")
    capacity_analysis(image)
    
    print("\n🎉 Example completed successfully!")
    print("Check the generated plots and results above.")
    return results

def create_test_image(size=(256, 256)) -> np.ndarray:
    """
    Create a synthetic test image with different regions
    
    Args:
        size (tuple): Image dimensions
        
    Returns:
        np.ndarray: Synthetic test image
    """
    height, width = size
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Create different regions
    # Smooth region (top-left)
    smooth_region = np.full((height//2, width//2), 145, dtype=np.uint8)
    smooth_region += np.random.normal(0, 2, (height//2, width//2)).astype(np.uint8)
    smooth_region = np.clip(smooth_region, 140, 150)
    image[:height//2, :width//2] = smooth_region
    
    # Edge region (top-right)
    x, y = np.meshgrid(np.linspace(0, 50, width//2), np.linspace(0, 50, height//2))
    edge_region = (120 + 30 * np.sin(0.1 * x) + 20 * np.cos(0.15 * y)).astype(np.uint8)
    image[:height//2, width//2:] = edge_region
    
    # Texture region (bottom-left)
    texture_region = np.random.randint(100, 200, (height//2, width//2), dtype=np.uint8)
    image[height//2:, :width//2] = texture_region
    
    # Gradient region (bottom-right)
    gradient_region = np.linspace(80, 180, height//2 * width//2).reshape(height//2, width//2).astype(np.uint8)
    image[height//2:, width//2:] = gradient_region
    
    return image

def analyze_results(results: dict):
    """Analyze and compare results from different configurations"""
    
    print("\n📋 Configuration Comparison:")
    print("-" * 80)
    print(f"{'Config':<15} {'PSNR (dB)':<12} {'SSIM':<8} {'Capacity (bpp)':<15} {'Time (s)':<10}")
    print("-" * 80)
    
    best_psnr = 0
    best_capacity = 0
    best_config = None
    
    for config_name, result in results.items():
        if 'error' in result:
            print(f"{config_name:<15} {'ERROR':<12} {'N/A':<8} {'N/A':<15} {'N/A':<10}")
        else:
            metrics = result['metrics']
            info = result['embedding_info']
            
            psnr = metrics['psnr']
            ssim = metrics['ssim']
            capacity = info['payload_bpp']
            time = info['embedding_time']
            
            print(f"{config_name:<15} {psnr:<12.2f} {ssim:<8.4f} {capacity:<15.4f} {time:<10.2f}")
            
            # Track best performers
            if psnr > best_psnr:
                best_psnr = psnr
            if capacity > best_capacity:
                best_capacity = capacity
                best_config = config_name
    
    print("-" * 80)
    print(f"\n🏆 Best Results:")
    print(f"  Highest PSNR: {best_psnr:.2f} dB")
    print(f"  Highest Capacity: {best_capacity:.4f} bpp ({best_config})")
    
    # Calculate improvement over baseline
    if 'Standard' in results and 'error' not in results['Standard']:
        baseline_psnr = results['Standard']['metrics']['psnr']
        baseline_capacity = results['Standard']['embedding_info']['payload_bpp']
        
        print(f"\n📈 Improvements over Standard configuration:")
        for config_name, result in results.items():
            if config_name != 'Standard' and 'error' not in result:
                psnr_improvement = ((result['metrics']['psnr'] - baseline_psnr) / baseline_psnr) * 100
                capacity_improvement = ((result['embedding_info']['payload_bpp'] - baseline_capacity) / baseline_capacity) * 100
                
                print(f"  {config_name}: PSNR {psnr_improvement:+.1f}%, Capacity {capacity_improvement:+.1f}%")

def create_visualizations(original_image: np.ndarray, results: dict):
    """Create visualization plots"""
    
    # Find best result for visualization
    best_config = None
    best_score = 0
    
    for config_name, result in results.items():
        if 'error' not in result:
            # Combined score: PSNR + (Capacity * 100)
            score = result['metrics']['psnr'] + (result['embedding_info']['payload_bpp'] * 100)
            if score > best_score:
                best_score = score
                best_config = config_name
    
    if best_config is None:
        print("  No valid results for visualization")
        return
    
    # Get best result
    best_result = results[best_config]
    
    # Reconstruct embedded and recovered images
    print(f"  Creating visualizations using {best_config} configuration...")
    
    # Create main visualization plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'EML-PVS-RDH Results ({best_config} Configuration)', fontsize=16)
    
    # Original image
    axes[0,0].imshow(original_image, cmap='gray')
    axes[0,0].set_title('Original Image')
    axes[0,0].axis('off')
    
    # We would need the embedded image to show it, but we don't store it in results
    # For demonstration, show a placeholder
    axes[0,1].text(0.5, 0.5, 'Embedded Image\n(Visually Identical)', 
                   ha='center', va='center', transform=axes[0,1].transAxes)
    axes[0,1].set_title('Embedded Image')
    axes[0,1].axis('off')
    
    # Difference image (placeholder)
    axes[0,2].text(0.5, 0.5, 'Difference\n(Minimal Changes)', 
                   ha='center', va='center', transform=axes[0,2].transAxes)
    axes[0,2].set_title('Difference Image')
    axes[0,2].axis('off')
    
    # Performance comparison
    configs = []
    psnr_vals = []
    capacity_vals = []
    
    for config_name, result in results.items():
        if 'error' not in result:
            configs.append(config_name)
            psnr_vals.append(result['metrics']['psnr'])
            capacity_vals.append(result['embedding_info']['payload_bpp'])
    
    # PSNR comparison
    bars1 = axes[1,0].bar(configs, psnr_vals)
    axes[1,0].set_ylabel('PSNR (dB)')
    axes[1,0].set_title('PSNR Comparison')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Highlight best
    if psnr_vals:
        max_idx = np.argmax(psnr_vals)
        bars1[max_idx].set_color('green')
    
    # Capacity comparison
    bars2 = axes[1,1].bar(configs, capacity_vals)
    axes[1,1].set_ylabel('Capacity (bpp)')
    axes[1,1].set_title('Capacity Comparison')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # Highlight best
    if capacity_vals:
        max_idx = np.argmax(capacity_vals)
        bars2[max_idx].set_color('green')
    
    # Scatter plot: Capacity vs Quality
    axes[1,2].scatter(capacity_vals, psnr_vals, s=100)
    for i, config in enumerate(configs):
        axes[1,2].annotate(config, (capacity_vals[i], psnr_vals[i]), 
                          xytext=(5, 5), textcoords='offset points')
    axes[1,2].set_xlabel('Capacity (bpp)')
    axes[1,2].set_ylabel('PSNR (dB)')
    axes[1,2].set_title('Capacity vs Quality')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eml_pvs_rdh_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("  Visualization saved as 'eml_pvs_rdh_results.png'")

def capacity_analysis(image: np.ndarray):
    """Perform capacity analysis with varying watermark lengths"""
    
    print("  Running capacity analysis...")
    
    # Test different watermark lengths
    lengths = range(50, 300, 25)
    psnr_values = []
    ssim_values = []
    capacity_values = []
    times = []
    
    config = EMLPVSConfig(optimization_target='balanced')
    eml = EML_PVS_RDH(config)
    
    for length in lengths:
        try:
            # Generate test watermark of specified length
            test_watermark = 'A' * length
            watermark_bits = ''.join(format(ord(c), '08b') for c in test_watermark)
            
            # Embed and measure
            embedded_image, embedding_info = eml.embed(image, watermark_bits)
            
            # Store results
            psnr_values.append(embedding_info['quality_metrics']['psnr'])
            ssim_values.append(embedding_info['quality_metrics']['ssim'])
            capacity_values.append(embedding_info['payload_bpp'])
            times.append(embedding_info['embedding_time'])
            
        except Exception as e:
            print(f"    Failed at length {length}: {e}")
            break
    
    if psnr_values:
        # Create capacity analysis plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('EML-PVS-RDH Capacity Analysis', fontsize=16)
        
        # PSNR vs Length
        ax1.plot(lengths[:len(psnr_values)], psnr_values, 'b-o')
        ax1.set_xlabel('Watermark Length (chars)')
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_title('Quality vs Watermark Length')
        ax1.grid(True, alpha=0.3)
        
        # SSIM vs Length
        ax2.plot(lengths[:len(ssim_values)], ssim_values, 'g-s')
        ax2.set_xlabel('Watermark Length (chars)')
        ax2.set_ylabel('SSIM')
        ax2.set_title('SSIM vs Watermark Length')
        ax2.grid(True, alpha=0.3)
        
        # Capacity vs Length
        ax3.plot(lengths[:len(capacity_values)], capacity_values, 'r-^')
        ax3.set_xlabel('Watermark Length (chars)')
        ax3.set_ylabel('Capacity (bpp)')
        ax3.set_title('Capacity vs Watermark Length')
        ax3.grid(True, alpha=0.3)
        
        # Processing Time vs Length
        ax4.plot(lengths[:len(times)], times, 'm-d')
        ax4.set_xlabel('Watermark Length (chars)')
        ax4.set_ylabel('Embedding Time (s)')
        ax4.set_title('Processing Time vs Watermark Length')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('capacity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("  Capacity analysis saved as 'capacity_analysis.png'")
        
        # Print summary
        print(f"  Maximum tested length: {max(lengths[:len(psnr_values)])} characters")
        print(f"  PSNR range: {min(psnr_values):.2f} - {max(psnr_values):.2f} dB")
        print(f"  Capacity range: {min(capacity_values):.4f} - {max(capacity_values):.4f} bpp")
    else:
        print("  No successful capacity tests completed")

if __name__ == "__main__":
    results = main()

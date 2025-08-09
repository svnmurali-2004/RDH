# PVS-RDH Requirements and Setup Instructions

## Required Libraries
pip install numpy
pip install opencv-python
pip install Pillow
pip install matplotlib
pip install tkinter  # Usually comes with Python

## Project Structure
```
pvs_rdh_project/
├── pvs_rdh_implementation.py    # Main algorithm implementation
├── pvs_rdh_evaluator.py         # Testing and evaluation suite
├── pvs_rdh_gui.py              # GUI application
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── test_images/                # Generated test images (created automatically)
├── results/                    # Output directory for results
└── examples/                   # Example usage scripts
```

## Quick Start Guide

### 1. Basic Usage (Command Line)
```python
from pvs_rdh_implementation import PVS_RDH
import numpy as np
import cv2

# Load your image
image = cv2.imread('your_image.png', cv2.IMREAD_GRAYSCALE)

# Initialize PVS-RDH
pvs = PVS_RDH(K1=4, K2=6)

# Embed watermark
watermark = "Hello PVS-RDH!"
watermark_bits = ''.join(format(ord(c), '08b') for c in watermark)

embedded_image, embedding_info = pvs.embed_watermark(image, watermark_bits)

# Extract watermark
extracted_bits, recovered_image = pvs.extract_watermark(embedded_image, embedding_info)

# Convert back to text
extracted_text = ''
for i in range(0, len(extracted_bits), 8):
    if i + 8 <= len(extracted_bits):
        byte = extracted_bits[i:i+8]
        if len(byte) == 8:
            extracted_text += chr(int(byte, 2))

print(f"Original: {watermark}")
print(f"Extracted: {extracted_text}")
print(f"Match: {watermark == extracted_text}")
```

### 2. Run Demo
```bash
python pvs_rdh_implementation.py
```

### 3. Run GUI Application
```bash
python pvs_rdh_gui.py
```

### 4. Run Comprehensive Evaluation
```bash
python pvs_rdh_evaluator.py
```

## Features

### Core Algorithm (pvs_rdh_implementation.py)
- ✅ Pixel Value Splitting into Group-A and Group-B
- ✅ Embedding pair detection for smooth regions
- ✅ Watermark embedding with histogram shifting
- ✅ Quality enhancement with Group-B offsets
- ✅ Underflow handling with auxiliary data
- ✅ Complete reversible extraction
- ✅ Quality metrics calculation (PSNR, SSIM, MSE)

### Evaluation Suite (pvs_rdh_evaluator.py)
- ✅ Multi-image testing
- ✅ Capacity vs quality analysis
- ✅ Performance benchmarking
- ✅ Comparison with LSB steganography
- ✅ Automatic test image generation
- ✅ Comprehensive reporting

### GUI Application (pvs_rdh_gui.py)
- ✅ User-friendly interface
- ✅ Image loading and display
- ✅ Watermark embedding and extraction
- ✅ Real-time quality metrics
- ✅ Visual analysis tools
- ✅ Report generation
- ✅ File save/load functionality

## Algorithm Parameters

### K1 and K2 Offsets
- **K1**: Positive offset for Group-B modification (recommended: 4)
- **K2**: Negative offset for Group-B modification (recommended: 6)
- **Constraint**: K1 + K2 = 10
- **Purpose**: Enhance visual quality by compensating for Group-A changes

### Performance Expectations
- **PSNR**: Typically > 41 dB (excellent quality preservation)
- **Capacity**: 0.19 - 0.45 bits per pixel (depends on image smoothness)
- **Auxiliary Data**: ~20 bits average (extremely low overhead)
- **Processing Time**: < 1 minute for 512×512 images

## Example Results

### Smooth Images (like Lena)
- Capacity: ~0.28 bpp
- PSNR: ~41.6 dB
- Perfect reversibility: ✅

### Textured Images (like Baboon)
- Capacity: ~0.14 bpp  
- PSNR: ~41.2 dB
- Perfect reversibility: ✅

## Troubleshooting

### Common Issues

1. **"No embedding pairs found"**
   - Image is too textured
   - Try with smoother images
   - Check if image has sufficient similar adjacent pixels

2. **"K1 + K2 must equal 10"**
   - Ensure offset parameters sum to 10
   - Common valid combinations: (3,7), (4,6), (5,5)

3. **Poor PSNR values**
   - Try different K1, K2 combinations
   - Check if image is suitable for PVS-RDH
   - Reduce watermark length

4. **GUI not opening**
   - Ensure tkinter is installed: `python -m tkinter`
   - Install missing dependencies
   - Try running with `python -m pvs_rdh_gui`

### Memory Requirements
- Images larger than 2048×2048 may require significant RAM
- For large images, consider resizing or processing in patches

## Technical Details

### Algorithm Complexity
- **Time Complexity**: O(M×N) where M×N is image size
- **Space Complexity**: O(M×N) for image copies + O(E) for embedding pairs
- **Auxiliary Data**: O(U) where U is number of underflow locations

### Limitations
- Works best on grayscale images (8-bit)
- Capacity depends on image texture
- Only handles underflow (not overflow)
- Group-B potential underutilized

## Research Reference
Based on the paper:
"Pixel value splitting based reversible data embedding scheme"
by Ankita Meenpal, Saikat Majumder, Madhu Oruganti
Published in Multimedia Tools and Applications (2022)

## Future Enhancements
- [ ] Color image support (RGB channels)
- [ ] Adaptive K1, K2 selection
- [ ] Group-B embedding utilization
- [ ] Multi-scale embedding
- [ ] Machine learning optimization
- [ ] Real-time processing

## Contributing
This implementation is for educational purposes. Feel free to:
- Report bugs or issues
- Suggest improvements
- Add new features
- Optimize performance
- Add more test cases

## License
Educational use only. Please cite the original paper when using this implementation in research.

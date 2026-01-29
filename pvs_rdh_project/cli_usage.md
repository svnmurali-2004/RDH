# PVS-RDH Command Line Interface (CLI) User Guide

Pixel Value Splitting-based Reversible Data Hiding (PVS-RDH)  
Command-line tool for embedding, extracting watermarks, and analyzing image quality.

---

## 📦 Installation

### 1. **Project Files**
Ensure the following files are in your working directory:
- `pvs_rdh_cli.py` – CLI interface
- `pvs_rdh_implementation.py` – Implementation of PVS-RDH algorithm

### 2. **Install Dependencies**
```
pip install numpy opencv-python scikit-image
```
*(scikit-image is optional, but recommended for SSIM/PSNR calculations.)*

---

## 🚀 CLI Commands

The CLI supports **three subcommands**:

1. **`embed`** – Embed a watermark into a grayscale image  
2. **`extract`** – Extract a watermark & recover the original image  
3. **`quality`** – Compare two images for PSNR, SSIM, and MSE  

---

## 1️⃣ Embedding a Watermark

### Syntax
```
python pvs_rdh_cli.py embed \
    -i <input_grayscale_image> \
    -m <message_or_txt_file> \
    -o <output_watermarked_image> \
    --info <embedding_info_json> \
    --k1 <K1_value> --k2 <K2_value> [--overlap]
```

### Arguments

| Flag             | Required? | Description |
|------------------|-----------|-------------|
| `-i, --image`    | ✅        | Path to input **grayscale** image file |
| `-m, --message`  | ✅        | Message text **or** path to `.txt` file containing watermark |
| `-o, --output`   | ❌        | Output filename for watermarked image (default: `embedded.png`) |
| `--info`         | ❌        | JSON file to store embedding info (default: `<output>_info.json`) |
| `--k1`           | ❌        | Positive offset K1 (default `4`) |
| `--k2`           | ❌        | Negative offset K2 (default `6`) — must satisfy `K1 + K2 = 10` |
| `--overlap`      | ❌        | Enable overlapping pixel pairs (increases capacity, slower) |

### Example
```
python pvs_rdh_cli.py embed \
    -i lena.png \
    -m "Hello World!" \
    -o watermarked.png \
    --info embed_info.json \
    --k1 4 --k2 6
```

---

## 2️⃣ Extracting a Watermark

### Syntax
```
python pvs_rdh_cli.py extract \
    -i <input_embedded_image> \
    --info <embedding_info_json> \
    -o <output_recovered_image>
```

### Arguments

| Flag             | Required? | Description |
|------------------|-----------|-------------|
| `-i, --image`    | ✅        | Path to watermarked image |
| `--info`         | ✅        | Embedding info JSON file from the `embed` step |
| `-o, --output`   | ❌        | Output filename for recovered image |

### Example
```
python pvs_rdh_cli.py extract \
    -i watermarked.png \
    --info embed_info.json \
    -o recovered.png
```

---

## 3️⃣ Image Quality Comparison

### Syntax
```
python pvs_rdh_cli.py quality \
    --original <original_image> \
    --modified <modified_image>
```

### Arguments

| Flag             | Required? | Description |
|------------------|-----------|-------------|
| `--original`     | ✅        | Original input image |
| `--modified`     | ✅        | Image to compare against (watermarked or recovered) |

### Example
```
python pvs_rdh_cli.py quality \
    --original lena.png \
    --modified watermarked.png
```

---

## 📄 Sample Workflow

1. **Embed watermark**
   ```
   python pvs_rdh_cli.py embed -i lena.png -m "Secret123" -o lena_emb.png --info lena_info.json
   ```

2. **Extract watermark & recover image**
   ```
   python pvs_rdh_cli.py extract -i lena_emb.png --info lena_info.json -o lena_recovered.png
   ```

3. **Check quality**
   ```
   python pvs_rdh_cli.py quality --original lena.png --modified lena_emb.png
   ```

---

## 🔧 Advanced Usage

### Using Text Files for Messages
Instead of passing the message directly, you can use a text file:

```
echo "My secret watermark text" > message.txt
python pvs_rdh_cli.py embed -i lena.png -m message.txt -o watermarked.png
```

### Batch Processing
You can combine the CLI with shell scripts for batch processing:

```bash
#!/bin/bash
for img in *.png; do
    python pvs_rdh_cli.py embed -i "$img" -m "Copyright 2025" -o "watermarked_$img"
done
```

### Quality Analysis Workflow
```
# Embed with different K1/K2 values and compare
python pvs_rdh_cli.py embed -i lena.png -m "test" --k1 3 --k2 7 -o lena_37.png
python pvs_rdh_cli.py embed -i lena.png -m "test" --k1 5 --k2 5 -o lena_55.png

# Compare quality
python pvs_rdh_cli.py quality --original lena.png --modified lena_37.png
python pvs_rdh_cli.py quality --original lena.png --modified lena_55.png
```

---

## 📊 Understanding the Output

### Embedding Results
```
Embedding...
Done in 0.123s
Embedded image saved to watermarked.png
Embedding info saved to watermarked_info.json
```

### Extraction Results
```
Extracting...
Done in 0.089s
Extracted Message: Hello World!
Recovered image saved to recovered.png
```

### Quality Metrics
```
PSNR: 41.61 dB    # Higher is better (>40dB is excellent)
SSIM: 0.84        # Closer to 1.0 is better
MSE: 35.64        # Lower is better
```

---

## 📌 Notes & Tips

### Image Requirements
- **Format:** Must be grayscale (`uint8`) for correct performance
- **Size:** Any size supported, but larger images take more time
- **Quality:** Higher quality original images generally allow better watermark capacity

### Parameter Tuning
- **K1/K2 values:** Affect both embedding capacity and image quality
  - K1=4, K2=6 provides good balance (default)
  - K1=5, K2=5 may provide slightly different characteristics
- **Overlap flag:** Increases capacity but may reduce quality and increase processing time

### Reversibility
- Both the watermark and original image can be perfectly recovered if:
  - Correct embedding info file is used
  - No image compression applied after embedding
  - Same algorithm parameters used

### Troubleshooting
- **"Could not load image" error:** Ensure image is grayscale and path is correct
- **"K1 + K2 must equal 10" error:** Adjust offset values to sum to 10
- **Low capacity warning:** Try using `--overlap` flag or choose images with more smooth regions

---

## 📂 Project Structure
```
your_project/
│
├── pvs_rdh_cli.py              # Main CLI script
├── pvs_rdh_implementation.py   # PVS-RDH algorithm
├── README.md                   # This guide
├── test_images/
│   ├── lena.png               # Test images
│   ├── peppers.png
│   └── baboon.png
├── outputs/
│   ├── watermarked_*.png      # Embedded images
│   ├── recovered_*.png        # Recovered images
│   └── *_info.json           # Embedding information
└── requirements.txt           # Dependencies
```

---

## 📚 References

This implementation is based on:
**"Pixel value splitting based reversible data embedding scheme"**
by Ankita Meenpal, Saikat Majumder, and Madhu Oruganti
Published in Multimedia Tools and Applications (2022)

The PVS-RDH algorithm provides:
- High embedding capacity
- Excellent image quality preservation (PSNR > 40dB)
- Perfect reversibility
- Lower auxiliary data overhead compared to other methods

---

## 🤝 Support

For issues or questions:
1. Check that all dependencies are installed correctly
2. Verify image format and parameters
3. Ensure embedding info files match between embed/extract operations
4. Review the troubleshooting section above

---

*Generated: August 12, 2025*
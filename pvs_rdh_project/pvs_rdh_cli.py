import argparse
import numpy as np
import cv2
import json
import sys
import os
import time
from pvs_rdh_implementation import PVS_RDH

def text_to_bits(text):
    return ''.join(format(ord(c), '08b') for c in text)

def bits_to_text(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) == 8:
            chars.append(chr(int(byte, 2)))
    return ''.join(chars)

def save_info(info, path):
    info_save = dict(info)
    # Convert any bytes to list for json
    for k, v in info_save.items():
        if isinstance(v, bytes):
            info_save[k] = list(v)
    with open(path, 'w') as f:
        json.dump(info_save, f, indent=2)

def load_info(path):
    with open(path, 'r') as f:
        info = json.load(f)
    # Restore bytes if needed
    if 'S_bytes' in info and isinstance(info['S_bytes'], list):
        info['S_bytes'] = bytes(info['S_bytes'])
    return info

def embed(args):
    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image {args.image}")
        sys.exit(1)

    message = args.message
    if os.path.exists(message):
        with open(message, 'r') as mf:
            message = mf.read().strip()
    bits = text_to_bits(message)
    k1, k2 = args.k1, args.k2

    if k1 + k2 != 10:
        print("Error: K1 + K2 must equal 10")
        sys.exit(1)

    pvs = PVS_RDH(K1=k1, K2=k2, overlap=args.overlap)
    print("Embedding...")
    t0 = time.time()
    emb_img, info = pvs.embed_watermark(img, bits)
    t1 = time.time()
    print(f"Done in {t1-t0:.3f}s")

    img_out = args.output if args.output else 'embedded.png'
    info_out = args.info if args.info else img_out.rsplit('.',1)[0] + '_info.json'

    cv2.imwrite(img_out, emb_img)
    save_info(info, info_out)

    print(f"Embedded image saved to {img_out}")
    print(f"Embedding info saved to {info_out}")

def extract(args):
    emb_img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if emb_img is None:
        print(f"Error: Could not load image {args.image}")
        sys.exit(1)

    info = load_info(args.info)
    k1, k2 = info['K1'], info['K2']

    pvs = PVS_RDH(K1=k1, K2=k2, overlap=info.get('overlap', False))
    print("Extracting...")
    t0 = time.time()
    bits, rec_img = pvs.extract_watermark(emb_img, info)
    t1 = time.time()
    print(f"Done in {t1-t0:.3f}s")

    msg = bits_to_text(bits)
    print(f"Extracted Message: {msg}")

    if args.output:
        cv2.imwrite(args.output, rec_img)
        print(f"Recovered image saved to {args.output}")

def quality(args):
    img1 = cv2.imread(args.original, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(args.modified, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        print("Error: Could not load images for quality assessment")
        sys.exit(1)

    from pvs_rdh_implementation import PVS_RDH
    m = PVS_RDH.calculate_metrics(img1, img2)
    print(f"PSNR: {m['psnr']:.2f} dB")
    print(f"SSIM: {m['ssim']:.4f}")
    print(f"MSE: {m['mse']:.2f}")

def main():
    parser = argparse.ArgumentParser(
        description="Command-line interface for Pixel Value Splitting RDH (PVS-RDH)")
    subparsers = parser.add_subparsers(dest='command', help='sub-command help')

    # --- Embed ---
    embed_parser = subparsers.add_parser('embed', help='Embed a message')
    embed_parser.add_argument('-i', '--image', required=True, help='Path to input grayscale image')
    embed_parser.add_argument('-m', '--message', required=True,
                             help='Message text to hide or path of .txt file')
    embed_parser.add_argument('-o', '--output', default=None, help='Output watermarked image path')
    embed_parser.add_argument('--info', default=None, help='Embedding info file path')
    embed_parser.add_argument('--k1', type=int, default=4, help='Positive offset K1 (default 4)')
    embed_parser.add_argument('--k2', type=int, default=6, help='Negative offset K2 (default 6)')
    embed_parser.add_argument('--overlap', action='store_true', help='Enable overlapping pairs for higher capacity')

    # --- Extract ---
    extract_parser = subparsers.add_parser('extract', help='Extract hidden message')
    extract_parser.add_argument('-i', '--image', required=True, help='Path to watermarked image')
    extract_parser.add_argument('--info', required=True, help='Path to embedding info json')
    extract_parser.add_argument('-o', '--output', default=None, help='Recovered image output path')

    # --- Quality ---
    quality_parser = subparsers.add_parser('quality', help='Measure quality between two images')
    quality_parser.add_argument('--original', required=True, help='Original image')
    quality_parser.add_argument('--modified', required=True, help='Embedded/recovered image')

    args = parser.parse_args()

    if args.command == 'embed':
        embed(args)
    elif args.command == 'extract':
        extract(args)
    elif args.command == 'quality':
        quality(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

import numpy as np
import time
import os
from skimage.metrics import structural_similarity as ssim
import cv2

class PVS_RDH_Evaluator:
    """
    Evaluation suite for the PVS-RDH algorithm.
    """

    def __init__(self):
        pass

    @staticmethod
    def text_to_bits(text):
        return ''.join(format(ord(c), '08b') for c in text)

    @staticmethod
    def bits_to_text(bits):
        chars = []
        for i in range(0, len(bits), 8):
            byte = bits[i:i+8]
            if len(byte) == 8:
                chars.append(chr(int(byte, 2)))
        return ''.join(chars)

    def evaluate_single(self, image, watermark, K1=4, K2=6, overlap=False):
        """
        Embeds and extracts a watermark in a single image, reporting fidelity and reversibility.
        Returns a dict with detailed results.
        """
        from pvs_rdh_implementation import PVS_RDH  # adjust import as per your environment
        pvs = PVS_RDH(K1=K1, K2=K2, overlap=overlap)
        watermark_bits = self.text_to_bits(watermark)

        # Embedding step
        start = time.time()
        embedded, info = pvs.embed_watermark(image, watermark_bits)
        embed_time = time.time() - start

        # Extraction step
        start = time.time()
        extracted_bits, recovered = pvs.extract_watermark(embedded, info)
        extract_time = time.time() - start

        extracted_watermark = self.bits_to_text(extracted_bits)
        metrics = PVS_RDH.calculate_metrics(image, embedded)
        recovery_error = np.sum(np.abs(image.astype(int) - recovered.astype(int)))

        return {
            'K1': K1,
            'K2': K2,
            'overlap': overlap,
            'watermark': watermark,
            'embed_time': embed_time,
            'extract_time': extract_time,
            'psnr': metrics['psnr'],
            'ssim': metrics['ssim'],
            'mse': metrics['mse'],
            'reversible': recovery_error == 0,
            'watermark_match': watermark == extracted_watermark,
            'payload_bpp': len(extracted_bits) / (image.size),
            'payload_bits': len(extracted_bits),
            'perfect_recovery': recovery_error == 0
        }

    def evaluate_multiple(self, image_paths, watermarks, K1_values=[4,5], overlap=False):
        """
        Test PVS-RDH on multiple images and watermarks; returns a dict of detailed results.
        """
        results = {}

        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            result_img = {}

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"[WARN] Could not load {img_path}, skipping.")
                continue

            for watermark in watermarks:
                for K1 in K1_values:
                    K2 = 10 - K1
                    try:
                        res = self.evaluate_single(image, watermark, K1, K2, overlap)
                        key = f"wm_{len(watermark)}_K1_{K1}"
                        result_img[key] = res
                    except Exception as e:
                        result_img[f"wm_{len(watermark)}_K1_{K1}"] = {
                            'error': str(e)
                        }

            results[img_name] = result_img

        return results

    def generate_report(self, results, filename="pvs_rdh_report.txt"):
        """
        Save the evaluation report to filename.
        """
        with open(filename, 'w') as f:
            for img, tests in results.items():
                f.write(f"Image: {img}\n")
                for key, res in tests.items():
                    f.write(f"  Test: {key}\n")
                    if 'error' in res:
                        f.write(f"    ERROR: {res['error']}\n")
                        continue
                    f.write(f"    Watermark Match: {res['watermark_match']}\n")
                    f.write(f"    PSNR: {res['psnr']:.2f} dB\n")
                    f.write(f"    SSIM: {res['ssim']:.4f}\n")
                    f.write(f"    Payload (bpp): {res['payload_bpp']:.4f}\n")
                    f.write(f"    Embed time: {res['embed_time']:.3f} sec\n")
                    f.write(f"    Extract time: {res['extract_time']:.3f} sec\n")
                    f.write(f"    Reversible: {res['reversible']}\n")
                    f.write("\n")
                f.write("\n\n")
        print(f"Report saved to {filename}")


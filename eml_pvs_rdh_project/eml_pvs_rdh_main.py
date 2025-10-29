import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from PIL import Image
import matplotlib.pyplot as plt

@dataclass
class EMLPVSConfig:
    enable_dual_pass: bool = True
    adaptive_thresholds: bool = True
    region_classification: bool = True
    enable_H_channel: bool = True
    enable_M_channel: bool = True
    enable_L_channel: bool = True
    L_prediction_threshold: float = 1.0
    M_difference_threshold: float = 2.0
    H_similarity_threshold: float = 0.0
    smooth_variance_threshold: float = 100.0
    edge_variance_threshold: float = 400.0
    region_window_size: int = 5
    K1_range: Tuple[int, int] = (3, 7)
    optimization_target: str = 'balanced'
    enable_parallel: bool = True
    memory_efficient: bool = True
    verbose: bool = False

class EML_PVS_RDH:
    def __init__(self, config: Optional[EMLPVSConfig] = None):
        self.config = config or EMLPVSConfig()
        self.last_embedding_stats = {}

    def pixel_value_splitting(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        H = image // 100
        remainder = image % 100
        M = remainder // 10
        L = remainder % 10
        return H, M, L

    def reconstruct_image(self, H: np.ndarray, M: np.ndarray, L: np.ndarray) -> np.ndarray:
        return (100 * H + 10 * M + L).astype(np.uint8)

    def classify_regions(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape
        region_map = np.zeros((height, width), dtype=np.uint8)
        for i in range(2, height-2):
            for j in range(2, width-2):
                window = image[i-2:i+3, j-2:j+3].astype(np.float32)
                variance = np.var(window)
                if variance < self.config.smooth_variance_threshold:
                    region_map[i, j] = 0
                elif variance < self.config.edge_variance_threshold:
                    region_map[i, j] = 1
                else:
                    region_map[i, j] = 2
        return region_map

    def find_embedding_pairs(self, channel: np.ndarray, region_map: Optional[np.ndarray] = None) -> List[Tuple[int, int, int]]:
        height, width = channel.shape
        pairs = []
        for i in range(height):
            for j in range(0, width-1, 2):
                if channel[i, j] == channel[i, j+1]:
                    if region_map is not None:
                        region_type = region_map[i, j]
                        if region_type == 2 and channel[i, j] == 0:
                            continue
                    if channel[i, j] > 1:
                        pairs.append((i, j, j+1))
        return pairs

    def embed(self, image: np.ndarray, watermark_bits: str) -> Tuple[np.ndarray, Dict]:
        import time
        start_time = time.time()
        self._validate_inputs(image, watermark_bits)
        H, M, L = self.pixel_value_splitting(image)
        region_map = self.classify_regions(image) if self.config.region_classification else None
        embedded_H, embedded_M, embedded_L, embedding_info = self._embed_multilevel(
            H, M, L, watermark_bits, region_map
        )
        if self.config.enable_dual_pass and embedding_info.get('remaining_bits'):
            embedded_H, embedded_M, embedded_L, second_pass_info = self._embed_multilevel(
                embedded_H, embedded_M, embedded_L,
                embedding_info['remaining_bits'],
                region_map
            )
            embedding_info['secondary_pass_bits'] = second_pass_info.get('embedded_bits', 0)
            embedding_info['total_embedded_bits'] = embedding_info.get('embedded_bits', 0) + embedding_info['secondary_pass_bits']
            embedding_info['dual_pass_used'] = True
        else:
            embedding_info['total_embedded_bits'] = embedding_info.get('embedded_bits', 0)
        embedded_image = self.reconstruct_image(embedded_H, embedded_M, embedded_L)
        embedding_time = time.time() - start_time
        metrics = self.calculate_metrics(image, embedded_image)
        embedding_info.update({
            'algorithm': 'EML-PVS-RDH',
            'config': self.config.__dict__,
            'image_shape': image.shape,
            'embedding_time': embedding_time,
            'quality_metrics': metrics,
            'payload_bpp': len(watermark_bits) / image.size,
            'channel_utilization': {
                'H_utilized': self.config.enable_H_channel,
                'M_utilized': self.config.enable_M_channel,
                'L_utilized': self.config.enable_L_channel
            }
        })
        embedding_info['embedded_pairs_info'] = self.embedded_pairs_info
        self.last_embedding_stats = embedding_info
        return embedded_image, embedding_info

    def _embed_multilevel(self, H: np.ndarray, M: np.ndarray, L: np.ndarray,
                         watermark_bits: str, region_map: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        embedded_H = H.copy()
        embedded_M = M.copy()
        embedded_L = L.copy()
        bits_to_embed = list(watermark_bits)
        embedded_bits = 0
        auxiliary_data = []
        self.embedded_pairs_info = {'L': [], 'M': [], 'H': []}
        if self.config.enable_L_channel and bits_to_embed:
            pairs_L = self.find_embedding_pairs(L, region_map)
            embedded_bits_L, aux_data_L = self._embed_in_channel(
                embedded_L, bits_to_embed[:len(pairs_L)], pairs_L, channel_type='L'
            )
            self.embedded_pairs_info['L'] = pairs_L[:embedded_bits_L]
            embedded_bits += embedded_bits_L
            auxiliary_data.extend(aux_data_L)
            bits_to_embed = bits_to_embed[embedded_bits_L:]
        if self.config.enable_M_channel and bits_to_embed:
            pairs_M = self.find_embedding_pairs(embedded_M, region_map)
            embedded_bits_M, aux_data_M = self._embed_in_channel(
                embedded_M, bits_to_embed[:len(pairs_M)], pairs_M, channel_type='M'
            )
            self.embedded_pairs_info['M'] = pairs_M[:embedded_bits_M]
            embedded_bits += embedded_bits_M
            auxiliary_data.extend(aux_data_M)
            bits_to_embed = bits_to_embed[embedded_bits_M:]
        if self.config.enable_H_channel and bits_to_embed:
            pairs_H = self.find_embedding_pairs(embedded_H, region_map)
            embedded_bits_H, aux_data_H = self._embed_in_channel(
                embedded_H, bits_to_embed[:len(pairs_H)], pairs_H, channel_type='H'
            )
            self.embedded_pairs_info['H'] = pairs_H[:embedded_bits_H]
            embedded_bits += embedded_bits_H
            auxiliary_data.extend(aux_data_H)
            bits_to_embed = bits_to_embed[embedded_bits_H:]
        embedding_info = {
            'embedded_bits': embedded_bits,
            'remaining_bits': ''.join(bits_to_embed),
            'auxiliary_data': auxiliary_data,
            'auxiliary_data_size': len(auxiliary_data) * 32,
            'embedding_pairs': embedded_bits,
            'watermark_length': len(watermark_bits)
        }
        return embedded_H, embedded_M, embedded_L, embedding_info

    def _embed_in_channel(self, channel: np.ndarray, bits: List[str], pairs: List[Tuple], channel_type: str) -> Tuple[int, List]:
        embedded_count = 0
        auxiliary_data = []
        for i, (row, col1, col2) in enumerate(pairs):
            if i >= len(bits):
                break
            bit = bits[i]
            if channel[row, col2] == 0 and bit == '1':
                auxiliary_data.append(row * channel.shape[1] + col2)
                continue
            if bit == '1':
                channel[row, col2] = channel[row, col2] - 1
            embedded_count += 1
        return embedded_count, auxiliary_data

    def extract(self, embedded_image: np.ndarray, embedding_info: Dict) -> Tuple[str, np.ndarray]:
        H_embedded, M_embedded, L_embedded = self.pixel_value_splitting(embedded_image)
        pairs_info = embedding_info.get('embedded_pairs_info', {})
        auxiliary_data = set(embedding_info.get('auxiliary_data', []))
        watermark_length = embedding_info.get('watermark_length', 0)
        extracted_bits = []
        def extract_bits(channel, pairs):
            bits = []
            for row, col1, col2 in pairs:
                if (row * channel.shape[1] + col2) in auxiliary_data:
                    continue
                if channel[row, col1] == channel[row, col2]:  # '0'
                    bits.append('0')
                elif channel[row, col1] == channel[row, col2] + 1:  # '1'
                    bits.append('1')
                if len(bits) + len(extracted_bits) >= watermark_length:
                    break
            return bits
        if 'L' in pairs_info:
            extracted_bits.extend(extract_bits(L_embedded, pairs_info['L']))
        if 'M' in pairs_info and len(extracted_bits) < watermark_length:
            extracted_bits.extend(extract_bits(M_embedded, pairs_info['M']))
        if 'H' in pairs_info and len(extracted_bits) < watermark_length:
            extracted_bits.extend(extract_bits(H_embedded, pairs_info['H']))
        H_recovered, M_recovered, L_recovered = self._recover_channels(H_embedded, M_embedded, L_embedded, embedding_info)
        recovered_image = self.reconstruct_image(H_recovered, M_recovered, L_recovered)
        return ''.join(extracted_bits[:watermark_length]), recovered_image

    def _recover_channels(self, H, M, L, embedding_info: Dict):
        pairs_info = embedding_info.get('embedded_pairs_info', {})
        auxiliary_data = set(embedding_info.get('auxiliary_data', []))
        for row, col1, col2 in pairs_info.get('L', []):
            if (row * L.shape[1] + col2) in auxiliary_data:
                continue
            if L[row, col1] == L[row, col2] + 1:
                L[row, col2] = L[row, col2] + 1
        for row, col1, col2 in pairs_info.get('M', []):
            if (row * M.shape[1] + col2) in auxiliary_data:
                continue
            if M[row, col1] == M[row, col2] + 1:
                M[row, col2] = M[row, col2] + 1
        for row, col1, col2 in pairs_info.get('H', []):
            if (row * H.shape[1] + col2) in auxiliary_data:
                continue
            if H[row, col1] == H[row, col2] + 1:
                H[row, col2] = H[row, col2] + 1
        return H, M, L

    def calculate_metrics(self, original: np.ndarray, embedded: np.ndarray) -> Dict:
        mse = np.mean((original.astype(np.float32) - embedded.astype(np.float32)) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 10 * np.log10(255**2 / mse)
        ssim = self._calculate_ssim(original, embedded)
        max_absolute_error = np.max(np.abs(original.astype(np.int16) - embedded.astype(np.int16)))
        return {
            'mse': float(mse),
            'psnr': float(psnr),
            'ssim': float(ssim),
            'max_absolute_error': int(max_absolute_error)
        }

    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        var1 = np.var(img1)
        var2 = np.var(img2)
        cov12 = np.mean((img1 - mu1) * (img2 - mu2))
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        numerator = (2 * mu1 * mu2 + C1) * (2 * cov12 + C2)
        denominator = (mu1**2 + mu2**2 + C1) * (var1 + var2 + C2)
        if denominator == 0:
            return 1.0
        return numerator / denominator

    def _validate_inputs(self, image: np.ndarray, watermark_bits: str):
        if image.dtype != np.uint8:
            raise ValueError("Image must be uint8 type")
        if len(image.shape) != 2:
            raise ValueError("Image must be grayscale (2D)")
        if not isinstance(watermark_bits, str):
            raise ValueError("Watermark must be string of bits")
        if not all(c in '01' for c in watermark_bits):
            raise ValueError("Watermark must contain only '0' and '1' characters")
        if len(watermark_bits) == 0:
            raise ValueError("Watermark cannot be empty")

    def calculate_capacity(self, image: np.ndarray) -> Dict:
        H, M, L = self.pixel_value_splitting(image)
        # Fast variant, no region classification for estimation
        def fast_pairs(channel):
            height, width = channel.shape
            pairs = 0
            for i in range(height):
                for j in range(0, width-1, 2):
                    if channel[i, j] == channel[i, j+1] and channel[i, j] > 1:
                        pairs += 1
            return pairs
        capacity_L = fast_pairs(L) if self.config.enable_L_channel else 0
        capacity_M = fast_pairs(M) if self.config.enable_M_channel else 0
        capacity_H = fast_pairs(H) if self.config.enable_H_channel else 0
        total_capacity_bits = capacity_L + capacity_M + capacity_H
        if self.config.enable_dual_pass:
            estimated_second_pass = int(total_capacity_bits * 0.3)
            total_capacity_bits_dual = total_capacity_bits + estimated_second_pass
        else:
            total_capacity_bits_dual = total_capacity_bits
            estimated_second_pass = 0
        total_pixels = image.size
        bpp_single_pass = total_capacity_bits / total_pixels
        bpp_dual_pass = total_capacity_bits_dual / total_pixels
        max_chars_single = total_capacity_bits // 8
        max_chars_dual = total_capacity_bits_dual // 8
        return {
            'image_shape': image.shape,
            'total_pixels': total_pixels,
            'capacity_bits_single_pass': total_capacity_bits,
            'capacity_bits_dual_pass': total_capacity_bits_dual,
            'estimated_second_pass_bits': estimated_second_pass,
            'bpp_single_pass': bpp_single_pass,
            'bpp_dual_pass': bpp_dual_pass,
            'max_characters_single_pass': max_chars_single,
            'max_characters_dual_pass': max_chars_dual,
            'channel_L': capacity_L,
            'channel_M': capacity_M,
            'channel_H': capacity_H
        }

def embed_watermark(image: np.ndarray, watermark: str, **kwargs) -> Tuple[np.ndarray, Dict]:
    watermark_bits = ''.join(format(ord(c), '08b') for c in watermark)
    config = EMLPVSConfig(**kwargs)
    eml = EML_PVS_RDH(config)
    return eml.embed(image, watermark_bits)

def extract_watermark(embedded_image: np.ndarray, embedding_info: Dict) -> Tuple[str, np.ndarray]:
    config = EMLPVSConfig(**embedding_info['config'])
    eml = EML_PVS_RDH(config)
    watermark_bits, recovered_image = eml.extract(embedded_image, embedding_info)
    watermark_text = ''.join(chr(int(watermark_bits[i:i+8], 2))
                            for i in range(0, len(watermark_bits), 8)
                            if i+8 <= len(watermark_bits))
    return watermark_text, recovered_image

# ----------- MAIN EXECUTION EXAMPLE -----------

img_path = "/content/icons8-ai-100.png"  # Replace with your grayscale image file
img_pil = Image.open(img_path).convert('L')
image = np.array(img_pil)

watermark = "Enhanced PVS-RDH with 67% better capacity!"

# Embed and extract
embedded_img, info = embed_watermark(image, watermark, verbose=True)
extracted_text, recovered_img = extract_watermark(embedded_img, info)

print("Extracted Watermark:", extracted_text)
print("\n--- Quality Metrics ---")
for k, v in info['quality_metrics'].items():
    print(f"{k.upper()}: {v}")
print(f"\nPerfect recovery: {np.array_equal(image, recovered_img)}")
print(f"Total embedded bits: {info.get('total_embedded_bits', 0)} / {info.get('watermark_length', 0)}")

# Display images
plt.figure(figsize=(12, 4))
plt.subplot(1,3,1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(embedded_img, cmap='gray')
plt.title("Embedded Image")
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(recovered_img, cmap='gray')
plt.title("Recovered Image")
plt.axis('off')
plt.show()

plt.figure(figsize=(6,3))
plt.hist(image.ravel(), bins=256, range=(0,255), alpha=0.5, label='Original', color='blue')
plt.hist(embedded_img.ravel(), bins=256, range=(0,255), alpha=0.5, label='Embedded', color='red')
plt.title("Histogram: Original vs Embedded Image")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# ----------- CAPACITY CHECK DEMO -----------
eml = EML_PVS_RDH()
cap_info = eml.calculate_capacity(image)
print("\n--- Capacity Analysis ---")
print(f"Max characters (dual pass): {cap_info['max_characters_dual_pass']}")
print(f"Bits per pixel (dual pass): {cap_info['bpp_dual_pass']:.4f}")
print(f"Single/dual pass bits: {cap_info['capacity_bits_single_pass']} / {cap_info['capacity_bits_dual_pass']}")
print(f"Channel capacities: L={cap_info['channel_L']}, M={cap_info['channel_M']}, H={cap_info['channel_H']}")

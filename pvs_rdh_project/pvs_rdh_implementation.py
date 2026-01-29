import numpy as np
import struct

class PVS_RDH:
    def __init__(self, K1=4, K2=6, overlap=False):
        """
        K1 + K2 must equal 10 (because B in [0..9]).
        overlap: if False (default) form non-overlapping horizontal pairs (q, q+1 with step 2).
                 if True, use every adjacent pair (slower, higher capacity).
        """
        if K1 + K2 != 10:
            raise ValueError("K1 + K2 must equal 10")
        self.K1 = int(K1)
        self.K2 = int(K2)
        self.overlap = bool(overlap)

    # --- helpers for bit-strings ---
    @staticmethod
    def bits_from_string(bitstr):
        """Take string of '0'/'1' characters and return list of ints (0/1)."""
        return [1 if c == '1' else 0 for c in bitstr]

    @staticmethod
    def string_from_bits(bit_list):
        return ''.join('1' if b else '0' for b in bit_list)

    @staticmethod
    def _pack_S(beta_indices):
        """
        Pack bookkeeping S:
        - 16-bit unsigned: count
        - For each index: 32-bit unsigned
        Returns bytes.
        """
        count = len(beta_indices)
        packed = struct.pack(">H", count)  # big-endian 16-bit
        if count:
            packed += b''.join(struct.pack(">I", int(idx)) for idx in beta_indices)
        return packed

    @staticmethod
    def _unpack_S(packed_bytes):
        """Return list of indices."""
        if len(packed_bytes) < 2:
            return []
        count = struct.unpack(">H", packed_bytes[:2])[0]
        indices = []
        offset = 2
        for _ in range(count):
            if offset + 4 > len(packed_bytes):
                raise ValueError("S data truncated")
            idx = struct.unpack(">I", packed_bytes[offset:offset+4])[0]
            indices.append(idx)
            offset += 4
        return indices

    def _scan_pairs(self, H, W):
        """Yield pair coordinate tuples (p,q,q+1) according to overlap mode."""
        if self.overlap:
            for r in range(H):
                for c in range(W - 1):
                    yield (r, c, c + 1)
        else:
            for r in range(H):
                for c in range(0, W - 1, 2):
                    yield (r, c, c + 1)

    def embed_watermark(self, img_uint8, watermark_bits):
        """
        img_uint8: 2D numpy uint8 grayscale image
        watermark_bits: string of '0'/'1' bits (e.g. ''.join(format(ord(c),'08b') for c in text))
        Returns: (embedded_image (uint8 numpy), info dict)
        info contains serialized S and metadata required for extraction.
        """
        if img_uint8.dtype != np.uint8:
            raise ValueError("Image must be uint8 grayscale")
        H, W = img_uint8.shape

        # Split
        A = (img_uint8 // 10).astype(np.int16)  # A in [0..25]
        B = (img_uint8 % 10).astype(np.int16)  # B in [0..9]

        # Build list of candidate pair indices (linear index for second pixel in pair)
        candidate_coords = []
        for r, c1, c2 in self._scan_pairs(H, W):
            if A[r, c1] == A[r, c2] and A[r, c1] > 1:
                candidate_coords.append((r, c1, c2))

        gamma = len(candidate_coords)
        # We'll build bookkeeping beta for underflow cases where A[r,c2]==0 and we'd need to -1
        beta = []
        # we will build a bitlist to embed (S + W'), but first compute S size
        # S is packed as: 16-bit count + 32-bit per index
        # convert watermark to bit list
        bits = self.bits_from_string(watermark_bits)

        # For simplicity, determine beta by seeing which candidates WOULD require subtract 1 when bit=1
        # but we only reserve/unmodify those at embedding time; still record their indices now:
        for (r, c1, c2) in candidate_coords:
            if A[r, c2] == 0:
                # would cause underflow if we tried to subtract 1 to embed '1'
                beta.append(r * W + c2)

        # pack S
        S_bytes = self._pack_S(beta)
        Sl = len(S_bytes) * 8  # bits

        # total available bits = gamma
        if gamma <= Sl:
            raise ValueError(f"Not enough capacity: gamma={gamma} <= bookkeeping bits Sl={Sl}")

        # remaining bits available for watermark payload
        payload_capacity = gamma - Sl
        # Trim watermark if necessary
        if len(bits) > payload_capacity:
            # either error or trim -- we trim (user can detect from info)
            bits = bits[:payload_capacity]

        # Build full bitstream W = S || W'
        # convert S_bytes to bit string
        S_bitstr = ''.join(f"{byte:08b}" for byte in S_bytes)
        full_bits = [1 if c == '1' else 0 for c in (S_bitstr + self.string_from_bits(bits))]

        # Now embed across candidates. We'll iterate candidate by candidate and consume full_bits.
        embA = A.copy()
        embB = B.copy()
        bit_ptr = 0
        # When embedding we must skip modifying pair if it was in beta (i.e., underflow)
        beta_set = set(beta)

        for (r, c1, c2) in candidate_coords:
            if bit_ptr >= len(full_bits):
                break
            lin_idx_c2 = r * W + c2
            # If this candidate is marked as in beta (underflow) we must NOT modify it (store index in S)
            if lin_idx_c2 in beta_set:
                # do nothing; BUT S recorded that index so extractor knows it was skipped
                continue
            bit = full_bits[bit_ptr]
            bit_ptr += 1
            if bit == 1:
                # subtract 1 from A(r,c2)
                embA[r, c2] = embA[r, c2] - 1
            else:
                # leave as is
                pass

            # After we change A we adjust B at position c2 as per eq (6)
            bval = embB[r, c2]
            if bval < self.K2:
                bnew = bval + self.K1
            else:
                bnew = bval - self.K2
            # ensure in [0..9]
            bnew = int(np.clip(bnew, 0, 9))
            embB[r, c2] = bnew

        # Compose embedded image
        emb_img = (embA * 10 + embB).astype(np.uint8)

        # Build info dictionary
        info = {
            "H": int(H), "W": int(W),
            "K1": int(self.K1), "K2": int(self.K2),
            "overlap": bool(self.overlap),
            "gamma": int(gamma),
            "S_bytes": S_bytes,
            "payload_len": len(bits),   # number of watermark bits actually embedded
            "full_len": len(full_bits)
        }
        return emb_img, info

    def extract_watermark(self, emb_img_uint8, info):
        """
        Return (extracted_bits_string, recovered_image_uint8)
        info must be the dict returned by embed_watermark.
        """
        H = info["H"]; W = info["W"]
        K1 = info["K1"]; K2 = info["K2"]
        overlap = info["overlap"]

        if emb_img_uint8.dtype != np.uint8:
            raise ValueError("Image must be uint8 grayscale")

        A_p = (emb_img_uint8 // 10).astype(np.int16)
        B_p = (emb_img_uint8 % 10).astype(np.int16)

        # Recreate scan order
        # build candidate coords in same order used in embed
        candidate_coords = []
        if overlap:
            for r in range(H):
                for c in range(W - 1):
                    if A_p[r, c] == A_p[r, c + 1] and A_p[r, c] > 1:
                        candidate_coords.append((r, c, c + 1))
        else:
            for r in range(H):
                for c in range(0, W - 1, 2):
                    if A_p[r, c] == A_p[r, c + 1] and A_p[r, c] > 1:
                        candidate_coords.append((r, c, c + 1))

        # Extract bits by scanning candidates. We need to know S length (from its 16-bit header).
        # But S was embedded as the first Sl bits across embed candidates (skipping beta positions during embedding).
        # So extraction approach:
        # 1) build a list of indices where extraction would read/would skip (we need to know beta indices so first bytes of S are present in the stream)
        # To recover S we must reconstruct same skipping logic from candidates: the extractor reconstructs the bitstream by reading / deriving bits from A' differences.
        # But recall in embedding, candidates with A==A and A>1 were used; those with A[c2]==0 were appended to beta and not modified.
        # During extraction we DO NOT know beta until we read S; this is circular. The paper handles this by embedding S first (i.e., it's part of W so first bits read correspond to S header). With the scanning rules used above the reading logic is:
        #  - We attempt to read bits from every candidate pair in order by checking D' = A'[p,q] - A'[p,q+1]
        #  - If D' == 1 -> extracted bit 1 and we restore A''[p,q+1] = A'[p,q+1] + 1
        #  - If D' == 0 -> extracted bit 0 and A'' unchanged
        #  - If D' >= 2 => it was a shifted pair (from D>=1 shifting) -> we restore by adding 1 to A'[p,q+1] (reverse shifting)
        #
        # So we can simply iterate candidates and reconstruct a stream of bits (0/1) and also apply reverse shifting immediately.
        extracted_bits = []
        restoredA = A_p.copy()
        restoredB = B_p.copy()

        # First pass: derive a raw sequence of bits and simultaneously restore shifted pairs.
        for (r, c1, c2) in candidate_coords:
            d = int(restoredA[r, c1] - restoredA[r, c2])
            if d == 1:
                extracted_bits.append(1)
                # recover A''(p,q+1) = A'(p,q+1) + 1
                restoredA[r, c2] = restoredA[r, c2] + 1
            elif d == 0:
                extracted_bits.append(0)
                # nothing to change
            elif d >= 2:
                # reverse shifting case (12)
                restoredA[r, c2] = restoredA[r, c2] + 1
                # this position was not used for embedding a bit (it was a shifted pair), so we do NOT append a bit
                # but paper says D'>=2 means re-shifting; in practice we should not map it to a message bit. So do nothing else.
            else:
                # negative difference shouldn't occur, but if it does, treat as 0 (safe fallback)
                extracted_bits.append(0)

        # extracted_bits now contains a bitstream that equals S || W'
        # Reconstruct S_bytes: first 16 bits give count, but S was packed as bytes; so convert first 16 bits to get count and then read that many 32-bit indices.
        if len(extracted_bits) < 16:
            raise ValueError("Not enough extracted bits to recover bookkeeping")

        # build bit-string
        bitstr = ''.join('1' if b else '0' for b in extracted_bits)
        # recover first bytes corresponding to S: but S length in bits is dynamic (2 + 4*count)*8
        # read first 16 bits as count
        count = int(bitstr[:16], 2)
        expected_S_bits = 16 + count * 32
        if len(bitstr) < expected_S_bits:
            # may occur if embed used less bits than possible; raise helpful error
            raise ValueError("Extracted stream too short to recover S fully (bad info or mismatch).")

        # reconstruct S_bytes from those bits
        S_bits = bitstr[:expected_S_bits]
        # convert every 8 bits to byte
        S_bytes = bytes(int(S_bits[i:i+8], 2) for i in range(0, len(S_bits), 8))
        beta_indices = self._unpack_S(S_bytes)

        # Now W' bits follow
        Wprime_bits = bitstr[expected_S_bits: expected_S_bits + (info["payload_len"])]
        extracted_payload_bits = [1 if c == '1' else 0 for c in Wprime_bits]

        # Now reverse B adjustments for every candidate position that was modified.
        # Note: some positions were in beta (skipped), others had B changed during embedding.
        beta_set = set(beta_indices)
        payload_ptr = 0
        for (r, c1, c2) in candidate_coords:
            lin_idx_c2 = r * W + c2
            if lin_idx_c2 in beta_set:
                # this location was unmodified at embedding time (we recorded it), so skip reverse op
                continue
            # If this candidate actually carried a bit in the embedded stream depends on whether we consumed a bit for it.
            # During extraction we used extracted_bits in same order; the first bits after S correspond to actual W'
            if payload_ptr >= len(extracted_payload_bits):
                break
            # reverse the B mapping used in embedding:
            bprime = int(restoredB[r, c2])
            if bprime < K2:
                # in embedding we did bnew = b + K1  => so now b = bnew - K1
                borig = bprime - K1
            else:
                # embedding did bnew = b - K2 => so now b = bnew + K2
                borig = bprime + K2
            # clip to 0..9
            restoredB[r, c2] = int(np.clip(borig, 0, 9))

            # Advance payload pointer (we consumed one message bit here)
            payload_ptr += 1

        # Recombine restoredA and restoredB to get recovered image
        recovered_img = (restoredA * 10 + restoredB).astype(np.uint8)

        # Return extracted watermark bits as string and recovered image
        extracted_bitstring = ''.join('1' if b else '0' for b in extracted_payload_bits)
        return extracted_bitstring, recovered_img
    
    def calculate_metrics(img1, img2):
        mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
        psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float('inf')
        from skimage.metrics import structural_similarity as ssim
        ssim_val = ssim(img1, img2, data_range=255)
        return {'psnr': psnr, 'ssim': ssim_val, 'mse': mse}

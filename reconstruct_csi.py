import numpy as np
import os
import glob
from matplotlib import pyplot as plt

# Directories
raw_dir = "saved_csi"
phase_dir = "saved_csi/PLL+jumpSplit+LRT"
output_dir = "saved_csi/Reconstructed"

os.makedirs(output_dir, exist_ok=True)

# Find raw CSI datasets
raw_files = glob.glob(os.path.join(raw_dir, "*.npy"))

for raw_path in raw_files:

    filename = os.path.basename(raw_path)
    name, _ = os.path.splitext(filename)

    if "correctedCSI" in name:
        continue

    print(f"\nProcessing {name}")

    # ---- Load raw CSI ----
    csi_data = np.load(raw_path, allow_pickle=True)

    amplitude = np.abs(csi_data)

    # ---- Load phase segments ----
    phase_segments = []
    for i in range(1, 5):
        path = f"{phase_dir}/{name}_PLL_interval_{i}_LRT.npy"
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue
        phase_segments.append(np.load(path))

    # ---- Reconstruct each segment separately ----
    start_idx = 0

    for i, phase_seg in enumerate(phase_segments):

        seg_len = phase_seg.shape[2]
        end_idx = start_idx + seg_len

        amp_seg = amplitude[:, :, start_idx:end_idx]

        # Shape check
        if amp_seg.shape != phase_seg.shape:
            raise ValueError("Mismatch between amplitude and phase segment")

        # Reconstruct CSI
        complex_seg = amp_seg * np.exp(1j * phase_seg)

        print(f"Segment {i+1} shape:", complex_seg.shape)

        # ---- Save ----
        save_path = os.path.join(
            output_dir,
            f"{name}_PLL_interval_{i+1}_LRT_corrected.npy"
        )

        np.save(save_path, complex_seg)
        print(f"Saved → {save_path}")

        # ---- Optional visualization ----
        plt.figure(figsize=(6,4))
        plt.imshow(
            np.abs(complex_seg[0, :2000, :]),
            aspect='auto'
        )
        plt.colorbar(label="Amplitude")
        plt.xlabel("Subcarrier (segment)")
        plt.ylabel("Packet")
        plt.title(f"{name} – Segment {i+1} amplitude evolution")
        plt.show()

        start_idx = end_idx

print("\nAll datasets reconstructed (per segment).")
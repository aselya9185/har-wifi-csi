import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from utils import build_full, get_data_indices, get_target_indices

# =========================
# 0. DIRECTORIES
# =========================
amp_dir = "dataset/interpolation/amplitude"
phase_dir = "dataset/phase_processing/LRT"
raw_dir = "dataset/saved_csi_raw"
output_dir = "dataset/reconstructed_csi"

os.makedirs(output_dir, exist_ok=True)

# =========================
# 1. INDICES
# =========================
data_indices = get_data_indices()     # 234
target_indices = get_target_indices() # 244

# =========================
# 2. LOAD FILES
# =========================
amp_files = sorted(glob.glob(os.path.join(amp_dir, "*_amp.npy")))

for amp_path in amp_files:

    filename = os.path.basename(amp_path)
    name = filename.replace("_amp.npy", "")

    phase_path = os.path.join(phase_dir, f"{name}_phase.npy")
    raw_path = os.path.join(raw_dir, f"{name}.npy")

    if not os.path.exists(phase_path):
        print(f"Missing phase file for {name}")
        continue

    if not os.path.exists(raw_path):
        print(f"Missing raw CSI for {name}")
        continue

    print(f"\nProcessing {name}")

    # =========================
    # LOAD DATA
    # =========================
    amp = np.load(amp_path)
    phase = np.load(phase_path)
    csi_raw = np.load(raw_path)

    if amp.shape != phase.shape:
        raise ValueError(f"Shape mismatch: {amp.shape} vs {phase.shape}")

    # =========================
    # RECONSTRUCT
    # =========================
    csi_reconstructed = amp * np.exp(1j * phase)

    save_path = os.path.join(output_dir, f"{name}_reconstructed.npy")
    np.save(save_path, csi_reconstructed)

    print(f"Saved → {save_path}")

    # =========================
    # BUILD FULL 256
    # =========================
    csi_raw_full = build_full(csi_raw, data_indices)
    csi_reconstructed_full = build_full(csi_reconstructed, target_indices)

    # =========================
    # VISUALIZATION
    # =========================
    PACKETS = 500

    # ---- RAW CSI ----
    plt.figure(figsize=(6,4))
    plt.imshow(
        np.abs(csi_raw_full[0, :PACKETS, :]),
        aspect='auto',
        extent=[-128, 127, PACKETS, 0]   # 👈 FIXED AXIS
    )
    plt.colorbar(label="Amplitude")
    plt.xlabel("Subcarrier index k")
    plt.ylabel("Packet index")
    plt.title(f"{name} – RAW CSI")
    plt.tight_layout()
    plt.show()

    # ---- RECONSTRUCTED CSI ----
    plt.figure(figsize=(6,4))
    plt.imshow(
        np.abs(csi_reconstructed_full[0, :PACKETS, :]),
        aspect='auto',
        extent=[-128, 127, PACKETS, 0]   # 👈 FIXED AXIS
    )
    plt.colorbar(label="Amplitude")
    plt.xlabel("Subcarrier index k")
    plt.ylabel("Packet index")
    plt.title(f"{name} – RECONSTRUCTED CSI")
    plt.tight_layout()
    plt.show()

print("\nAll datasets reconstructed.")
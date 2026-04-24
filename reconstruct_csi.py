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
# 2. RECONSTRUCTION (ALL FILES)
# =========================
amp_files = sorted(glob.glob(os.path.join(amp_dir, "*_amp.npy")))

for amp_path in amp_files:

    filename = os.path.basename(amp_path)
    name = filename.replace("_amp.npy", "")

    phase_path = os.path.join(phase_dir, f"{name}_phase.npy")

    if not os.path.exists(phase_path):
        print(f"Missing phase file for {name}")
        continue

    amp = np.load(amp_path)
    phase = np.load(phase_path)

    if amp.shape != phase.shape:
        raise ValueError(f"Shape mismatch: {amp.shape} vs {phase.shape}")

    csi_reconstructed = amp * np.exp(1j * phase)

    save_path = os.path.join(output_dir, f"{name}_reconstructed.npy")
    np.save(save_path, csi_reconstructed)

    print(f"Saved → {save_path}")

print("\nAll datasets reconstructed.")

# =========================
# 3. LOAD ONLY 2 DATASETS
# =========================
datasets = ["r1_empty", "r1_walking_1"]

data = {}

for name in datasets:

    raw_path = os.path.join(raw_dir, f"{name}.npy")
    rec_path = os.path.join(output_dir, f"{name}_reconstructed.npy")

    if not os.path.exists(raw_path) or not os.path.exists(rec_path):
        raise FileNotFoundError(f"Missing files for {name}")

    csi_raw = np.load(raw_path)
    csi_rec = np.load(rec_path)

    # build full 256
    raw_full = build_full(csi_raw, data_indices)
    rec_full = build_full(csi_rec, target_indices)

    data[name] = {
        "raw": raw_full,
        "rec": rec_full
    }

# =========================
# 4. VISUALIZATION SETTINGS
# =========================
PACKETS = 500

# =========================
# 5. BEFORE (RAW)
# =========================
fig_before, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig_before.suptitle("Before Reconstruction (Raw CSI Magnitude Heatmap)", fontsize=14)

im0 = axs[0].imshow(
    np.abs(data["r1_empty"]["raw"][0, :PACKETS, :]),
    aspect='auto',
    extent=[-128, 127, PACKETS, 0]
)
axs[0].set_title("Empty Room")
axs[0].set_xlabel("Subcarrier index k")
axs[0].set_ylabel("Packet index")

im1 = axs[1].imshow(
    np.abs(data["r1_walking_1"]["raw"][0, :PACKETS, :]),
    aspect='auto',
    extent=[-128, 127, PACKETS, 0]
)
axs[1].set_title("Walking")
axs[1].set_xlabel("Subcarrier index k")

# reserve space for colorbar
fig_before.subplots_adjust(left=0.06, right=0.88, top=0.88, bottom=0.12, wspace=0.15)

# add colorbar OUTSIDE
cbar_ax = fig_before.add_axes([0.90, 0.15, 0.02, 0.7])
fig_before.colorbar(im1, cax=cbar_ax, label="Amplitude")

plt.show()

# =========================
# 6. AFTER (RECONSTRUCTED)
# =========================
fig_after, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig_after.suptitle("After Reconstruction (CSI Magnitude Heatmap)", fontsize=14)

im0 = axs[0].imshow(
    np.abs(data["r1_empty"]["rec"][0, :PACKETS, :]),
    aspect='auto',
    extent=[-128, 127, PACKETS, 0]
)
axs[0].set_title("Empty Room")
axs[0].set_xlabel("Subcarrier index k")
axs[0].set_ylabel("Packet index")

im1 = axs[1].imshow(
    np.abs(data["r1_walking_1"]["rec"][0, :PACKETS, :]),
    aspect='auto',
    extent=[-128, 127, PACKETS, 0]
)
axs[1].set_title("Walking")
axs[1].set_xlabel("Subcarrier index k")

# reserve space
fig_after.subplots_adjust(left=0.06, right=0.88, top=0.88, bottom=0.12, wspace=0.15)

# colorbar outside
cbar_ax = fig_after.add_axes([0.90, 0.15, 0.02, 0.7])
fig_after.colorbar(im1, cax=cbar_ax, label="Amplitude")

plt.show()
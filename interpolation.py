import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from utils import (
    k_axis,
    set_pi_ticks,
    get_target_indices,
    build_full
)

# =========================
# 1. DEFINE CONSTANTS
# =========================

REMOVED = [0,1,2,3,4,5,25,53,89,117,127,128,129,139,167,203,231,251,252,253,254,255]
PILOTS = [25, 53, 89, 117, 127, 129, 139, 167, 203, 231]

# =========================
# 2. INDEX MAPPING
# =========================

all_idx = np.arange(256)

mask = np.ones(256, dtype=bool)
mask[REMOVED] = False
data_indices = all_idx[mask]   # 234

target_indices = get_target_indices()  # 244

# =========================
# 3. INTERPOLATION
# =========================

def interpolate_pilots_packet(csi_packet):
    """
    Returns:
        amplitude_interp (244,)
        phase_interp (244,)  ← UNWRAPPED (correct domain)
    """

    full_amp = np.full(256, np.nan)
    full_phase = np.full(256, np.nan)

    # --- extract ---
    amp = np.abs(csi_packet)
    phase = np.unwrap(np.angle(csi_packet))

    # --- place known ---
    full_amp[data_indices] = amp
    full_phase[data_indices] = phase

    # --- interpolate pilots ---
    missing = sorted(PILOTS)

    i = 0
    while i < len(missing):

        start = missing[i]
        j = i

        while j + 1 < len(missing) and missing[j+1] == missing[j] + 1:
            j += 1

        end = missing[j]

        left = start - 1
        while left >= 0 and np.isnan(full_amp[left]):
            left -= 1

        right = end + 1
        while right < 256 and np.isnan(full_amp[right]):
            right += 1

        if left >= 0 and right < 256:
            for p in range(start, end + 1):
                ratio = (p - left) / (right - left)

                full_amp[p] = full_amp[left] + (full_amp[right] - full_amp[left]) * ratio
                full_phase[p] = full_phase[left] + (full_phase[right] - full_phase[left]) * ratio
        else:
            for p in range(start, end + 1):
                full_amp[p] = np.nan
                full_phase[p] = np.nan

        i = j + 1

    return full_amp[target_indices], full_phase[target_indices]

# =========================
# 4. PROCESS CSI
# =========================

def process_csi_array(csi):
    A, T, _ = csi.shape

    amp_out = np.full((A, T, len(target_indices)), np.nan)
    phase_out = np.full((A, T, len(target_indices)), np.nan)

    for a in range(A):
        for t in range(T):
            amp_out[a, t], phase_out[a, t] = interpolate_pilots_packet(csi[a, t])

    return amp_out, phase_out

# =========================
# 5. SAVE
# =========================

def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".npy"):
            print(f"Processing: {filename}")

            csi = np.load(os.path.join(input_dir, filename))

            amp, phase = process_csi_array(csi)

            np.save(os.path.join(output_dir, filename.replace(".npy", "_amp.npy")), amp)
            np.save(os.path.join(output_dir, filename.replace(".npy", "_phase.npy")), phase)

# =========================
# 6. HELPERS
# =========================

def build_full_before(csi):
    A, T, _ = csi.shape
    full = np.full((A, T, 256), np.nan, dtype=np.complex128)
    full[:, :, data_indices] = csi
    return full

# =========================
# 7. PLOTTING
# =========================

def plot_dataset(ax, data, is_phase=False):
    k = k_axis()

    for a in range(data.shape[0]):
        ax.plot(k, np.ma.masked_invalid(data[a]), label=f"Antenna {a}", linewidth=1)

    for pilot in PILOTS:
        ax.axvline(x=pilot - 128, color='pink', linestyle='--', linewidth=0.7)

    ax.grid(True)
    ax.legend(fontsize=8)

# =========================
# 8. PHASE UNWRAP WITH NAN HElPER
# =========================

def unwrap_with_nan(x):
    out = x.copy()
    for a in range(x.shape[0]):
        valid = ~np.isnan(x[a])
        out[a, valid] = np.unwrap(x[a, valid])
    return out

# =========================
# 9. CORE PLOT
# =========================

def plot_pair(f1, f2, title):

    PACKET_IDX = 1000

    # BEFORE
    csi1_b = build_full_before(np.load(os.path.join("dataset/saved_csi_raw", f1)))[:, PACKET_IDX, :]
    csi2_b = build_full_before(np.load(os.path.join("dataset/saved_csi_raw", f2)))[:, PACKET_IDX, :]

    # AFTER
    amp1 = np.load(f"dataset/interpolation/{f1.replace('.npy', '_amp.npy')}")
    phase1 = np.load(f"dataset/interpolation/{f1.replace('.npy', '_phase.npy')}")

    amp2 = np.load(f"dataset/interpolation/{f2.replace('.npy', '_amp.npy')}")
    phase2 = np.load(f"dataset/interpolation/{f2.replace('.npy', '_phase.npy')}")

    amp1 = build_full(amp1, target_indices)[:, PACKET_IDX, :]
    phase1 = build_full(phase1, target_indices)[:, PACKET_IDX, :]

    amp2 = build_full(amp2, target_indices)[:, PACKET_IDX, :]
    phase2 = build_full(phase2, target_indices)[:, PACKET_IDX, :]

    phase1 = unwrap_with_nan(phase1)
    phase2 = unwrap_with_nan(phase2)

    # =========================
    # BEFORE
    # =========================
    fig, axs = plt.subplots(2,2, figsize=(12,8), sharex=True, sharey='row')

    axs[0,0].set_title("Room 1")
    axs[0,1].set_title("Room 2")

    fig.text(0.02, 0.75, "Amplitude", va='center', rotation='vertical', fontsize=12)
    fig.text(0.02, 0.25, "Phase", va='center', rotation='vertical', fontsize=12)

    plot_dataset(axs[0,0], np.abs(csi1_b))
    plot_dataset(axs[0,1], np.abs(csi2_b))

    phase1_b = unwrap_with_nan(np.angle(csi1_b))
    phase2_b = unwrap_with_nan(np.angle(csi2_b))

    plot_dataset(axs[1, 0], phase1_b)
    set_pi_ticks(axs[1, 0])

    plot_dataset(axs[1, 1], phase2_b)
    set_pi_ticks(axs[1, 1])

    fig.suptitle(f"{title} - BEFORE Interpolation")
    plt.tight_layout(rect=[0.06, 0, 1, 1])
    plt.show()

    # =========================
    # AFTER
    # =========================
    fig, axs = plt.subplots(2, 2, figsize=(12,8), sharex=True, sharey='row')

    axs[0,0].set_title("Room 1")
    axs[0,1].set_title("Room 2")

    fig.text(0.02, 0.75, "Amplitude", va='center', rotation='vertical', fontsize=12)
    fig.text(0.02, 0.25, "Phase", va='center', rotation='vertical', fontsize=12)

    plot_dataset(axs[0,0], amp1)
    plot_dataset(axs[0,1], amp2)

    plot_dataset(axs[1, 0], phase1)
    set_pi_ticks(axs[1, 0])

    plot_dataset(axs[1, 1], phase2)
    set_pi_ticks(axs[1, 1])

    fig.suptitle(f"{title} - AFTER Interpolation")
    plt.tight_layout(rect=[0.06, 0, 1, 1])
    plt.show()

# =========================
# 10. RUN
# =========================

process_directory("dataset/saved_csi_raw", "dataset/interpolation")

pairs = [
    ("r1_empty.npy", "r2_empty_1.npy", "Empty"),
    ("r1_sitting_1.npy", "r2_sit_1.npy", "Sitting"),
    ("r1_standing_1.npy", "r2_standing_1.npy", "Standing"),
    ("r1_walking_1.npy", "r2_walk_1.npy", "Walking"),
]

for f1, f2, title in pairs:
    plot_pair(f1, f2, title)
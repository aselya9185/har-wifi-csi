import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import matplotlib.ticker as ticker

# =========================
# 0. DIRECTORIES
# =========================
input_dir = "dataset/interpolation"
output_dir = "dataset/phase_processing/PLL"
os.makedirs(output_dir, exist_ok=True)

PACKET_IDX = 1000

# =========================
# 1. INDEX MAPPING (IMPORTANT)
# =========================

REMOVED = [0,1,2,3,4,5,25,53,89,117,127,128,129,139,167,203,231,251,252,253,254,255]
NON_RECONSTRUCT = [0,1,2,3,4,5,128,251,252,253,254,255]

all_idx = np.arange(256)

target_mask = np.ones(256, dtype=bool)
target_mask[NON_RECONSTRUCT] = False
target_indices = np.where(target_mask)[0]   # 242

# =========================
# 2. HELPERS
# =========================

def k_axis():
    return np.arange(-128, 128)

# -------- reconstruct full 256 --------
def build_full_phase(phase):
    A, T, K = phase.shape
    full = np.full((A, T, 256), np.nan)
    full[:, :, target_indices] = phase
    return full

# -------- NaN-safe unwrap --------
def unwrap_with_nan_full(phase):
    out = phase.copy()
    for a in range(phase.shape[0]):
        for t in range(phase.shape[1]):
            valid = ~np.isnan(phase[a, t])
            if np.sum(valid) > 1:
                out[a, t, valid] = np.unwrap(phase[a, t, valid])
    return out

# -------- single packet unwrap --------
def unwrap_with_nan_single(x):
    out = x.copy()
    for a in range(x.shape[0]):
        valid = ~np.isnan(x[a])
        if np.sum(valid) > 1:
            out[a, valid] = np.unwrap(x[a, valid])
    return out

# -------- PLL bias removal --------
def remove_per_antenna_bias_full(phases):
    out = phases.copy()

    for t in range(phases.shape[1]):

        # find reference index from antenna 0
        valid = ~np.isnan(phases[0, t])
        if not np.any(valid):
            continue

        ref_idx = np.where(valid)[0][0]

        for a in range(phases.shape[0]):
            if not np.isnan(phases[a, t, ref_idx]):
                bias = phases[a, t, ref_idx]
                valid_a = ~np.isnan(phases[a, t])
                out[a, t, valid_a] = phases[a, t, valid_a] - bias

    return out

# -------- plotting --------
def plot_dataset(ax, data):
    k = k_axis()
    for a in range(data.shape[0]):
        ax.plot(k, np.ma.masked_invalid(data[a]), label=f"Antenna {a}", linewidth=1)

    ax.grid(True)
    ax.legend(fontsize=8)

# -------- π ticks --------
def set_pi_ticks(ax):
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2*np.pi))
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda val, pos: f"{val/np.pi:.0g}π" if val != 0 else "0")
    )

# =========================
# 3. PROCESS ALL FILES
# =========================

files = glob.glob(os.path.join(input_dir, "*_phase.npy"))

for file_path in files:
    filename = os.path.basename(file_path)
    print(f"Processing {filename}")

    phase = np.load(file_path)

    phase_unwrapped = unwrap_with_nan_full(phase)
    phase_pll = remove_per_antenna_bias_full(phase_unwrapped)

    np.save(os.path.join(output_dir, filename), phase_pll)

print("All files processed and saved.")

# =========================
# 4. LOAD EXAMPLES
# =========================

phase_empty = np.load(os.path.join(input_dir, "r1_empty_phase.npy"))
phase_walk = np.load(os.path.join(input_dir, "r1_walking_1_phase.npy"))

phase_empty_pll = np.load(os.path.join(output_dir, "r1_empty_phase.npy"))
phase_walk_pll = np.load(os.path.join(output_dir, "r1_walking_1_phase.npy"))

# reconstruct full
empty_full = build_full_phase(phase_empty)
walk_full = build_full_phase(phase_walk)

empty_pll_full = build_full_phase(phase_empty_pll)
walk_pll_full = build_full_phase(phase_walk_pll)

# extract packet
empty_before = unwrap_with_nan_single(empty_full[:, PACKET_IDX, :])
walk_before = unwrap_with_nan_single(walk_full[:, PACKET_IDX, :])

empty_after = unwrap_with_nan_single(empty_pll_full[:, PACKET_IDX, :])
walk_after = unwrap_with_nan_single(walk_pll_full[:, PACKET_IDX, :])

# =========================
# 5. FIX Y-LIMITS (NO AUTOSCALE)
# =========================

combined = np.concatenate([
    empty_before.flatten(),
    walk_before.flatten(),
    empty_after.flatten(),
    walk_after.flatten()
])

valid = combined[~np.isnan(combined)]

y_min = np.min(valid)
y_max = np.max(valid)

margin = 0.05 * (y_max - y_min)
y_min -= margin
y_max += margin

# =========================
# 6. PLOT BEFORE
# =========================

fig_before, axs_before = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

fig_before.suptitle("Before PLL Bias Removal", fontsize=14)

axs_before[0].set_title("Empty - BEFORE PLL")
axs_before[1].set_title("Walking - BEFORE PLL")

plot_dataset(axs_before[0], empty_before)
plot_dataset(axs_before[1], walk_before)

for ax in axs_before:
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Subcarrier Index")
    ax.set_ylabel("Phase")
    set_pi_ticks(ax)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# =========================
# 7. PLOT AFTER
# =========================

fig_after, axs_after = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

fig_after.suptitle("After PLL Bias Removal", fontsize=14)

axs_after[0].set_title("Empty - AFTER PLL")
axs_after[1].set_title("Walking - AFTER PLL")

plot_dataset(axs_after[0], empty_after)
plot_dataset(axs_after[1], walk_after)

for ax in axs_after:
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Subcarrier Index")
    ax.set_ylabel("Phase")
    set_pi_ticks(ax)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
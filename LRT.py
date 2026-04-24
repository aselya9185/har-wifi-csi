import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from utils import (
    build_full,
    get_target_indices,
    unwrap_with_nan_single,
    plot_dataset,
    set_pi_ticks
)
from matplotlib.lines import Line2D


# =========================
# 0. DIRECTORIES
# =========================
input_dir = "dataset/phase_processing/PLL"
output_dir = "dataset/phase_processing/LRT"
os.makedirs(output_dir, exist_ok=True)

PACKET_IDX = 1000

# =========================
# 1. INDEX SETUP
# =========================
target_indices = get_target_indices()  # 244

# segment boundaries (on 244 domain)
segments_bounds = [
    (0, 58),
    (58, 122),
    (122, 185),
    (185, 244)
]

# =========================
# 2. LRT FUNCTION
# =========================

def apply_lrt_segmented(phase):
    """
    phase: (A, T, 244)
    """
    A, T, K = phase.shape
    corrected = np.full_like(phase, np.nan)

    for a in range(A):
        for t in range(T):

            for (start, end) in segments_bounds:

                seg = phase[a, t, start:end]

                valid = ~np.isnan(seg)
                if np.sum(valid) < 2:
                    continue

                x = np.arange(start, end)[valid]
                y = seg[valid]

                # linear regression
                slope, intercept = np.polyfit(x, y, 1)
                trend = slope * x + intercept

                corrected[a, t, start:end][valid] = y - trend

    return corrected

# =========================
# 2. DRAW STITCHING LINES
# =========================

def plot_stitching_lines(ax, target_indices, segments_bounds):
    for i in range(1, len(segments_bounds)):

        prev_end = segments_bounds[i-1][1]
        curr_start = segments_bounds[i][0]

        # map both indices
        k_prev = target_indices[prev_end - 1]
        k_curr = target_indices[curr_start]

        # midpoint → visually correct boundary
        k_plot = ((k_prev + k_curr) / 2) - 128

        ax.axvline(x=k_plot, color='deeppink', linestyle='--', linewidth=0.7)


# =========================
# 3. PROCESS ALL FILES
# =========================

files = glob.glob(os.path.join(input_dir, "*_phase.npy"))

for file_path in files:
    filename = os.path.basename(file_path)
    print(f"Processing {filename}")

    phase = np.load(file_path)  # already unwrapped + PLL removed

    phase_lrt = apply_lrt_segmented(phase)

    np.save(os.path.join(output_dir, filename), phase_lrt)

print("All files processed.")

# =========================
# 4. LOAD EXAMPLES
# =========================

phase_empty = np.load(os.path.join(input_dir, "r1_empty_phase.npy"))
phase_walk = np.load(os.path.join(input_dir, "r1_walking_1_phase.npy"))

phase_empty_lrt = np.load(os.path.join(output_dir, "r1_empty_phase.npy"))
phase_walk_lrt = np.load(os.path.join(output_dir, "r1_walking_1_phase.npy"))

# =========================
# 5. REBUILD FULL 256 FOR PLOTTING
# =========================

empty_full = build_full(phase_empty, target_indices)
walk_full = build_full(phase_walk, target_indices)

empty_lrt_full = build_full(phase_empty_lrt, target_indices)
walk_lrt_full = build_full(phase_walk_lrt, target_indices)

# extract packet
empty_before = unwrap_with_nan_single(empty_full[:, PACKET_IDX, :])
walk_before = unwrap_with_nan_single(walk_full[:, PACKET_IDX, :])

empty_after = unwrap_with_nan_single(empty_lrt_full[:, PACKET_IDX, :])
walk_after = unwrap_with_nan_single(walk_lrt_full[:, PACKET_IDX, :])

# =========================
# 6. FIX Y LIMITS
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
# 7. PLOT BEFORE
# =========================

fig_before, axs_before = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

fig_before.suptitle("Before LRT (After PLL)", fontsize=14)

axs_before[0].set_title("Empty")
axs_before[1].set_title("Walking")

plot_dataset(axs_before[0], empty_before)
plot_dataset(axs_before[1], walk_before)

plot_stitching_lines(axs_before[0], target_indices, segments_bounds)
plot_stitching_lines(axs_before[1], target_indices, segments_bounds)

stitch_legend = Line2D(
    [0], [0],
    color='deeppink',
    linestyle='--',
    lw=1.2,
    label='Subband stitching points'
)

fig_before.legend(
    handles=[stitch_legend],
    loc='upper center',
    bbox_to_anchor=(0.5, 0.92),
    frameon=False
)

for ax in axs_before:
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Subcarrier Index")
    ax.set_ylabel("Phase")
    set_pi_ticks(ax)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# =========================
# 8. PLOT AFTER
# =========================

fig_after, axs_after = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

fig_after.suptitle("After LRT (Slope Removed)", fontsize=14)

axs_after[0].set_title("Empty")
axs_after[1].set_title("Walking")

plot_dataset(axs_after[0], empty_after)
plot_dataset(axs_after[1], walk_after)

plot_stitching_lines(axs_after[0], target_indices, segments_bounds)
plot_stitching_lines(axs_after[1], target_indices, segments_bounds)

stitch_legend = Line2D(
    [0], [0],
    color='deeppink',
    linestyle='--',
    lw=1.2,
    label='Subband stitching points'
)

fig_after.legend(
    handles=[stitch_legend],
    loc='upper center',
    bbox_to_anchor=(0.5, 0.92),
    frameon=False
)

for ax in axs_after:
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Subcarrier Index")
    ax.set_ylabel("Phase")
    set_pi_ticks(ax)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
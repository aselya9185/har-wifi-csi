import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import glob

# Directories
input_dir = "phase_processing/segments"
output_dir = "phase_processing/segments_LRT"
os.makedirs(output_dir, exist_ok=True)

packet_idx = 1000


# ---- Apply LRT to whole dataset and save ----
phase_files = glob.glob(os.path.join(input_dir, "*.npy"))

for file_path in phase_files:
    filename = os.path.basename(file_path)
    name, ext = os.path.splitext(filename)

    phase_data = np.load(file_path)
    num_ant, num_packets, num_sub = phase_data.shape
    subcarriers = np.arange(num_sub)

    corrected_dataset = np.zeros_like(phase_data)

    for ant in range(num_ant):
        for pkt in range(num_packets):
            raw_phase = phase_data[ant, pkt, :]
            slope, intercept = np.polyfit(subcarriers, raw_phase, 1)
            regression_line = slope * subcarriers + intercept
            corrected_dataset[ant, pkt, :] = raw_phase - regression_line

    save_path = os.path.join(output_dir, f"{name}_LRT.npy")
    np.save(save_path, corrected_dataset)


# ---- Plotting function (minimal change) ----
def plot_lrt_on_axis(ax, loaded_phase, title):

    num_antennas = loaded_phase.shape[0]

    for antenna in range(num_antennas):

        raw_phase = loaded_phase[antenna, packet_idx, :]
        subcarriers = np.arange(len(raw_phase))

        # LRT
        slope, intercept = np.polyfit(subcarriers, raw_phase, 1)
        regression_line = slope * subcarriers + intercept
        corrected_phase = raw_phase - regression_line

        color = f"C{antenna}"

        # Raw
        ax.plot(subcarriers, raw_phase,
                color=color, alpha=0.6)

        # Regression
        ax.plot(subcarriers, regression_line,
                linestyle='--', alpha=0.5, lw=0.8, color=color)

        # Corrected
        ax.plot(subcarriers, corrected_phase, color=color)

    ax.axhline(0, color='gray', linestyle=':')
    ax.set_title(title)
    ax.set_xlabel("Subcarrier Index $k$")
    ax.set_ylabel("Phase [radians]")
    ax.grid(True)
    ax.legend()

    ax.set_box_aspect(1)

# ---- Helper: load segments ----
def load_segments(prefix):

    segments = []
    for i in range(1,5):
        file = f"{input_dir}/{prefix}_seg_{i}.npy"
        segments.append(np.load(file))

    return segments

# ---- Load datasets ----
empty_segments = load_segments("r1_empty")
walking_segments = load_segments("r1_walking_1")


# ---- Plot Empty Room ----
fig_empty, axes = plt.subplots(2,2, figsize=(8,8), sharey=True)
axes = axes.flatten()

for i, seg in enumerate(empty_segments):
    plot_lrt_on_axis(
        axes[i],
        seg,
        f"Segment {i+1}"
    )

fig_empty.suptitle("LRT on CSI Phase Segments – Empty Room")


# ---- Plot Walking Human ----
fig_walk, axes = plt.subplots(2,2, figsize=(8,8), sharey=True)
axes = axes.flatten()

for i, seg in enumerate(walking_segments):
    plot_lrt_on_axis(
        axes[i],
        seg,
        f"Segment {i+1}"
    )

fig_walk.suptitle("LRT on CSI Phase Segments – Walking Human")


# ---- Style legend ----
style_legend = [
    Line2D([0], [0], color='black', lw=1, alpha=0.6, label='Raw Phase'),
    Line2D([0], [0], color='black', lw=0.8, alpha=0.5, linestyle='--', label='Regression Line $r_s(k)$'),
    Line2D([0], [0], color='black', lw=2, label='Corrected Phase')
]

fig_empty.legend(
    handles=style_legend,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.93),
    ncol=3,
    frameon=False
)

fig_walk.legend(
    handles=style_legend,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.93),
    ncol=3,
    frameon=False
)

antenna_legend = [
    Line2D([0], [0], color=f"C{i}", lw=2, label=f"Antenna {i}")
    for i in range(4)
]

fig_empty.legend(
    handles=antenna_legend,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.02),
    ncol=4,
    frameon=False
)

fig_walk.legend(
    handles=antenna_legend,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.02),
    ncol=4,
    frameon=False
)

fig_empty.tight_layout(rect=[0, 0.05, 1, 0.9])
fig_walk.tight_layout(rect=[0, 0.05, 1, 0.9])

plt.show()

# ----------------------------------------------------------------------

# # ---- Example plotting (Packet 1000 only) ----
# phase_empty = np.load("saved_csi/removed_PLL/r1_empty_PLL.npy")
# phase_walking = np.load("saved_csi/removed_PLL/r1_walking_1_PLL.npy")

# fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
#
# plot_lrt_on_axis(axes[0], phase_empty,
#                  "Empty Room – LRT per Antenna (Packet 1000)")
#
# plot_lrt_on_axis(axes[1], phase_walking,
#                  "Walking Human – LRT per Antenna (Packet 1000)")
#
# # ---- Style legend ----
# style_legend = [
#     Line2D([0], [0], color='black', lw=1, alpha=0.4, label='Raw Phase'),
#     Line2D([0], [0], color='black', lw=1, linestyle='--', label='Regression Line $r_s(k)'),
#     Line2D([0], [0], color='black', lw=2, label='Corrected Phase')
# ]
#
# fig.legend(handles=style_legend,
#            loc='upper center',
#            ncol=3,
#            frameon=False)
#
# plt.tight_layout(rect=[0, 0, 1, 0.92])
# plt.show()
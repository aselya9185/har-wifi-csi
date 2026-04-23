import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import glob

# Directories
input_dir = "saved_csi"
output_dir = os.path.join(input_dir, "LRT")
os.makedirs(output_dir, exist_ok=True)

packet_idx = 1000

# ---- Apply LRT to whole dataset and save phase-only ----
csi_files = glob.glob(os.path.join(input_dir, "*.npy"))

for file_path in csi_files:
    filename = os.path.basename(file_path)
    name, ext = os.path.splitext(filename)

    csi_data = np.load(file_path)
    num_ant, num_packets, num_sub = csi_data.shape
    subcarriers = np.arange(num_sub)

    corrected_dataset = np.zeros((num_ant, num_packets, num_sub))

    for ant in range(num_ant):
        for pkt in range(num_packets):
            raw_phase = np.unwrap(np.angle(csi_data[ant, pkt, :]))
            slope, intercept = np.polyfit(subcarriers, raw_phase, 1)
            regression_line = slope * subcarriers + intercept
            corrected_dataset[ant, pkt, :] = raw_phase - regression_line

    save_path = os.path.join(output_dir, f"{name}_LRT.npy")
    np.save(save_path, corrected_dataset)


# ---- Plotting function (unchanged logic) ----
def plot_lrt_on_axis(ax, loaded_csi, title):

    num_antennas = loaded_csi.shape[0]

    for antenna in range(num_antennas):

        csi_packet = loaded_csi[antenna, packet_idx, :]
        subcarriers = np.arange(len(csi_packet))

        # Raw phase
        raw_phase = np.unwrap(np.angle(csi_packet))

        # LRT
        slope, intercept = np.polyfit(subcarriers, raw_phase, 1)
        regression_line = slope * subcarriers + intercept
        corrected_phase = raw_phase - regression_line

        color = f"C{antenna}"

        ax.plot(subcarriers, raw_phase, color=color, alpha=0.5)
        ax.plot(subcarriers, regression_line, linestyle='--', lw=0.8, color=color, alpha=0.6)
        ax.plot(subcarriers, corrected_phase,
                color=color, label=f'Antenna {antenna}')

    ax.axhline(0, color='gray', linestyle=':')
    ax.set_title(title)
    ax.set_xlabel("Subcarrier Index $k$")
    ax.set_ylabel("Phase [radians]")
    ax.grid(True)
    ax.legend()


# ---- Example plotting ----
csi_empty = np.load('dataset/saved_csi_raw/r1_empty.npy')
csi_walking = np.load('dataset/saved_csi_raw/r1_walking_1.npy')

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

plot_lrt_on_axis(axes[0], csi_empty,
                 "Empty Room – LRT per Antenna (Packet 1000)")

plot_lrt_on_axis(axes[1], csi_walking,
                 "Walking Human – LRT per Antenna (Packet 1000)")

style_legend = [
    Line2D([0], [0], color='black', lw=1, alpha=0.5, label='Raw Phase'),
    Line2D([0], [0], color='black', lw=0.8, alpha=0.6, linestyle='--', label='Regression Line'),
    Line2D([0], [0], color='black', lw=1, label='Corrected Phase')
]

fig.legend(handles=style_legend,
           loc='upper center',
           ncol=3,
           frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from segment_utils import unwrap_phase_dataset

# -------- directories --------
input_dir = "dataset/interpolation"
output_dir = "phase_processing/segments"
os.makedirs(output_dir, exist_ok=True)

packet_idx = 1000

# stitching boundaries after deletion
split_points = [58, 121, 185]  # boundaries between segments

# -------- helper to split array --------
def split_phase_data(phase_data):

    segments = []

    start = 0
    for sp in split_points:
        segments.append(phase_data[:, :, start:sp])
        start = sp

    segments.append(phase_data[:, :, start:])  # last segment

    return segments


# -------- split and save dataset --------
phase_files = glob.glob(os.path.join(input_dir, "*.npy"))

for file_path in phase_files:

    filename = os.path.basename(file_path)
    name, _ = os.path.splitext(filename)

    phase_data = unwrap_phase_dataset(np.load(file_path))

    segments = split_phase_data(phase_data)

    for i, seg in enumerate(segments):

        save_path = os.path.join(
            output_dir,
            f"{name}_seg_{i+1}.npy"
        )

        np.save(save_path, seg)

print("Segments saved.")


# -------- plotting function --------
def plot_segments(dataset, title):

    num_ant = dataset.shape[0]

    segments = split_phase_data(dataset)

    fig, axes = plt.subplots(1, len(segments), figsize=(18,4), sharey=True)

    for seg_idx, seg in enumerate(segments):

        ax = axes[seg_idx]

        subcarriers = np.arange(seg.shape[2])

        for ant in range(num_ant):

            phase = seg[ant, packet_idx, :]
            phase = np.unwrap(phase)

            ax.plot(subcarriers, phase, label=f"Antenna {ant}")

        ax.set_title(f"{title}\nSegment {seg_idx+1}")
        ax.set_xlabel("Subcarrier Index")
        ax.grid(True)

        if seg_idx == 0:
            ax.set_ylabel("Phase (rad)")

    axes[-1].legend()

    plt.tight_layout()
    plt.show()


# -------- load example datasets --------
phase_empty = unwrap_phase_dataset(np.load("dataset/interpolation/r1_empty.npy"))
phase_walking = unwrap_phase_dataset(np.load("dataset/interpolation/r1_walking_1.npy"))

plot_segments(phase_empty, "Empty Room – Phase Segments")
plot_segments(phase_walking, "Walking Human – Phase Segments")
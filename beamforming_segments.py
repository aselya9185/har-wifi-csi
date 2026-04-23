import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from collections import defaultdict

input_dir = "saved_csi/Reconstructed"
# output_dir = "saved_csi/csi_distance_correct"
output_dir = "saved_csi/csi_distance_correct_empty_walking"

os.makedirs(output_dir, exist_ok=True)

W = 16
theta_vals = np.linspace(0, 2*np.pi, 180)

# ---- Which datasets to plot ----
plot_keywords = ["empty", "walking_1"]

# ---- Group segment files ----
groups = defaultdict(list)

files = glob.glob(os.path.join(input_dir, "*.npy"))

target_datasets = ["r1_empty", "r1_walking_1"]


for f in files:
    name = os.path.basename(f)
    base = name.split("_PLL_interval")[0]
    # groups[base].append(f)
    if base in target_datasets:
        groups[base].append(f)

# ---- Process all datasets ----
for base_name, segment_files in groups.items():

    print(f"\nProcessing dataset: {base_name}")

    segment_files = sorted(segment_files)
    segments = [np.load(f) for f in segment_files]

    print("segment_files:", segment_files)

    for i, seg in enumerate(segments):
        print(f"\n--- Segment {i + 1} ---")
        print("Shape:", seg.shape)

        print("Sample (antenna 0, first 3 packets, first 5 subcarriers):\n")
        print(seg[0, :3, :5])


    packets = segments[0].shape[1]  # number of packets

    best_distances = []
    best_thetas = []

    # ---- Sliding window ----
    for start in range(0, packets - W):

        max_mean_distance = 0
        best_theta = 0

        # ---- Beam scan ----
        for theta in theta_vals:

            w2 = np.exp(1j * theta)

            sum_sq_all = None

            for seg in segments:

                h1 = seg[0, start:start+W, :] # seg[antenna, packet, subcarrier]
                h2 = seg[1, start:start+W, :] # h2.shape = (W, subcarriers)

                ybf = h1 + w2 * h2 # ybf.shape = (W, subcarriers)

                diff = ybf[1:] - ybf[:-1] # y(n+1) - y(n) | shape = (W-1, subcarriers) | change of subcarrier k between time n and n+1

                sq = np.abs(diff)**2 # (W-1, subcarriers) == np.linalg.norm(diff, axis=1) = vector norm
                sq_sum = np.sqrt(np.sum(sq, axis=1)) # sum of ALL subcarrier changes in THIS segment for each time n

                if sum_sq_all is None:
                    sum_sq_all = sq_sum
                else:
                    sum_sq_all += sq_sum # sum of ALL subcarrier changes in ALL segments

            mean_d = np.mean(sum_sq_all) # average distance across window

            if mean_d > max_mean_distance:
                max_mean_distance = mean_d
                best_theta = theta

        best_distances.append(max_mean_distance)
        best_thetas.append(best_theta)

    best_distances = np.array(best_distances) # for all windows
    best_thetas = np.array(best_thetas)
    best_thetas_wrapped = (best_thetas + np.pi) % (2 * np.pi) - np.pi

    # ---- Save all datasets ----
    np.save(os.path.join(output_dir, f"{base_name}_distance.npy"), best_distances)
    np.save(os.path.join(output_dir, f"{base_name}_theta.npy"), best_thetas)

    # ---- Plot only selected datasets ----
    if any(keyword in base_name for keyword in plot_keywords):

        print(f"Plotting: {base_name}")

        # CSI distance
        plt.figure(figsize=(8,4))
        plt.plot(best_distances)
        plt.title(f"CSI Distance (correct) – {base_name}")
        plt.xlabel("Window index")
        plt.ylabel("Mean CSI distance")
        plt.grid(True)
        plt.show()

        # Beam angle
        plt.figure(figsize=(8,4))
        plt.plot(best_thetas_wrapped)
        plt.title(f"Optimal Beam Angle θ – {base_name}")
        plt.xlabel("Window index")
        plt.ylabel("θ (radians)")
        plt.grid(True)
        plt.show()

print("\nAll datasets processed. Only selected ones plotted.")

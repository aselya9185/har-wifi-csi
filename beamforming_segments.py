import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# SETTINGS
# =========================

input_dir = "dataset/reconstructed_csi"
output_dir = "dataset/csi_distance"

os.makedirs(output_dir, exist_ok=True)

W = 16
theta_vals = np.linspace(0, 2*np.pi, 180)

datasets = ["r1_empty", "r1_walking_1"]

# same segmentation as before
segments_bounds = [
    (0, 58),
    (58, 122),
    (122, 185),
    (185, 244)
]

# try antenna pairs
antenna_pairs = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3)
]

# =========================
# PROCESS
# =========================

for name in datasets:

    print(f"\nProcessing {name}")

    csi = np.load(os.path.join(input_dir, f"{name}_reconstructed.npy"))

    packets = csi.shape[1]

    for (ant1, ant2) in antenna_pairs:

        print(f"  Antennas {ant1}-{ant2}")

        best_distances = []
        best_thetas = []

        # ---- Sliding window ----
        for start in range(0, packets - W):

            max_mean_distance = 0
            best_theta = 0

            # ---- Beam scan ----
            for theta in theta_vals:

                w = np.exp(1j * theta)

                sum_sq_all = None

                for (start_k, end_k) in segments_bounds:

                    h1 = csi[ant1, start:start+W, start_k:end_k]
                    h2 = csi[ant2, start:start+W, start_k:end_k]

                    ybf = h1 + w * h2

                    diff = ybf[1:] - ybf[:-1]

                    sq = np.abs(diff)**2
                    sq_sum = np.sqrt(np.sum(sq, axis=1))

                    if sum_sq_all is None:
                        sum_sq_all = sq_sum
                    else:
                        sum_sq_all += sq_sum

                mean_d = np.mean(sum_sq_all)

                if mean_d > max_mean_distance:
                    max_mean_distance = mean_d
                    best_theta = theta

            best_distances.append(max_mean_distance)
            best_thetas.append(best_theta)

        best_distances = np.array(best_distances)
        best_thetas = np.array(best_thetas)

        best_thetas_wrapped = (best_thetas + np.pi) % (2*np.pi) - np.pi

        # =========================
        # SAVE
        # =========================

        np.save(os.path.join(output_dir, f"{name}_ant{ant1}{ant2}_dist.npy"), best_distances)
        np.save(os.path.join(output_dir, f"{name}_ant{ant1}{ant2}_theta.npy"), best_thetas)

        # =========================
        # PLOT
        # =========================

        plt.figure(figsize=(8,4))
        plt.plot(best_distances)
        plt.title(f"{name} – CSI Distance (Ant {ant1}-{ant2})")
        plt.xlabel("Window index")
        plt.ylabel("Mean CSI distance")
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(8,4))
        plt.plot(best_thetas_wrapped)
        plt.title(f"{name} – Optimal Beam θ (Ant {ant1}-{ant2})")
        plt.xlabel("Window index")
        plt.ylabel("θ (rad)")
        plt.grid(True)
        plt.show()

print("\nDone.")
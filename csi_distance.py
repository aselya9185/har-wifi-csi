import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# DIRECTORIES
# =========================
input_dir = "dataset/reconstructed_csi"
dist_dir = "dataset/csi_distance"
theta_dir = "dataset/theta"

os.makedirs(dist_dir, exist_ok=True)
os.makedirs(theta_dir, exist_ok=True)

# =========================
# PARAMETERS
# =========================
W = 16
theta_vals = np.linspace(-np.pi, np.pi, 180, endpoint=False)

datasets = [
    "r1_empty", "r2_empty_1",
    "r1_sitting_1", "r2_sit_1",
    "r1_standing_1", "r2_standing_1",
    "r1_walking_1", "r2_walk_1"
]

antenna_pairs = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3)
]

# =========================
# HELPER: circular distance
# =========================
def angle_diff(a, b):
    return np.angle(np.exp(1j * (a - b)))

import numpy as np

def set_wrapped_pi_ticks(ax):
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels([
        r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"
    ])

# =========================
# MAIN PROCESS
# =========================
for name in datasets:

    print(f"\nProcessing: {name}")

    path = os.path.join(input_dir, f"{name}_reconstructed.npy")

    if not os.path.exists(path):
        print(f"Missing file: {path}")
        continue

    csi = np.load(path)
    packets = csi.shape[1]

    for (ant1, ant2) in antenna_pairs:

        print(f"  Antennas: {ant1}-{ant2}")

        h1 = csi[ant1, :, :]
        h2 = csi[ant2, :, :]

        best_distances = []
        best_thetas = []

        prev_theta = 0.0

        for start in range(0, packets - W):

            window_h1 = h1[start:start + W]
            window_h2 = h2[start:start + W]

            candidates = []

            # ---- scan all theta ----
            for theta in theta_vals:
                w2 = np.exp(1j * theta)

                ybf = window_h1 + w2 * window_h2

                diff = ybf[1:] - ybf[:-1]
                d = np.sqrt(np.sum(np.abs(diff) ** 2, axis=1))
                mean_d = np.mean(d)

                candidates.append((theta, mean_d))

            # ---- sort by distance (descending) ----
            candidates = sorted(candidates, key=lambda x: -x[1])

            # ---- take top-K to avoid noise ----
            top_k = candidates[:5]

            # ---- choose theta closest to previous ----
            best_theta, best_distance = min(
                top_k,
                key=lambda x: abs(angle_diff(x[0], prev_theta))
            )

            prev_theta = best_theta

            best_distances.append(best_distance)
            best_thetas.append(best_theta)

        best_distances = np.array(best_distances)
        best_thetas = np.array(best_thetas)

        # =========================
        # WRAPPED

        # =========================
        best_thetas_wrapped = (best_thetas + np.pi) % (2 * np.pi) - np.pi

        # =========================
        # SAVE
        # =========================
        np.save(os.path.join(dist_dir, f"{name}_ant{ant1}{ant2}_dist.npy"), best_distances)
        np.save(os.path.join(theta_dir, f"{name}_ant{ant1}{ant2}_theta.npy"), best_thetas_wrapped)

        # =========================
        # DISTANCE PLOT
        # =========================
        plt.figure(figsize=(8, 4))
        plt.plot(best_distances)
        plt.title(f"{name} – (Ant {ant1}-{ant2})")
        plt.xlabel("Window index")
        plt.ylabel("Mean CSI distance")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # =========================
        # THETA PLOT
        # =========================
        plt.figure(figsize=(8, 4))

        plt.plot(best_thetas_wrapped)

        plt.title(f"{name} – (Ant {ant1}-{ant2})")
        plt.xlabel("Window index")
        plt.ylabel("θ")

        set_wrapped_pi_ticks(plt.gca())

        plt.grid(True)
        plt.tight_layout()
        plt.show()
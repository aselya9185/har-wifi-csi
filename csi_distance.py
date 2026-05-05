import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# DIRECTORIES
# =========================
input_dir = "dataset/reconstructed_csi"
dist_dir = "dataset/csi_distance"
theta_dir = "dataset/theta"

plot_dist_dir = "plots/csi_distance_new"
plot_theta_dir = "plots/optimal_beam_new"
plot_compare_dir = "plots/csi_distance_compare"

os.makedirs(dist_dir, exist_ok=True)
os.makedirs(theta_dir, exist_ok=True)

# =========================
# PARAMETERS
# =========================
W = 16
theta_vals = np.linspace(-np.pi, np.pi, 180, endpoint=False)

files = sorted([
    f for f in os.listdir(input_dir)
    if f.endswith("_reconstructed.npy")
])

antenna_pairs = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3)
]

# store for comparison plots
comparison_data = {}

# =========================
# HELPERS
# =========================
def angle_diff(a, b):
    return np.angle(np.exp(1j * (a - b)))

def set_wrapped_pi_ticks(ax):
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels([
        r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"
    ])

# =========================
# MAIN PROCESS
# =========================
for filename in files:

    name = filename.replace("_reconstructed.npy", "")
    path = os.path.join(input_dir, filename)

    if not os.path.exists(path):
        print(f"Missing file: {path}")
        continue

    # create folders for plots
    os.makedirs(os.path.join(plot_dist_dir, name), exist_ok=True)
    os.makedirs(os.path.join(plot_theta_dir, name), exist_ok=True)

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

        best_thetas_wrapped = (best_thetas + np.pi) % (2 * np.pi) - np.pi

        # =========================
        # SAVE
        # =========================
        np.save(os.path.join(dist_dir, f"{name}_ant{ant1}{ant2}_dist.npy"), best_distances)
        np.save(os.path.join(theta_dir, f"{name}_ant{ant1}{ant2}_theta.npy"), best_thetas_wrapped)

        # =========================
        # STORE FOR COMPARISON
        # =========================
        key = f"{name}_ant{ant1}{ant2}"
        comparison_data[key] = best_distances

        # =========================
        # DISTANCE PLOT
        # =========================
        plt.figure(figsize=(12, 6), dpi=150)

        plt.plot(best_distances)
        plt.title(f"{name} – (Ant {ant1}-{ant2})")
        plt.xlabel("Window index")
        plt.ylabel("Mean CSI distance")
        plt.ylim(0, 8e13)
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(
            plot_dist_dir, name, f"{name}_{ant1}-{ant2}.png"
        )
        plt.savefig(save_path)
        plt.close()

        # =========================
        # THETA PLOT
        # =========================
        plt.figure(figsize=(12, 6), dpi=150)

        plt.plot(best_thetas_wrapped)
        plt.title(f"{name} – (Ant {ant1}-{ant2})")
        plt.xlabel("Window index")
        plt.ylabel("θ")
        set_wrapped_pi_ticks(plt.gca())
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(
            plot_theta_dir, name, f"{name}_{ant1}-{ant2}.png"
        )
        plt.savefig(save_path)
        plt.close()


# =========================
# CSI DISTANCE COMPARISON PLOTS
# =========================

def plot_comparison(room_prefix, activity_names):
    os.makedirs(os.path.join(plot_compare_dir, room_prefix), exist_ok=True)

    for (ant1, ant2) in antenna_pairs:

        plt.figure(figsize=(12, 6), dpi=150)

        for name in activity_names:
            key = f"{name}_ant{ant1}{ant2}"
            if key in comparison_data:
                plt.plot(comparison_data[key], label=name)

        plt.title(f"{room_prefix.upper()} – Ant {ant1}-{ant2}")
        plt.xlabel("Window index")
        plt.ylabel("CSI distance")
        plt.ylim(0, 8e13)

        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(
            plot_compare_dir,
            room_prefix,
            f"{room_prefix}_ant_{ant1}-{ant2}.png"
        )

        plt.savefig(save_path)
        plt.close()


# R1 comparison
plot_comparison("r1", [
    "r1_empty",
    "r1_sitting_1",
    "r1_standing_1",
    "r1_walking_1"
])

# R2 comparison
plot_comparison("r2", [
    "r2_empty_1",
    "r2_sit_1",
    "r2_standing_1",
    "r2_walk_1"
])
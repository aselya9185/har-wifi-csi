import numpy as np
import matplotlib.pyplot as plt
import os

input_dir = "dataset/reconstructed_csi"
output_dir = "dataset/csi_distance_no_trick"

os.makedirs(output_dir, exist_ok=True)

W = 16
theta_vals = np.linspace(-np.pi, np.pi, 180, endpoint=False)

datasets = ["r1_empty", "r1_walking_1"]

antenna_pairs = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3)
]

# =========================
# TICK HELPERS (same as yours)
# =========================
def set_wrapped_pi_ticks(ax):
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels([
        r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"
    ])

def set_2pi_ticks(ax, data):
    y_min = np.nanmin(data)
    y_max = np.nanmax(data)

    k_min = int(np.floor(y_min / (2*np.pi)))
    k_max = int(np.ceil(y_max / (2*np.pi)))

    ticks = [k * 2*np.pi for k in range(k_min, k_max + 1)]

    labels = []
    for k in range(k_min, k_max + 1):
        if k == 0:
            labels.append("0")
        elif k == 1:
            labels.append(r"$2\pi$")
        elif k == -1:
            labels.append(r"$-2\pi$")
        else:
            labels.append(rf"${k}\cdot 2\pi$")

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)

# =========================
# MAIN PROCESS (PURE VERSION)
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

        for start in range(0, packets - W):

            window_h1 = h1[start:start + W]
            window_h2 = h2[start:start + W]

            max_mean_distance = -np.inf
            best_theta = 0.0

            for theta in theta_vals:

                w2 = np.exp(1j * theta)
                ybf = window_h1 + w2 * window_h2

                diff = ybf[1:] - ybf[:-1]
                d = np.sqrt(np.sum(np.abs(diff) ** 2, axis=1))
                mean_d = np.mean(d)

                if mean_d > max_mean_distance:
                    max_mean_distance = mean_d
                    best_theta = theta

            best_distances.append(max_mean_distance)
            best_thetas.append(best_theta)

        best_distances = np.array(best_distances)
        best_thetas = np.array(best_thetas)

        best_thetas_wrapped = (best_thetas + np.pi) % (2*np.pi) - np.pi
        best_thetas_unwrapped = np.unwrap(best_thetas_wrapped)

        # =========================
        # SAVE
        # =========================
        np.save(os.path.join(output_dir, f"{name}_ant{ant1}{ant2}_dist.npy"), best_distances)
        np.save(os.path.join(output_dir, f"{name}_ant{ant1}{ant2}_theta_wrapped.npy"), best_thetas_wrapped)
        np.save(os.path.join(output_dir, f"{name}_ant{ant1}{ant2}_theta_unwrapped.npy"), best_thetas_unwrapped)

        # =========================
        # DISTANCE PLOT
        # =========================
        plt.figure(figsize=(8, 4))
        plt.plot(best_distances)
        plt.title(f"{name} – CSI Distance (Ant {ant1}-{ant2})")
        plt.xlabel("Window index")
        plt.ylabel("Mean CSI distance")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # =========================
        # THETA PLOTS
        # =========================
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        fig.suptitle(f"{name} – Optimal Beam θ (Ant {ant1}-{ant2})", fontsize=13)

        # Wrapped
        axs[0].plot(best_thetas_wrapped)
        axs[0].set_title("Wrapped θ ∈ [-π, π]")
        axs[0].set_ylabel("θ")
        set_wrapped_pi_ticks(axs[0])
        axs[0].grid(True)

        # Unwrapped
        axs[1].plot(best_thetas_unwrapped)
        axs[1].set_title("Unwrapped θ (continuous)")
        axs[1].set_xlabel("Window index")
        axs[1].set_ylabel("θ")
        set_2pi_ticks(axs[1], best_thetas_unwrapped)
        axs[1].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

print("\nDone.")
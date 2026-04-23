import numpy as np
import matplotlib.pyplot as plt
import glob
import os

input_dir = "saved_csi/Reconstructed"
output_dir = "saved_csi/csi_distance"

os.makedirs(output_dir, exist_ok=True)

W = 16
theta_vals = np.linspace(0, 2*np.pi, 180)

files = glob.glob(os.path.join(input_dir, "*.npy"))

for file_path in files:

    name = os.path.basename(file_path).replace(".npy","")
    print("Processing:", name)

    csi = np.load(file_path)

    h1 = csi[0,:,:]
    h2 = csi[1,:,:]

    packets = h1.shape[0]

    best_distances = []
    best_thetas = []

    for start in range(0, packets-W):

        window_h1 = h1[start:start+W]
        window_h2 = h2[start:start+W]

        max_mean_distance = 0
        best_theta = 0

        for theta in theta_vals:

            w1 = 1
            w2 = np.exp(1j*theta)

            ybf = w1*window_h1 + w2*window_h2

            diff = ybf[1:] - ybf[:-1]

            d = np.sqrt(np.sum(np.abs(diff)**2, axis=1))

            mean_d = np.mean(d)

            if mean_d > max_mean_distance:
                max_mean_distance = mean_d
                best_theta = theta

        best_distances.append(max_mean_distance)
        best_thetas.append(best_theta)

    best_distances = np.array(best_distances)
    best_thetas = np.array(best_thetas)

    save_path = os.path.join(output_dir, f"{name}_csi_distance.npy")
    np.save(save_path, best_distances)

    # ---- Plot CSI distance ----
    plt.figure(figsize=(8, 4))
    plt.plot(best_distances)
    plt.title(f"CSI distance – {name}")
    plt.xlabel("Window index")
    plt.ylabel("Mean CSI distance")
    plt.grid(True)
    plt.show()

    # ---- Plot best beam angle ----
    plt.figure(figsize=(8, 4))
    plt.plot(best_thetas)
    plt.title(f"Optimal Beam Angle θ – {name}")
    plt.xlabel("Window index")
    plt.ylabel("θ (radians)")
    plt.grid(True)
    plt.show()
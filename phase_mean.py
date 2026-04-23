# Re-import libraries and re-load files after code environment reset
import numpy as np
import matplotlib.pyplot as plt

# Reload CSI files
csi_empty = np.load('dataset/saved_csi_raw/r1_empty.npy', allow_pickle=True)
csi_walk = np.load('dataset/saved_csi_raw/r1_walking_1.npy', allow_pickle=True)

# Function to compute mean phase per antenna across packets
def compute_mean_phase(csi_data):
    phases = np.angle(csi_data)  # Extract raw phase
    mean_phase = np.mean(phases, axis=1)  # Average over packets
    return mean_phase

# Compute mean phase
mean_phase_empty = compute_mean_phase(csi_empty)
mean_phase_walk = compute_mean_phase(csi_walk)

# Plot comparison
fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
for i in range(4):
    ax = axs[i // 2, i % 2]
    ax.plot(mean_phase_empty[i], label="Empty Room", linestyle="--")
    ax.plot(mean_phase_walk[i], label="Walking", linestyle="-")
    ax.set_title(f"Antenna {i}")
    ax.set_ylabel("Mean Phase (radians)")
    ax.set_xlabel("Subcarrier Index")
    ax.grid(True)
    ax.legend()

plt.suptitle("Mean CSI Phase per Subcarrier: Empty Room vs Walking Human")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

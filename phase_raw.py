import numpy as np
import matplotlib.pyplot as plt

# Load CSI
loaded_csi = np.load('dataset/saved_csi_raw/r1_empty.npy')  # Adjust path if needed

# loaded_csi = np.load('saved_csi/r1_walking_1.npy')  # Adjust path if needed

# Check shape
print("CSI shape:", loaded_csi.shape)  # (4, 4962, 234)

# Pick one packet (e.g., packet index 1000) from each antenna
packet_idx = 1000

plt.figure(figsize=(12, 6))
for ant in range(loaded_csi.shape[0]):
    # Extract complex CSI vector for this antenna and packet
    csi_packet = loaded_csi[ant, packet_idx, :]

    # Compute phase and unwrap it for smoothness
    phase = np.unwrap(np.angle(csi_packet))

    plt.plot(phase, label=f"Antenna {ant}")

plt.xlabel("Subcarrier Index")
plt.ylabel("Phase (radians)")
plt.title(f"CSI Phase In an Empty Room Across Subcarriers (Packet {packet_idx})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

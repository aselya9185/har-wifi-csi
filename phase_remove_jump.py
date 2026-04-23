# # Remove subband stitching and plot corrected unwrapped phase
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Load datasets
# csi_empty = np.load('saved_csi_raw/r1_empty.npy', allow_pickle=True)
# csi_walk = np.load('saved_csi_raw/r1_walking_1.npy', allow_pickle=True)
#
# # Known approximate subband boundaries for 80 MHz (empirically verified)
# subc = csi_empty.shape[2]
# boundaries = [55, 116, 176]  # approximate 20 MHz stitching points
#
# def remove_stitching(csi, boundaries, num_packets=400):
#     antennas, packets, subc = csi.shape
#     idxs = np.linspace(0, packets-1, min(num_packets, packets), dtype=int)
#     csi_corr = csi.copy().astype(np.complex128)
#
#     # Compute reference phase from first subband
#     ref_phase = np.angle(csi[:, idxs, :boundaries[0]]).mean()
#
#     for i, b in enumerate(boundaries):
#         start = b
#         end = boundaries[i+1] if i+1 < len(boundaries) else subc
#         sub_phase = np.angle(csi[:, idxs, start:end]).mean()
#         delta = sub_phase - ref_phase
#         csi_corr[:, :, start:end] *= np.exp(-1j * delta)
#
#     return csi_corr
#
# # Apply correction
# csi_empty_corr = remove_stitching(csi_empty, boundaries)
# csi_walk_corr = remove_stitching(csi_walk, boundaries)
#
# # Compute averaged unwrapped phase per antenna
# def mean_unwrapped_phase(csi, num_packets=400):
#     antennas, packets, subc = csi.shape
#     idxs = np.linspace(0, packets-1, min(num_packets, packets), dtype=int)
#     out = []
#     for ant in range(antennas):
#         phases = [np.unwrap(np.angle(csi[ant, p, :])) for p in idxs]
#         out.append(np.mean(phases, axis=0))
#     return np.array(out)
#
# mean_empty_corr = mean_unwrapped_phase(csi_empty_corr)
# mean_walk_corr = mean_unwrapped_phase(csi_walk_corr)
#
# # Plot corrected results
# plt.figure(figsize=(12,4))
# plt.subplot(1,2,1)
# for ant in range(mean_empty_corr.shape[0]):
#     plt.plot(mean_empty_corr[ant])
# plt.title("After Stitching Removal (Empty Room)")
# plt.xlabel("Subcarrier Index")
# plt.ylabel("Phase (rad)")
# plt.grid(True)
#
# plt.subplot(1,2,2)
# for ant in range(mean_walk_corr.shape[0]):
#     plt.plot(mean_walk_corr[ant])
# plt.title("After Stitching Removal (Walking)")
# plt.xlabel("Subcarrier Index")
# plt.ylabel("Phase (rad)")
# plt.grid(True)
#
# plt.tight_layout()
# plt.show()

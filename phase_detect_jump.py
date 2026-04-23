# # Analyze CSI datasets to detect subband stitching boundaries typical of 80MHz = 4x20MHz bonding.
# # We'll check both empty and walking datasets for consistent large phase jumps at ~1/4, 2/4, 3/4 of 234 subcarriers.
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
#
# data_dir = Path('/mnt/data')
# empty_path = 'saved_csi/r1_empty.npy'
# walk_path = 'saved_csi/r1_walking_1.npy'
#
# csi_empty = np.load(empty_path, allow_pickle=True)
# csi_walk = np.load(walk_path, allow_pickle=True)
#
# print("Shapes: empty", csi_empty.shape, "walk", csi_walk.shape)
#
# def detect_large_jumps(csi, num_packets=500, top_k=3):
#     # csi shape (antennas, packets, subcarriers)
#     antennas, packets, subc = csi.shape
#     # sample uniformly across packets if too many
#     if num_packets is None or num_packets > packets:
#         num_packets = packets
#     idxs = np.linspace(0, packets-1, num_packets, dtype=int)
#     top_jump_indices = []
#     top_jump_values = []
#     for p in idxs:
#         phases = np.angle(csi[:, p, :])          # (antennas, subc)
#         mean_phase = phases.mean(axis=0)         # (subc,)
#         # unwrap to reveal larger jumps
#         unp = np.unwrap(mean_phase)
#         dif = np.diff(unp)
#         # find indices of largest abs diffs
#         abs_dif = np.abs(dif)
#         topk = np.argsort(abs_dif)[-top_k:][::-1]
#         top_jump_indices.append(topk)
#         top_jump_values.append(dif[topk])
#     return np.array(top_jump_indices), np.array(top_jump_values)
#
# # Detect jumps for both datasets
# idxs_empty, vals_empty = detect_large_jumps(csi_empty, num_packets=800, top_k=4)
# idxs_walk, vals_walk = detect_large_jumps(csi_walk, num_packets=800, top_k=4)
#
# # Flatten and compute histograms
# flat_idxs_empty = idxs_empty.flatten()
# flat_idxs_walk = idxs_walk.flatten()
#
# # Plot histogram of detected jump indices
# plt.figure(figsize=(10,3))
# plt.hist(flat_idxs_empty, bins=np.arange(0, csi_empty.shape[2]+1)-0.5, alpha=0.6, label='empty')
# plt.hist(flat_idxs_walk, bins=np.arange(0, csi_walk.shape[2]+1)-0.5, alpha=0.6, label='walk')
# plt.title("Histogram of top detected phase-jump indices (both datasets)")
# plt.xlabel("Subcarrier index")
# plt.ylabel("Count (packets * top_k)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# # Compute mean jump index and show most frequent
# from collections import Counter
# cnt_empty = Counter(flat_idxs_empty)
# cnt_walk = Counter(flat_idxs_walk)
# most_common_empty = cnt_empty.most_common(10)
# most_common_walk = cnt_walk.most_common(10)
# print("Most common jump indices (empty):", most_common_empty[:10])
# print("Most common jump indices (walk):", most_common_walk[:10])
#
# # Plot a representative packet's mean phase and mark quarter indices (approx)
# rep_p = 1000  # representative packet index
# phases_rep = np.unwrap(np.angle(csi_empty[:, rep_p, :]).mean(axis=0))
# plt.figure(figsize=(10,3))
# plt.plot(phases_rep, label='mean phase (empty) packet {}'.format(rep_p))
# # mark most common boundaries observed
# quarter = csi_empty.shape[2]/4.0
# for q in range(1,4):
#     plt.axvline(int(round(q*quarter)), color='k', linestyle='--', alpha=0.7)
# plt.title("Representative mean phase and quarter markers")
# plt.xlabel("Subcarrier index")
# plt.ylabel("Phase (rad)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# # Show the mean of mean phases across many packets to reduce noise and mark peaks
# mean_of_means_empty = np.unwrap(np.angle(csi_empty).mean(axis=0).mean(axis=0))
# plt.figure(figsize=(10,3))
# plt.plot(mean_of_means_empty, label='mean of mean phases (empty)')
# for q in range(1,4):
#     plt.axvline(int(round(q*quarter)), color='k', linestyle='--', alpha=0.7)
# plt.title("Averaged mean phase across packets (empty) with quarter markers")
# plt.xlabel("Subcarrier index")
# plt.ylabel("Phase (rad)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# # Print approximate quarter indices
# print("Total subcarriers:", csi_empty.shape[2], "quarter indices approx:", [round(q*quarter) for q in range(1,4)])
#
# # Save summary
# summary = {
#     'most_common_empty': most_common_empty[:10],
#     'most_common_walk': most_common_walk[:10],
#     'quarter_indices': [int(round(q*quarter)) for q in range(1,4)],
#     'shape': csi_empty.shape
# }
# import json
# with open('/mnt/data/subband_jump_summary.json','w') as f:
#     json.dump(summary, f)
# print("Summary saved to /mnt/data/subband_jump_summary.json")
#
#
# # Retry analysis: detect subband stitching boundaries in empty and walk datasets.
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# from collections import Counter
# data_dir = Path('/mnt/data')
# empty_path = data_dir / 'r1_empty.npy'
# walk_path = data_dir / 'r1_walking_1.npy'
#
# csi_empty = np.load(empty_path, allow_pickle=True)
# csi_walk = np.load(walk_path, allow_pickle=True)
# print("Shapes: empty", csi_empty.shape, "walk", csi_walk.shape)
#
# def detect_large_jumps(csi, num_packets=400, top_k=3):
#     antennas, packets, subc = csi.shape
#     if num_packets > packets:
#         num_packets = packets
#     idxs = np.linspace(0, packets-1, num_packets, dtype=int)
#     top_jump_indices = []
#     top_jump_values = []
#     for p in idxs:
#         phases = np.angle(csi[:, p, :])          # (antennas, subc)
#         mean_phase = phases.mean(axis=0)         # (subc,)
#         unp = np.unwrap(mean_phase)
#         dif = np.diff(unp)
#         abs_dif = np.abs(dif)
#         topk = np.argsort(abs_dif)[-top_k:][::-1]
#         top_jump_indices.append(list(topk))
#         top_jump_values.append(list(dif[topk]))
#     return np.array(top_jump_indices), np.array(top_jump_values)
#
# idxs_empty, vals_empty = detect_large_jumps(csi_empty, num_packets=400, top_k=4)
# idxs_walk, vals_walk = detect_large_jumps(csi_walk, num_packets=400, top_k=4)
#
# flat_idxs_empty = idxs_empty.flatten()
# flat_idxs_walk = idxs_walk.flatten()
#
# plt.figure(figsize=(10,3))
# plt.hist(flat_idxs_empty, bins=np.arange(0, csi_empty.shape[2]+1)-0.5, alpha=0.6, label='empty')
# plt.hist(flat_idxs_walk, bins=np.arange(0, csi_walk.shape[2]+1)-0.5, alpha=0.6, label='walk')
# plt.title("Histogram of Top Detected Phase-Jump Indices (empty vs walk)")
# plt.xlabel("Subcarrier index")
# plt.ylabel("Count")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# most_common_empty = Counter(flat_idxs_empty).most_common(10)
# most_common_walk = Counter(flat_idxs_walk).most_common(10)
# print("Most common jump indices (empty):", most_common_empty[:10])
# print("Most common jump indices (walk):", most_common_walk[:10])
#
# # Representative packet plots
# rep_p = 100
# phases_rep = np.unwrap(np.angle(csi_empty[:, rep_p, :]).mean(axis=0))
# subc = csi_empty.shape[2]
# quarter = subc / 4.0
#
# plt.figure(figsize=(10,3))
# plt.plot(phases_rep, label='mean phase (empty) packet {}'.format(rep_p))
# for q in range(1,4):
#     plt.axvline(int(round(q*quarter)), color='k', linestyle='--', alpha=0.7)
# plt.title("Representative mean phase and quarter markers (empty)")
# plt.xlabel("Subcarrier index")
# plt.ylabel("Phase (rad)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# # Average across packets to highlight persistent structure
# mean_of_means_empty = np.unwrap(np.angle(csi_empty).mean(axis=0).mean(axis=0))
# plt.figure(figsize=(10,3))
# plt.plot(mean_of_means_empty, label='averaged mean phase (empty)')
# for q in range(1,4):
#     plt.axvline(int(round(q*quarter)), color='k', linestyle='--', alpha=0.7)
# plt.title("Averaged mean phase across packets (empty) with quarter markers")
# plt.xlabel("Subcarrier index")
# plt.ylabel("Phase (rad)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# print("Total subcarriers:", subc, "quarter indices approx:", [int(round(q*quarter)) for q in range(1,4)])
#
# # Save summary
# summary = {
#     'most_common_empty': most_common_empty[:10],
#     'most_common_walk': most_common_walk[:10],
#     'quarter_indices': [int(round(q*quarter)) for q in range(1,4)],
#     'shape': csi_empty.shape
# }
# import json
# with open('/mnt/data/subband_jump_summary.json','w') as f:
#     json.dump(summary, f)
# print("Summary saved to /mnt/data/subband_jump_summary.json")




##################################################################################

# # Visualizing persistent phase-jump structure across 400 packets
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Load datasets
# csi_empty = np.load('saved_csi/r1_empty.npy', allow_pickle=True)
# csi_walk = np.load('saved_csi/r1_walking_1.npy', allow_pickle=True)
#
#
# def average_phase_jump_profile(csi, num_packets=400):
#     antennas, packets, subc = csi.shape
#     if num_packets > packets:
#         num_packets = packets
#     idxs = np.linspace(0, packets - 1, num_packets, dtype=int)
#
#     jump_profiles = []
#     for p in idxs:
#         phases = np.angle(csi[:, p, :])
#         mean_phase = phases.mean(axis=0)
#         unp = np.unwrap(mean_phase)
#         dif = np.abs(np.diff(unp))
#         jump_profiles.append(dif)
#
#     jump_profiles = np.array(jump_profiles)
#     mean_jump = jump_profiles.mean(axis=0)
#     return mean_jump
#
#
# mean_jump_empty = average_phase_jump_profile(csi_empty, 400)
# mean_jump_walk = average_phase_jump_profile(csi_walk, 400)
#
# # Plot 1: Empty room
# plt.figure()
# plt.plot(mean_jump_empty)
# plt.title("Average Adjacent Phase Difference Across 400 Packets (Empty Room)")
# plt.xlabel("Subcarrier Index")
# plt.ylabel("Mean |Δ Phase| (rad)")
# plt.grid(True)
# plt.show()
#
# # Plot 2: Walking room
# plt.figure()
# plt.plot(mean_jump_walk)
# plt.title("Average Adjacent Phase Difference Across 400 Packets (Walking)")
# plt.xlabel("Subcarrier Index")
# plt.ylabel("Mean |Δ Phase| (rad)")
# plt.grid(True)
# plt.show()


# Plot unwrapped mean phase across 400 packets per antenna
import numpy as np
import matplotlib.pyplot as plt

# Load datasets
csi_empty = np.load('dataset/saved_csi_raw/r1_empty.npy', allow_pickle=True)
csi_walk = np.load('dataset/saved_csi_raw/r1_walking_1.npy', allow_pickle=True)


def mean_unwrapped_phase(csi, num_packets=400):
    antennas, packets, subc = csi.shape
    if num_packets > packets:
        num_packets = packets
    idxs = np.linspace(0, packets - 1, num_packets, dtype=int)

    # Average phase over selected packets for each antenna
    mean_phases = []
    for ant in range(antennas):
        phases = []
        for p in idxs:
            ph = np.unwrap(np.angle(csi[ant, p, :]))
            phases.append(ph)
        phases = np.array(phases)
        mean_phases.append(phases.mean(axis=0))
    return np.array(mean_phases)


mean_phase_empty = mean_unwrapped_phase(csi_empty, 400)
mean_phase_walk = mean_unwrapped_phase(csi_walk, 400)

# Plot 1: Empty room
plt.figure()
for ant in range(mean_phase_empty.shape[0]):
    plt.plot(mean_phase_empty[ant])
plt.title("Unwrapped Mean Phase Across 400 Packets (Empty Room)")
plt.xlabel("Subcarrier Index")
plt.ylabel("Phase (rad)")
plt.grid(True)
plt.show()

# Plot 2: Walking room
plt.figure()
for ant in range(mean_phase_walk.shape[0]):
    plt.plot(mean_phase_walk[ant])
plt.title("Unwrapped Mean Phase Across 400 Packets (Walking)")
plt.xlabel("Subcarrier Index")
plt.ylabel("Phase (rad)")
plt.grid(True)
plt.show()

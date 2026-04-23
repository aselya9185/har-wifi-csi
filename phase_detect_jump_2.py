# Recompute counts near quarter indices for clearer proof
import numpy as np
from collections import Counter
from pathlib import Path

csi_empty = np.load('dataset/saved_csi_raw/r1_empty.npy', allow_pickle=True)
csi_walk = np.load('dataset/saved_csi_raw/r1_walking_1.npy', allow_pickle=True)

def detect_topk_jumps_flat(csi, num_packets=400, top_k=4):
    antennas, packets, subc = csi.shape
    if num_packets > packets:
        num_packets = packets
    idxs = np.linspace(0, packets-1, num_packets, dtype=int)
    flat = []
    for p in idxs:
        phases = np.angle(csi[:, p, :])
        mean_phase = phases.mean(axis=0)
        unp = np.unwrap(mean_phase)
        dif = np.diff(unp)
        abs_dif = np.abs(dif)
        topk = np.argsort(abs_dif)[-top_k:][::-1]
        flat.extend(list(topk))
    return flat

flat_empty = detect_topk_jumps_flat(csi_empty, num_packets=400, top_k=4)
flat_walk = detect_topk_jumps_flat(csi_walk, num_packets=400, top_k=4)

subc = csi_empty.shape[2]
quarter = subc / 4.0
quarters = [int(round(q*quarter)) for q in range(1,4)]
print("Quarter indices:", quarters)

def count_near(flat_idxs, centers, tol=3):
    cnts = {}
    for c in centers:
        cnts[c] = sum(1 for x in flat_idxs if abs(x - c) <= tol)
    return cnts

cnts_empty = count_near(flat_empty, quarters, tol=3)
cnts_walk = count_near(flat_walk, quarters, tol=3)

print("Counts near quarter indices (±3): empty:", cnts_empty)
print("Counts near quarter indices (±3): walk:", cnts_walk)

# Also show overall top 10 most common indices
print("Top 10 most common indices empty:", Counter(flat_empty).most_common(25))
print("Top 10 most common indices walk:", Counter(flat_walk).most_common(25))

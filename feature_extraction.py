import numpy as np
import os
from collections import defaultdict

# =========================
# DIRECTORIES
# =========================
dist_dir = "dataset/csi_distance"
theta_dir = "dataset/theta"
output_root = "dataset/features"

os.makedirs(output_root, exist_ok=True)

# =========================
# PARAMETERS
# =========================
W2_values = [10, 20, 30, 40, 50]

antenna_pairs = [
    "ant01", "ant02", "ant03",
    "ant12", "ant13", "ant23"
]

labels_map = {
    "empty": 0,
    "sit": 1,
    "sitting": 1,
    "standing": 2,
    "walk": 3,
    "walking": 3
}

# =========================
# HELPERS
# =========================
def get_label(name):
    for key in labels_map:
        if key in name:
            return labels_map[key]
    raise ValueError(f"Unknown label in {name}")


def get_room(name):
    if name.startswith("r1"):
        return "r1"
    elif name.startswith("r2"):
        return "r2"
    else:
        return "unknown"

# =========================
# SAVE
# =========================
def save_dataset(data, base_out, subfolder, filename):
    if len(data) == 0:
        return

    os.makedirs(os.path.join(base_out, subfolder), exist_ok=True)

    arr = np.array(data)
    np.save(
        os.path.join(base_out, subfolder, filename),
        arr
    )

# =========================
# LOAD FILE INDEX
# =========================
files = sorted([f for f in os.listdir(dist_dir) if f.endswith("_dist.npy")])

# group files by antenna pair
pair_files = defaultdict(list)

for f in files:
    for pair in antenna_pairs:
        if pair in f:
            pair_files[pair].append(f)

# =========================
# MAIN FEATURE EXTRACTION
# =========================
for W2 in W2_values:

    print(f"\nProcessing W2 = {W2}")

    base_out = os.path.join(output_root, f"w2_{W2}")

    for pair in antenna_pairs:

        print(f"  Pair: {pair}")

        # containers per split
        data_r1 = []
        data_r2 = []
        data_all = []

        for dist_file in pair_files[pair]:

            name = dist_file.replace(f"_{pair}_dist.npy", "")

            theta_file = f"{name}_{pair}_theta.npy"

            dist_path = os.path.join(dist_dir, dist_file)
            theta_path = os.path.join(theta_dir, theta_file)

            if not os.path.exists(theta_path):
                continue

            dist = np.load(dist_path)
            theta = np.load(theta_path)

            # unwrap for stability
            theta = np.unwrap(theta)

            label = get_label(name)
            room = get_room(name)

            T = len(dist)

            for start in range(0, T - W2):

                d_win = dist[start:start + W2]
                t_win = theta[start:start + W2]

                features = [
                    np.mean(d_win), # mean distance
                    np.std(d_win),  # variability
                    np.max(d_win),  # peaks
                    np.var(t_win),  # beam stability
                ]

                sample = features + [label]

                # store per split
                if room == "r1":
                    data_r1.append(sample)
                elif room == "r2":
                    data_r2.append(sample)

                data_all.append(sample)

        save_dataset(data_r1, base_out, "r1", f"r1_{pair}.npy")
        save_dataset(data_r2, base_out, "r2", f"r2_{pair}.npy")
        save_dataset(data_all, base_out, "r1_r2", f"r1_r2_{pair}.npy")

print("\nAll feature datasets saved.")

# =========================
# DEBUG: CHECK ONE DATASET
# =========================
sample_path = os.path.join(
    output_root,
    "w2_20",        # pick any W2
    "r1_r2",        # pick split
    "r1_r2_ant01.npy"  # pick antenna
)

if os.path.exists(sample_path):
    data = np.load(sample_path)

    print("\n=== SAMPLE DATASET ===")
    print("Path:", sample_path)
    print("Shape:", data.shape)
    print("\nFirst 5 rows:")
    print(data[:5])
else:
    print("\nSample file not found:", sample_path)

# =========================
# DEBUG: WALKING SAMPLES
# =========================
if os.path.exists(sample_path):
    data = np.load(sample_path)

    labels = data[:, -1].astype(int)

    walking_data = data[labels == 3]

    print("\n=== WALKING SAMPLES (label = 3) ===")
    print("Total walking samples:", walking_data.shape[0])

    print("\nFirst 5 walking rows:")
    print(walking_data[:5])
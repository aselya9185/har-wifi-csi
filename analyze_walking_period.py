import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from scipy.signal import find_peaks


# =========================
# DIRECTORIES
# =========================
input_dir = "dataset/csi_distance"

plot_root = "plots/walking_peak_analysis"
csv_root = "plots/walking_peak_analysis/csv"

os.makedirs(plot_root, exist_ok=True)
os.makedirs(csv_root, exist_ok=True)

target_ant = "ant01"

# estimated effective sampling frequency
# (window index updates every packet/window shift)
fs = 5.0  # Hz

# smoothing window
smooth_window = 5

# peak detection parameters
peak_prominence = 0.6e13
peak_distance = 15  # minimum samples between peaks (to avoid detecting multiple peaks in one step)
height = 2e13

# =========================
# HELPERS
# =========================
def moving_average(x, w=5):
    return np.convolve(x, np.ones(w) / w, mode='same')


def load_walking_files():
    files = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith("_dist.npy")
        and target_ant in f
        and ("walk" in f or "walking" in f)
    ])
    return files


def detect_peaks(signal):
    peaks, properties = find_peaks(
        signal,
        prominence=peak_prominence,
        distance=peak_distance,
        height=height
    )
    return peaks, properties


# =========================
# MAIN ANALYSIS
# =========================
summary_results = []

files = load_walking_files()

print("\nWalking datasets found:")
for f in files:
    print(f)

for filename in files:

    print(f"\nProcessing: {filename}")

    path = os.path.join(input_dir, filename)

    signal = np.load(path)

    smooth_signal = moving_average(signal, smooth_window)

    # DETECT PEAKS
    peaks, properties = detect_peaks(smooth_signal)

    peak_values = smooth_signal[peaks]

    # PEAK INTERVALS
    delta_n = np.diff(peaks)

    delta_t = delta_n / fs

    # METRICS
    mean_delta_n = np.mean(delta_n) if len(delta_n) > 0 else np.nan
    std_delta_n = np.std(delta_n) if len(delta_n) > 0 else np.nan

    mean_period = np.mean(delta_t) if len(delta_t) > 0 else np.nan
    std_period = np.std(delta_t) if len(delta_t) > 0 else np.nan

    regularity_cv = (
        std_delta_n / mean_delta_n
        if mean_delta_n > 0 else np.nan
    )

    n_peaks = len(peaks)

    # SAVE SUMMARY
    summary_results.append([
        filename,
        n_peaks,
        mean_delta_n,
        std_delta_n,
        mean_period,
        std_period,
        regularity_cv
    ])

    # CREATE OUTPUT FOLDER
    dataset_name = filename.replace(".npy", "")

    dataset_plot_dir = os.path.join(plot_root, dataset_name)

    os.makedirs(dataset_plot_dir, exist_ok=True)

    # ==========================================================
    # PLOT 1: SIGNAL + DETECTED PEAKS
    # ==========================================================
    plt.figure(figsize=(14, 6), dpi=150)

    plt.plot(signal, alpha=0.4, label="Raw signal")

    plt.plot(
        smooth_signal,
        linewidth=2,
        label="Smoothed signal"
    )

    plt.plot(
        peaks,
        peak_values,
        "ro",
        markersize=5,
        label="Detected peaks"
    )

    plt.title(f"{dataset_name} - CSI Distance Peaks")
    plt.xlabel("Window index")
    plt.ylabel("Mean CSI distance")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(dataset_plot_dir, "signal_with_peaks.png")
    )

    plt.close()

    # ==========================================================
    # PLOT 2: INTERVALS Δnk
    # ==========================================================
    plt.figure(figsize=(12, 5), dpi=150)

    plt.plot(delta_n, marker='o')

    plt.title(f"{dataset_name} - Peak Intervals Δn")
    plt.xlabel("Peak index")
    plt.ylabel("Δn (samples)")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(dataset_plot_dir, "peak_intervals.png")
    )

    plt.close()

    # ==========================================================
    # PLOT 3: INTERVALS IN SECONDS
    # ==========================================================
    plt.figure(figsize=(12, 5), dpi=150)

    plt.plot(delta_t, marker='o')

    plt.title(f"{dataset_name} - Peak-to-Peak Motion Period")
    plt.xlabel("Peak index")
    plt.ylabel("Period (seconds)")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(dataset_plot_dir, "motion_period_seconds.png")
    )

    plt.close()

    # ==========================================================
    # PLOT 4: HISTOGRAM OF Δn
    # ==========================================================
    plt.figure(figsize=(10, 5), dpi=150)

    plt.hist(delta_n, bins=15)

    plt.title(f"{dataset_name} - Histogram of Δn")
    plt.xlabel("Δn (samples)")
    plt.ylabel("Count")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(dataset_plot_dir, "histogram_delta_n.png")
    )

    plt.close()

    # ==========================================================
    # PLOT 5: HISTOGRAM OF PERIODS
    # ==========================================================
    plt.figure(figsize=(10, 5), dpi=150)

    plt.hist(delta_t, bins=15)

    plt.title(f"{dataset_name} - Histogram of Periods")
    plt.xlabel("Period (seconds)")
    plt.ylabel("Count")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(dataset_plot_dir, "histogram_periods.png")
    )

    plt.close()

    # ==========================================================
    # SAVE PEAKS CSV
    # ==========================================================
    peaks_df = pd.DataFrame({
        "peak_index": peaks,
        "peak_value": peak_values
    })

    peaks_df.to_csv(
        os.path.join(dataset_plot_dir, "detected_peaks.csv"),
        index=False
    )

    # ==========================================================
    # SAVE INTERVALS CSV
    # ==========================================================
    intervals_df = pd.DataFrame({
        "delta_n_samples": delta_n,
        "delta_t_seconds": delta_t
    })

    intervals_df.to_csv(
        os.path.join(dataset_plot_dir, "peak_intervals.csv"),
        index=False
    )

    print(f"Detected peaks: {n_peaks}")
    print(f"Mean Δn: {mean_delta_n:.2f}")
    print(f"Mean period: {mean_period:.2f} s")
    print(f"Regularity CV: {regularity_cv:.3f}")


# =========================
# FINAL SUMMARY TABLE
# =========================
summary_df = pd.DataFrame(summary_results, columns=[
    "dataset",
    "num_peaks",
    "mean_delta_n",
    "std_delta_n",
    "mean_period_seconds",
    "std_period_seconds",
    "regularity_cv"
])

summary_csv_path = os.path.join(
    csv_root,
    "walking_peak_summary.csv"
)

summary_df.to_csv(summary_csv_path, index=False)

print("\n===================================")
print("Walking peak analysis completed.")
print("===================================")

print("\nSummary:")
print(summary_df)
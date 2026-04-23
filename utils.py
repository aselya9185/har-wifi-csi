import numpy as np
import matplotlib.ticker as ticker

# =========================
# INDEX MAPPING
# =========================

NON_RECONSTRUCT = [0,1,2,3,4,5,128,251,252,253,254,255]

def get_target_indices():
    mask = np.ones(256, dtype=bool)
    mask[NON_RECONSTRUCT] = False
    return np.where(mask)[0]


# =========================
# AXIS
# =========================

def k_axis():
    return np.arange(-128, 128)


# =========================
# RECONSTRUCTION
# =========================

def build_full(x, indices):
    A, T, _ = x.shape
    full = np.full((A, T, 256), np.nan)
    full[:, :, indices] = x
    return full


# =========================
# UNWRAP (NaN SAFE)
# =========================

def unwrap_with_nan_full(x):
    out = x.copy()
    for a in range(x.shape[0]):
        for t in range(x.shape[1]):
            valid = ~np.isnan(x[a, t])
            if np.sum(valid) > 1:
                out[a, t, valid] = np.unwrap(x[a, t, valid])
    return out


def unwrap_with_nan_single(x):
    out = x.copy()
    for a in range(x.shape[0]):
        valid = ~np.isnan(x[a])
        if np.sum(valid) > 1:
            out[a, valid] = np.unwrap(x[a, valid])
    return out


# =========================
# PLOTTING
# =========================

def plot_dataset(ax, data):
    k = k_axis()
    for a in range(data.shape[0]):
        ax.plot(k, np.ma.masked_invalid(data[a]), label=f"Antenna {a}", linewidth=1)

    ax.grid(True)
    ax.legend(fontsize=8)


def set_pi_ticks(ax):
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2*np.pi))
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda val, pos: f"{val/np.pi:.0g}π" if val != 0 else "0")
    )
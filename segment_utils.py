import numpy as np

# correct split points for 256 grid
SPLIT_POINTS = [64, 128, 192]

def split_phase_data(phase_data):
    """
    phase_data: (A, T, 256)
    returns: list of 4 segments
    """
    segments = []

    start = 0
    for sp in SPLIT_POINTS:
        segments.append(phase_data[:, :, start:sp])
        start = sp

    segments.append(phase_data[:, :, start:])

    return segments


def unwrap_phase_dataset(csi):
    """
    unwrap phase along subcarrier axis
    """
    return np.unwrap(np.angle(csi), axis=2)
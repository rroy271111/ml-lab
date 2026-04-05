import numpy as np


def detect_regime(bar_lengths):
    regimes = []
    average = np.mean(bar_lengths)

    for length in bar_lengths:
        if length < average * 0.5:
            regimes.append("high_activity")
        elif length > average * 1.5:
            regimes.append("low_activity")
        else:
            regimes.append("normal")
    return regimes

from financial_ml_tick_imbalance.bars.tick_rule import compute_bt
from financial_ml_tick_imbalance.utils.estimate_expectations import ewma
import numpy as np


def volume_imbalance_bars(prices, volumes, alpha=0.1, init_T=50):
    bt = compute_bt(prices)

    bars = []
    theta = 0
    T = 0

    expectation_T = init_T
    expectation_imbalance = 0

    start = 0

    for i in range(len(bt)):
        theta += bt[i] * volumes[i]
        T += 1

        threshold = expectation_T * abs(expectation_imbalance)

        if threshold == 0:
            threshold = init_T * np.mean(volumes)

        if abs(theta) >= threshold:
            bars.append((start, i))

            expectation_T = ewma(expectation_T, T, alpha)
            expectation_imbalance = ewma(expectation_imbalance, theta / T, alpha)

            start = i + 1
            theta = 0
            T = 0

    return bars

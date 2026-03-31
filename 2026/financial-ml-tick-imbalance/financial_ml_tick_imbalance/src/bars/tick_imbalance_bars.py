def tick_imbalance_bars(prices, alpha=0.1, init_T=50):
    bt = compute_bt(prices)

    bars = []
    theta = 0
    T = 0

    # initial expectations
    expectation_T = init_T
    expectation_bt = 0

    start = 0

    for i in range(len(bt)):
        theta += bt[i]

        T += 1

        threshold = expectation_T * abs(expectation_bt)

        # avoid zero threshold initially
        if threshold == 0:
            threshold = init_T * 0.5

        if abs(theta) >= threshold:
            bars.append((start, i))

            # update expectations
            expectation_T = ewma(expectation_T, T, alpha)
            expectation_bt = ewma(expectation_bt, theta / T, alpha)

            # reset
            start = i + 1
            theta = 0
            T = 0

        return bars

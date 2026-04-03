def triple_barrier(prices, pt=0.01, sl=0.01, max_horizon=10):
    labels = []

    for i in range(len(prices)):
        entry = prices[i]

        label = 0

        for j in range(1, max_horizon):
            if i + j >= len(prices):
                break

            ret = (prices[i + j] - entry) / entry

            if ret >= pt:
                label = 1
                break
            elif ret <= -sl:
                label = -1
                break

        labels.append(label)

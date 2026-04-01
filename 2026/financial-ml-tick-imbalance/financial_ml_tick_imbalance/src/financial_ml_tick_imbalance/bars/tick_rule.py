def compute_bt(prices):
    bt = [0]  # first value placeholder

    for i in range(1, len(prices)):
        delta = prices[i] - prices[i - 1]
        if delta > 0:
            bt.append(1)
        elif delta < 0:
            bt.append(-1)
        else:
            bt.append(bt[-1])  # carry forward

    return bt

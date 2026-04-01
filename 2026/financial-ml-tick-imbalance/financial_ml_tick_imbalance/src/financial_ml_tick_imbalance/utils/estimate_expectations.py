def ewma(prev, new, alpha=0.1):
    return alpha * new + (1 - alpha) * prev

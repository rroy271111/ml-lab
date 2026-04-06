def apply_regime_filter(signals, regimes, active_regime="high_activity"):
    filtered_signals = []

    for signal, regime in zip(signals, regimes):
        if regime == active_regime:
            filtered_signals.append(signal)
        else:
            filtered_signals.append(0)
    return filtered_signals

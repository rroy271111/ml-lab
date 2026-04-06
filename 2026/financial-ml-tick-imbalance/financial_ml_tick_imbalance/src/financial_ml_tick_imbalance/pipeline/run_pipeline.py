from financial_ml_tick_imbalance.models.dataset_builder import (
    build_training_dataset,
)
from financial_ml_tick_imbalance.models.train_test_splitter import (
    temporal_train_test_split,
)
from financial_ml_tick_imbalance.models.primary_classifier import (
    train_primary_classifier,
)
from financial_ml_tick_imbalance.models.evaluation import evaluate_classifier
from financial_ml_tick_imbalance.models.signal_generation import (
    generate_trade_signals,
)
from financial_ml_tick_imbalance.models.signal_filter import apply_regime_filter
from financial_ml_tick_imbalance.models.market_regime import detect_regime


def run_pipeline(features, labels, bar_lengths):
    df = builder_training_dataset(features, labels)

    train_df, test_df = temporal_train_test_split(df)

    model = train_primary_classifier(train_df)

    report = evaluate_classifier(model, test_df)

    signals = generate_trade_signals(model, test_df)

    regimes = detect_regime(bar_lengths)

    filtered_signals = apply_regime_filter(signals, regimes)

    return report, filtered_signals

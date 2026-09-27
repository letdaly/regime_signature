"""Run the corrected path-level Heston regime-classification baseline.

This script deliberately keeps the classifier fixed while comparing feature
representations.  Reusing the same master seed makes every representation see
the same independently generated train, validation, and test paths.
"""

import argparse
from typing import Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from regime_pipeline import DEFAULT_N_PATHS, CausalFeatureExtractor, build_path_level_splits


def evaluate_representation(
    representation: str,
    n_paths: tuple,
    n_steps: int,
    master_seed: int,
    window_size: int,
    signature_kind: str,
) -> Dict[str, float]:
    extractor = CausalFeatureExtractor(
        window_size=window_size,
        short_window=min(10, window_size),
        signature_level=3,
        representation=representation,
        signature_kind=signature_kind,
    )
    splits = build_path_level_splits(
        extractor=extractor,
        n_paths=n_paths,
        n_steps=n_steps,
        master_seed=master_seed,
        min_dwell=window_size,
    )
    splits.assert_disjoint()

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight="balanced",
        random_state=master_seed,
        n_jobs=-1,
    )
    model.fit(splits.train.X, splits.train.y)

    # Validation is reserved for future threshold and smoothing calibration.
    # It is reported separately and never merged into the test partition.
    validation_prediction = model.predict(splits.validation.X)
    test_prediction = model.predict(splits.test.X)
    return {
        "n_features": float(splits.train.X.shape[1]),
        "validation_balanced_accuracy": balanced_accuracy_score(
            splits.validation.y, validation_prediction
        ),
        "test_accuracy": accuracy_score(splits.test.y, test_prediction),
        "test_balanced_accuracy": balanced_accuracy_score(
            splits.test.y, test_prediction
        ),
        "test_macro_f1": f1_score(
            splits.test.y, test_prediction, average="macro", zero_division=0
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-paths", type=int, default=DEFAULT_N_PATHS[0])
    parser.add_argument("--validation-paths", type=int, default=DEFAULT_N_PATHS[1])
    parser.add_argument("--test-paths", type=int, default=DEFAULT_N_PATHS[2])
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--signature-kind", choices=("log", "raw"), default="log")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_paths = (args.train_paths, args.validation_paths, args.test_paths)
    results = {}
    for representation in ("statistics", "signature", "combined"):
        results[representation] = evaluate_representation(
            representation=representation,
            n_paths=n_paths,
            n_steps=args.steps,
            master_seed=args.seed,
            window_size=args.window,
            signature_kind=args.signature_kind,
        )

    columns = (
        "representation",
        "features",
        "validation balanced accuracy",
        "test accuracy",
        "test balanced accuracy",
        "test macro F1",
    )
    print(" | ".join(columns))
    print(" | ".join("---" for _ in columns))
    for representation, metrics in results.items():
        print(
            "{} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f}".format(
                representation,
                int(metrics["n_features"]),
                metrics["validation_balanced_accuracy"],
                metrics["test_accuracy"],
                metrics["test_balanced_accuracy"],
                metrics["test_macro_f1"],
            )
        )


if __name__ == "__main__":
    main()


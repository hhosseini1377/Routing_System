#!/usr/bin/env python3
"""
Train a regression model to estimate TTFT, TBT, or average latency from hyperparameters.

Reads performance data JSON (from collect_performance_data.py), builds features from
setup (thread_percentage, memory_util, load_rps, tensor_parallel_size, and optionally
max_model_len, max_num_seqs, max_num_batched_tokens), and trains either linear
(polynomial) regression or gradient boosting to predict the chosen target (ms) per experiment.

Models:
  linear            - LinearRegression with optional degree-3 polynomial features
  gradient_boosting - HistGradientBoostingRegressor (captures non-linearity and interactions)

Targets:
  ttft   - Time To First Token (mean of ttfts_ms)
  tbt    - Time Between Tokens (mean of all TBT values in tbt_lists_ms)
  latency - Average end-to-end latency (performance.avg_latency_ms)

Usage:
    python -m profiler.train_metric_regression --target ttft --model-type linear --input performance_data.json
    python -m profiler.train_metric_regression --target ttft --model-type gradient_boosting --input performance_data.json --output profiler/ttft_gb_model.joblib
"""

import argparse
import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# Defaults for optional setup fields (old JSON without max_model_len etc.)
DEFAULT_MAX_MODEL_LEN = 2048
DEFAULT_MAX_NUM_SEQS = 0  # 0 = "not set" for linear model
DEFAULT_MAX_NUM_BATCHED_TOKENS = 0
USE_POLYNOMIAL_FEATURES = True  # only used when model_type == "linear"

MODEL_TYPE_CHOICES = ("linear", "gradient_boosting")  


def load_performance_data(path: str) -> list:
    """Load performance data from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def build_feature_vector(setup: dict, use_optional: bool = True) -> np.ndarray:
    """
    Build numeric feature vector from setup dict.

    Features: thread_percentage, memory_util, load_rps, tensor_parallel_size,
    and optionally max_model_len, max_num_seqs, max_num_batched_tokens.
    """
    thread = float(setup["thread_percentage"])
    memory = float(setup["memory_util"])
    load_rps = float(setup["load_rps"])
    tp_size = int(setup["tensor_parallel_size"])

    if use_optional:
        max_model_len = setup.get("max_model_len")
        max_num_seqs = setup.get("max_num_seqs")
        max_num_batched_tokens = setup.get("max_num_batched_tokens")
        if max_model_len is None:
            max_model_len = DEFAULT_MAX_MODEL_LEN
        if max_num_seqs is None:
            max_num_seqs = DEFAULT_MAX_NUM_SEQS
        if max_num_batched_tokens is None:
            max_num_batched_tokens = DEFAULT_MAX_NUM_BATCHED_TOKENS
        return np.array([thread, memory, load_rps, tp_size, max_model_len, max_num_seqs, max_num_batched_tokens], dtype=np.float64)
    return np.array([thread, memory, load_rps, tp_size], dtype=np.float64)


TARGET_CHOICES = ("ttft", "tbt", "latency")


def get_ttft_target(result: dict) -> float | None:
    """
    Get mean TTFT (ms) from a result dict. Returns None if failed or no TTFT data.
    """
    if result.get("failed"):
        return None
    ttfts = result.get("ttfts_ms")
    if ttfts is None or len(ttfts) == 0:
        perf = result.get("performance", {})
        metrics_after = result.get("metrics_after", {})
        if "avg_ttft_ms" in metrics_after:
            return float(metrics_after["avg_ttft_ms"])
        if "avg_ttft_ms" in perf:
            return float(perf["avg_ttft_ms"])
        return None
    return float(np.mean(ttfts))


def get_tbt_target(result: dict) -> float | None:
    """
    Get mean TBT (Time Between Tokens, ms) from a result dict.
    Flattens tbt_lists_ms and returns mean of all TBT values. Returns None if failed or no TBT data.
    """
    if result.get("failed"):
        return None
    tbt_lists = result.get("tbt_lists_ms")
    if tbt_lists is None or len(tbt_lists) == 0:
        metrics_after = result.get("metrics_after", {})
        if "avg_tbt_ms" in metrics_after:
            return float(metrics_after["avg_tbt_ms"])
        return None
    all_tbts = [t for lst in tbt_lists for t in lst]
    if not all_tbts:
        return None
    return float(np.mean(all_tbts))


def get_latency_target(result: dict) -> float | None:
    """
    Get average latency (ms) from a result dict. Returns None if failed or no latency data.
    """
    if result.get("failed"):
        return None
    perf = result.get("performance", {})
    if "avg_latency_ms" not in perf:
        return None
    return float(perf["avg_latency_ms"])


def get_target_value(result: dict, target: str) -> float | None:
    """Dispatch to the appropriate get_*_target by target name."""
    if target == "ttft":
        return get_ttft_target(result)
    if target == "tbt":
        return get_tbt_target(result)
    if target == "latency":
        return get_latency_target(result)
    raise ValueError(f"Unknown target: {target}. Choose from {TARGET_CHOICES}")


def extract_xy(
    data: list,
    target: str,
    use_optional_features: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract feature matrix X and target vector y from performance data list.
    Skips failed experiments and rows with missing target.
    """
    X_list = []
    y_list = []
    for item in data:
        setup = item["setup"]
        result = item["result"]
        value = get_target_value(result, target)
        if value is None:
            continue
        x = build_feature_vector(setup, use_optional=use_optional_features)
        X_list.append(x)
        y_list.append(value)
    if not X_list:
        raise ValueError(
            f"No valid (setup, {target}) pairs found in the input file. "
            "Check for 'failed' results or missing data for the chosen target."
        )
    return np.array(X_list), np.array(y_list)


def get_feature_names(use_optional: bool) -> list[str]:
    base = ["thread_percentage", "memory_util", "load_rps", "tensor_parallel_size"]
    if use_optional:
        return base + ["max_model_len", "max_num_seqs", "max_num_batched_tokens"]
    return base


def predict_metric(artifact: dict, setup: dict) -> float:
    """
    Predict target (ms) for a setup dict using a saved artifact.
    Works for any target (ttft, tbt, latency) stored in artifact["target"].

    Usage:
        artifact = joblib.load("profiler/ttft_model.joblib")
        pred_ms = predict_ms(artifact, {"thread_percentage": 50, "memory_util": 0.5, ...})
    """
    use_optional = artifact.get("use_optional_features", True)
    x = build_feature_vector(setup, use_optional=use_optional).reshape(1, -1)
    if artifact.get("poly"):
        x = artifact["poly"].transform(x)
    return float(artifact["model"].predict(x)[0])


def main():
    parser = argparse.ArgumentParser(
        description="Train linear regression to predict TTFT, TBT, or latency (ms) from hyperparameters"
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=TARGET_CHOICES,
        default="ttft",
        help="Target to predict: ttft (Time To First Token), tbt (Time Between Tokens), or latency (avg latency)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="performance_data.json",
        help="Path to performance data JSON from collect_performance_data.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the trained model (joblib). Default: profiler/<target>_model.joblib",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of data for test set (0 = use all for training)",
    )
    parser.add_argument(
        "--no-optional-features",
        action="store_true",
        help="Use only thread_percentage, memory_util, load_rps, tensor_parallel_size (no max_model_len etc.)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=MODEL_TYPE_CHOICES,
        default="linear",
        help="Model to train: linear (with optional polynomial features), or gradient_boosting",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = f"profiler/{args.target}_model.joblib"

    use_optional = not args.no_optional_features
    feature_names = get_feature_names(use_optional)
    use_poly = args.model_type == "linear" and USE_POLYNOMIAL_FEATURES

    print(f"Target: {args.target}")
    print(f"Model type: {args.model_type}")
    print("Loading performance data...")
    data = load_performance_data(args.input)
    print(f"  Loaded {len(data)} experiments")

    print("Extracting features and targets...")
    X, y = extract_xy(data, target=args.target, use_optional_features=use_optional)
    n = len(y)
    print(f"  Valid samples: {n} ({args.target} mean={y.mean():.2f} ms, std={y.std():.2f} ms)")

    if args.test_fraction > 0 and args.test_fraction < 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_fraction, random_state=args.seed
        )
        print(f"  Train: {len(y_train)}, Test: {len(y_test)}")
    else:
        X_train, y_train = X, y
        X_test, y_test = None, None
        print("  Using all data for training (no test split)")

    poly = None
    if args.model_type == "linear":
        print("Training LinearRegression...")
        if use_poly:
            poly = PolynomialFeatures(degree=3)
            X_train = poly.fit_transform(X_train)
            if X_test is not None and len(X_test) > 0:
                X_test = poly.transform(X_test)
        model = LinearRegression()
        model.fit(X_train, y_train)
    else:
        # gradient_boosting: no polynomial features; model learns non-linearity
        print("Training HistGradientBoostingRegressor...")
        model = HistGradientBoostingRegressor(
            max_iter=200,
            max_depth=6,
            learning_rate=0.1,
            min_samples_leaf=5,
            random_state=args.seed,
        )
        model.fit(X_train, y_train)

    # Metrics on training set
    if use_poly:
        y_train_pred = model.predict(X_train).flatten()
    else:
        y_train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    print(f"\nTrain set:")
    print(f"  MAE:  {train_mae:.2f} ms")
    print(f"  RMSE: {train_rmse:.2f} ms")
    print(f"  R²:   {train_r2:.4f}")

    if X_test is not None and len(X_test) > 0:
        if use_poly:
            y_test_pred = model.predict(X_test).flatten()
        else:
            y_test_pred = model.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        print(f"\nTest set:")
        print(f"  MAE:  {test_mae:.2f} ms")
        print(f"  RMSE: {test_rmse:.2f} ms")
        print(f"  R²:   {test_r2:.4f}")

    if args.model_type == "linear":
        n_coef = len(model.coef_)
        n_names = len(feature_names)
        print("\nCoefficients:")
        if n_coef == n_names:
            for name, coef in zip(feature_names, model.coef_):
                print(f"  {name}: {coef:.4f}")
        else:
            for i, coef in enumerate(model.coef_[:n_names]):
                print(f"  {feature_names[i]}: {coef:.4f}")
            if n_coef > n_names:
                print(f"  ... ({n_coef - n_names} polynomial terms)")
        print(f"  intercept: {model.intercept_:.4f}")
    else:
        if hasattr(model, "feature_importances_"):
            print("\nFeature importances:")
            for name, imp in zip(feature_names, model.feature_importances_):
                print(f"  {name}: {imp:.4f}")
        else:
            print("\nFeature importances: not available for this model/sklearn version")

    # Save model and metadata for inference
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "model_type": args.model_type,
        "target": args.target,
        "feature_names": feature_names,
        "use_optional_features": use_optional,
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "train_r2": train_r2,
    }
    if poly is not None:
        artifact["poly"] = poly
    if X_test is not None and len(X_test) > 0:
        artifact["test_mae"] = test_mae
        artifact["test_rmse"] = test_rmse
        artifact["test_r2"] = test_r2
    joblib.dump(artifact, out_path)
    print(f"\nSaved model and metadata to {out_path}")
    print(f"  Target: {args.target}, Model: {args.model_type}")
    print("  To predict: artifact = joblib.load(path); pred = predict_metric(artifact, setup)")


if __name__ == "__main__":
    main()

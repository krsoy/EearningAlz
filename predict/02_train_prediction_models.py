#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_train_prediction_models.py

Train logistic-regression prediction models for transcript-signal propagation.

Input files expected in --input-dir:
    prediction_dataset_cross_quarter.parquet
    prediction_dataset_same_quarter_ordered.parquet

Output files written to --out-dir:
    prediction_model_metrics.csv
    prediction_model_metrics.parquet
    run_summary.json

    models/
        <dataset>__<specification>__logistic_regression.joblib
        <dataset>__<specification>__logistic_regression.metadata.json
        <dataset>__<specification>__logistic_regression.features.json

    test_predictions/
        <dataset>__<specification>__logistic_regression.test_predictions.parquet

Target:
    y = 1 if target_active and direction_match, else 0

Run example:
    python 02_train_prediction_models.py ^
      --input-dir prediction_model_outputs ^
      --out-dir prediction_model_outputs\\model_results ^
      --test-start-quarter 2024Q1 ^
      --dataset both
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# Config
# ============================================================

DATASET_FILES = {
    "cross_quarter": "prediction_dataset_cross_quarter.parquet",
    "same_quarter_ordered": "prediction_dataset_same_quarter_ordered.parquet",
}

MODEL_FAMILY = "logistic_regression"

SPECIFICATIONS = [
    "target_history_baseline",
    "community_baseline",
    "network_source_model",
    "full_network_community_model",
]


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        default="prediction_model_outputs",
        help="Directory containing prediction_dataset_*.parquet files.",
    )
    parser.add_argument(
        "--out-dir",
        default="prediction_model_outputs/model_results",
        help="Directory where metrics, models, metadata, and predictions are saved.",
    )
    parser.add_argument(
        "--test-start-quarter",
        default="2024Q1",
        help="First target quarter used for the test set, e.g. 2024Q1.",
    )
    parser.add_argument(
        "--dataset",
        default="both",
        choices=["both", "cross_quarter", "same_quarter_ordered"],
        help="Which dataset to train on.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for LogisticRegression.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for y_pred from y_prob.",
    )
    parser.add_argument(
        "--class-weight",
        default="balanced",
        choices=["balanced", "none"],
        help="Class weight for LogisticRegression.",
    )
    parser.add_argument(
        "--min-category-frequency",
        type=int,
        default=5,
        help=(
            "Minimum category frequency for OneHotEncoder if supported by the local "
            "scikit-learn version. Set 1 to effectively disable."
        ),
    )
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Do not save row-level test predictions.",
    )

    return parser.parse_args()


# ============================================================
# Basic helpers
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(x: Any) -> str:
    s = str(x)
    s = s.replace("\\", "_").replace("/", "_").replace(":", "_")
    s = re.sub(r"[^A-Za-z0-9_.\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def quarter_to_index(q: Any) -> float:
    m = re.match(r"^(\d{4})Q([1-4])$", str(q).strip())
    if not m:
        return np.nan
    return int(m.group(1)) * 4 + int(m.group(2))


def to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
    )


def safe_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


# ============================================================
# Target and split
# ============================================================

def infer_target(df: pd.DataFrame) -> pd.Series:
    """
    Use existing y column if present.
    Otherwise derive:
        y = 1 if target_active and direction_match, else 0
    """
    if "y" in df.columns:
        y = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int)
        return y

    required = {"target_active", "direction_match"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Cannot infer target y. Missing columns: "
            + ", ".join(sorted(missing))
            + ". Expected either a column named 'y' or both "
            "'target_active' and 'direction_match'."
        )

    target_active = to_bool_series(df["target_active"])
    direction_match = to_bool_series(df["direction_match"])
    y = (target_active & direction_match).astype(int)
    return y


def find_target_quarter_column(df: pd.DataFrame) -> str:
    candidates = [
        "target_quarter",
        "q_prime",
        "target_q",
        "quarter_target",
        "quarter",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "Cannot find target quarter column. Tried: "
        + ", ".join(candidates)
    )


def time_split(
    df: pd.DataFrame,
    y: pd.Series,
    test_start_quarter: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    q_col = find_target_quarter_column(df)
    q_index = df[q_col].map(quarter_to_index)
    test_start_index = quarter_to_index(test_start_quarter)

    if pd.isna(test_start_index):
        raise ValueError(f"Invalid --test-start-quarter: {test_start_quarter}")

    valid = q_index.notna()
    df = df.loc[valid].copy()
    y = y.loc[valid].copy()
    q_index = q_index.loc[valid]

    train_mask = q_index < test_start_index
    test_mask = q_index >= test_start_index

    if train_mask.sum() == 0:
        raise ValueError("Training set is empty after time split.")
    if test_mask.sum() == 0:
        raise ValueError("Test set is empty after time split.")

    return (
        df.loc[train_mask].copy(),
        df.loc[test_mask].copy(),
        y.loc[train_mask].astype(int),
        y.loc[test_mask].astype(int),
    )


# ============================================================
# Feature selection
# ============================================================

def is_history_like(col: str) -> bool:
    c = col.lower()
    return any(
        k in c
        for k in [
            "history",
            "hist",
            "past",
            "lag",
            "prev",
            "previous",
            "rolling",
            "prior",
        ]
    )


def is_leakage_or_bad_column(col: str) -> bool:
    """
    Exclude labels/outcomes/current target facts/provenance/date diagnostics
    that would leak the answer or are not useful for modelling.

    History-like target features are allowed if their names contain history/past/lag/etc.
    """
    c = col.lower()

    # Target column itself
    if c in {"y", "target", "label"}:
        return True

    # Outcome / evaluation columns
    outcome_patterns = [
        "direction_match",
        "exact_match",
        "prediction_correct",
        "prediction_accuracy",
        "success",
        "non_direction",
        "non_exact",
        "falsification",
    ]
    if any(p in c for p in outcome_patterns):
        return True

    # Current target signal values leak the target.
    # But target history / previous values are allowed.
    if not is_history_like(c):
        target_leak_patterns = [
            "target_active",
            "target_label",
            "target_direction",
            "target_score",
            "target_value",
            "target_outlook",
            "target_signal",
        ]
        if any(p in c for p in target_leak_patterns):
            return True

    # Publication-date diagnostics are not used as predictive features here.
    date_patterns = [
        "publish_date",
        "release_date",
        "date_gap",
        "publish_gap",
        "source_before_target",
        "date_observation",
        "mean_publish",
        "median_publish",
    ]
    if any(p in c for p in date_patterns):
        return True

    # Raw provenance / text / ids
    provenance_patterns = [
        "doc_id",
        "doc_ids",
        "row_id",
        "row_ids",
        "chunk",
        "evidence",
        "parquet",
        "_file",
        "source_file",
        "target_file",
        "raw_text",
        "transcript",
        "text",
    ]
    if any(p in c for p in provenance_patterns):
        return True

    return False


def base_allowed_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if is_leakage_or_bad_column(col):
            continue
        cols.append(col)
    return cols


def has_any(col: str, keywords: list[str]) -> bool:
    c = col.lower()
    return any(k.lower() in c for k in keywords)


def select_feature_columns(df: pd.DataFrame, specification: str) -> list[str]:
    """
    Four feature specifications:

    target_history_baseline:
        Target identity + signal setup + target past/history features.

    community_baseline:
        Community information + target history + basic signal setup.

    network_source_model:
        Source-side and relation/network exposure features.

    full_network_community_model:
        All non-leakage columns.
    """
    allowed = base_allowed_columns(df)
    allowed_set = set(allowed)

    always_keep = [
        "analysis_mode",
        "signal",
        "source_label",
        "source_direction",
        "relation_group",
        "source_quarter",
        "target_quarter",
        "source_node",
        "target_node",
        "source_company_node",
        "target_company_node",
    ]
    always_keep = [c for c in always_keep if c in allowed_set]

    if specification == "full_network_community_model":
        selected = allowed

    elif specification == "target_history_baseline":
        keywords = [
            "target_history",
            "target_hist",
            "target_past",
            "target_prev",
            "target_previous",
            "target_lag",
            "target_rolling",
            "target_prior",
            "target_rate",
            "baseline",
        ]
        selected = [
            c for c in allowed
            if c in always_keep
            or has_any(c, keywords)
        ]

        # Keep target identity if available because target history is target-specific.
        for c in ["target_node", "target_company_node"]:
            if c in allowed_set and c not in selected:
                selected.append(c)

    elif specification == "community_baseline":
        keywords = [
            "community",
            "cluster",
            "same_community",
            "source_community",
            "target_community",
            "target_history",
            "target_hist",
            "target_past",
            "target_prev",
            "target_lag",
            "target_rolling",
            "baseline",
        ]
        selected = [
            c for c in allowed
            if c in always_keep
            or has_any(c, keywords)
        ]

    elif specification == "network_source_model":
        keywords = [
            "source_",
            "relation",
            "network",
            "edge",
            "degree",
            "centrality",
            "exposure",
            "source_history",
            "source_hist",
            "source_past",
            "source_prev",
            "source_lag",
            "source_rolling",
        ]
        selected = [
            c for c in allowed
            if c in always_keep
            or has_any(c, keywords)
        ]

    else:
        raise ValueError(f"Unknown specification: {specification}")

    # Drop columns that are completely empty or constant.
    cleaned = []
    for col in selected:
        if col not in df.columns:
            continue
        s = df[col]
        if s.notna().sum() == 0:
            continue
        if s.astype(str).nunique(dropna=True) <= 1:
            continue
        cleaned.append(col)

    # Preserve order, remove duplicates.
    seen = set()
    out = []
    for c in cleaned:
        if c not in seen:
            out.append(c)
            seen.add(c)

    if not out:
        raise ValueError(
            f"No feature columns selected for specification={specification}. "
            "Check dataset schema or feature selection rules."
        )

    return out


def split_feature_types(df: pd.DataFrame, feature_columns: list[str]) -> tuple[list[str], list[str]]:
    numeric_cols = []
    categorical_cols = []

    for col in feature_columns:
        if pd.api.types.is_bool_dtype(df[col]):
            numeric_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


# ============================================================
# Model pipeline
# ============================================================

def make_one_hot_encoder(min_frequency: int) -> OneHotEncoder:
    """
    Handle older sklearn versions that do not support min_frequency.
    """
    try:
        if min_frequency and min_frequency > 1:
            return OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=min_frequency,
                sparse_output=True,
            )
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        # older sklearn uses `sparse`, not `sparse_output`,
        # and may not support min_frequency.
        try:
            if min_frequency and min_frequency > 1:
                return OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=min_frequency,
                    sparse=True,
                )
            return OneHotEncoder(handle_unknown="ignore", sparse=True)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_model_pipeline(
    df: pd.DataFrame,
    feature_columns: list[str],
    args: argparse.Namespace,
) -> Pipeline:
    numeric_cols, categorical_cols = split_feature_types(df, feature_columns)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder(args.min_category_frequency)),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )

    class_weight = None if args.class_weight == "none" else "balanced"

    clf = LogisticRegression(
        max_iter=args.max_iter,
        class_weight=class_weight,
        solver="lbfgs",
        random_state=args.random_state,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clf),
        ]
    )

    return model


# ============================================================
# Metrics and saving
# ============================================================

def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float], np.ndarray]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred = (y_prob >= threshold).astype(int)

    metrics: dict[str, float] = {}

    if len(np.unique(y_true_arr)) >= 2:
        metrics["test_auc"] = float(roc_auc_score(y_true_arr, y_prob))
    else:
        metrics["test_auc"] = float("nan")

    metrics["test_balanced_accuracy"] = float(balanced_accuracy_score(y_true_arr, y_pred))
    metrics["test_f1"] = float(f1_score(y_true_arr, y_pred, zero_division=0))
    metrics["test_precision"] = float(precision_score(y_true_arr, y_pred, zero_division=0))
    metrics["test_recall"] = float(recall_score(y_true_arr, y_pred, zero_division=0))
    metrics["test_brier"] = float(brier_score_loss(y_true_arr, y_prob))

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    metrics["test_tn"] = int(tn)
    metrics["test_fp"] = int(fp)
    metrics["test_fn"] = int(fn)
    metrics["test_tp"] = int(tp)

    return metrics, y_pred


def fit_eval_save_model(
    df_train: pd.DataFrame,
    y_train: pd.Series,
    df_test: pd.DataFrame,
    y_test: pd.Series,
    dataset_name: str,
    specification: str,
    args: argparse.Namespace,
    model_dir: Path,
    pred_dir: Path,
) -> dict[str, Any]:
    feature_columns = select_feature_columns(df_train, specification)

    X_train = df_train[feature_columns].copy()
    X_test = df_test[feature_columns].copy()

    # Clean numeric infinities without converting categorical columns.
    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            X_train[col] = safe_numeric_series(X_train[col])
            X_test[col] = safe_numeric_series(X_test[col])
        else:
            X_train[col] = X_train[col].astype("object").where(X_train[col].notna(), None)
            X_test[col] = X_test[col].astype("object").where(X_test[col].notna(), None)

    model = build_model_pipeline(df_train, feature_columns, args)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    metric_values, y_pred = compute_metrics(y_test, y_prob, args.threshold)

    result = {
        "dataset": dataset_name,
        "model_family": MODEL_FAMILY,
        "specification": specification,
        "train_n": int(len(y_train)),
        "test_n": int(len(y_test)),
        "train_positive_rate": float(np.mean(y_train)),
        "test_positive_rate": float(np.mean(y_test)),
        "feature_count": int(len(feature_columns)),
        **metric_values,
    }

    safe_dataset = safe_name(dataset_name)
    safe_spec = safe_name(specification)
    safe_family = safe_name(MODEL_FAMILY)
    stem = f"{safe_dataset}__{safe_spec}__{safe_family}"

    model_path = model_dir / f"{stem}.joblib"
    metadata_path = model_dir / f"{stem}.metadata.json"
    features_path = model_dir / f"{stem}.features.json"
    predictions_path = pred_dir / f"{stem}.test_predictions.parquet"

    # Save full fitted pipeline, not only LogisticRegression.
    joblib.dump(model, model_path)

    save_json(feature_columns, features_path)

    metadata = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_name,
        "model_family": MODEL_FAMILY,
        "specification": specification,
        "target": "y = 1 if target_active and direction_match, else 0",
        "test_start_quarter": args.test_start_quarter,
        "threshold": args.threshold,
        "class_weight": args.class_weight,
        "min_category_frequency": args.min_category_frequency,
        "train_n": int(len(y_train)),
        "test_n": int(len(y_test)),
        "train_positive_rate": float(np.mean(y_train)),
        "test_positive_rate": float(np.mean(y_test)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "metrics": result,
        "model_path": str(model_path),
        "features_path": str(features_path),
        "predictions_path": str(predictions_path),
    }
    save_json(metadata, metadata_path)

    if not args.no_save_predictions:
        pred_df = pd.DataFrame(
            {
                "row_index": df_test.index.to_numpy(),
                "y_true": np.asarray(y_test).astype(int),
                "y_prob": y_prob,
                "y_pred": y_pred.astype(int),
            }
        )

        # Preserve useful identifiers if present.
        keep_cols = [
            "analysis_mode",
            "source_node",
            "target_node",
            "source_company_node",
            "target_company_node",
            "source_quarter",
            "target_quarter",
            "signal",
            "source_label",
            "source_direction",
            "relation_group",
        ]
        for c in keep_cols:
            if c in df_test.columns:
                pred_df[c] = df_test[c].to_numpy()

        pred_df.to_parquet(predictions_path, index=False)

    print(
        f"  {MODEL_FAMILY:19s} | {specification:30s}: "
        f"test_auc={result['test_auc']:.3f}, "
        f"bal_acc={result['test_balanced_accuracy']:.3f}, "
        f"f1={result['test_f1']:.3f}"
    )
    print(f"    SAVED MODEL: {model_path}")

    return result


# ============================================================
# Dataset loading
# ============================================================

def datasets_to_run(args: argparse.Namespace) -> list[str]:
    if args.dataset == "both":
        return ["cross_quarter", "same_quarter_ordered"]
    return [args.dataset]


def load_dataset_file(input_dir: Path, dataset_name: str) -> pd.DataFrame:
    filename = DATASET_FILES[dataset_name]
    path = input_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")

    df = pd.read_parquet(path)
    print(f"Loaded {dataset_name}: {path} rows={len(df):,}, cols={len(df.columns):,}")
    return df


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    out_dir = ensure_dir(Path(args.out_dir))
    model_dir = ensure_dir(out_dir / "models")
    pred_dir = ensure_dir(out_dir / "test_predictions")

    print("=" * 90)
    print("Train prediction models")
    print(f"Input dir          : {input_dir}")
    print(f"Output dir         : {out_dir}")
    print(f"Model dir          : {model_dir}")
    print(f"Prediction dir     : {pred_dir}")
    print(f"Test start quarter : {args.test_start_quarter}")
    print(f"Dataset            : {args.dataset}")
    print(f"Class weight       : {args.class_weight}")
    print("=" * 90)

    all_results: list[dict[str, Any]] = []
    run_summary: dict[str, Any] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "test_start_quarter": args.test_start_quarter,
        "dataset_arg": args.dataset,
        "target": "y = 1 if target_active and direction_match, else 0",
        "datasets": {},
        "models": [],
    }

    for dataset_name in datasets_to_run(args):
        df = load_dataset_file(input_dir, dataset_name)
        y = infer_target(df)

        df_train, df_test, y_train, y_test = time_split(
            df=df,
            y=y,
            test_start_quarter=args.test_start_quarter,
        )

        print(
            f"\nDataset {dataset_name}: "
            f"rows={len(df):,}, "
            f"positive_rate={float(np.mean(y)):.3f}, "
            f"train={len(df_train):,}, "
            f"test={len(df_test):,}"
        )

        run_summary["datasets"][dataset_name] = {
            "rows": int(len(df)),
            "positive_rate": float(np.mean(y)),
            "train_rows": int(len(df_train)),
            "test_rows": int(len(df_test)),
            "train_positive_rate": float(np.mean(y_train)),
            "test_positive_rate": float(np.mean(y_test)),
        }

        for specification in SPECIFICATIONS:
            result = fit_eval_save_model(
                df_train=df_train,
                y_train=y_train,
                df_test=df_test,
                y_test=y_test,
                dataset_name=dataset_name,
                specification=specification,
                args=args,
                model_dir=model_dir,
                pred_dir=pred_dir,
            )
            all_results.append(result)
            run_summary["models"].append(result)

    metrics_df = pd.DataFrame(all_results)

    preferred_order = [
        "dataset",
        "model_family",
        "specification",
        "train_n",
        "test_n",
        "train_positive_rate",
        "test_positive_rate",
        "feature_count",
        "test_auc",
        "test_balanced_accuracy",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_brier",
        "test_tn",
        "test_fp",
        "test_fn",
        "test_tp",
    ]
    cols = [c for c in preferred_order if c in metrics_df.columns] + [
        c for c in metrics_df.columns if c not in preferred_order
    ]
    metrics_df = metrics_df[cols]

    csv_path = out_dir / "prediction_model_metrics.csv"
    parquet_path = out_dir / "prediction_model_metrics.parquet"
    summary_path = out_dir / "run_summary.json"

    metrics_df.to_csv(csv_path, index=False)
    metrics_df.to_parquet(parquet_path, index=False)

    run_summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    run_summary["metrics_csv"] = str(csv_path)
    run_summary["metrics_parquet"] = str(parquet_path)
    run_summary["model_dir"] = str(model_dir)
    run_summary["prediction_dir"] = str(pred_dir)
    save_json(run_summary, summary_path)

    print(f"\nSAVED {csv_path}")
    print(f"SAVED {parquet_path}")
    print(f"SAVED {summary_path}")

    print("\n=== TEST METRICS SUMMARY ===")
    display_cols = [
        "dataset",
        "model_family",
        "specification",
        "test_n",
        "test_positive_rate",
        "test_auc",
        "test_balanced_accuracy",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_brier",
    ]
    display_cols = [c for c in display_cols if c in metrics_df.columns]
    print(
        metrics_df[display_cols]
        .sort_values(["dataset", "test_auc"], ascending=[True, False])
        .to_string(index=False)
    )

    print("\nDONE.")


if __name__ == "__main__":
    main()
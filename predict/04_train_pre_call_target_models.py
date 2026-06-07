#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_train_pre_call_target_models.py

Train pre-call target-firm-quarter-signal prediction models.

Input:
    pre_call_target_signal_dataset.parquet

Targets:
    --task active
        Binary classification:
        y = 1 if the target firm mentions the signal in the target quarter call.

    --task direction
        Multi-class classification:
        y in {not_active, positive, negative, neutral, mixed}

    --task both
        Train both active and direction models.

Models:
    --model logistic_regression   (original)
    --model xgboost               (new)
    --model all                   (run both)

Run:
    python 04_train_pre_call_target_models.py ^
      --input prediction_model_outputs\\pre_call_target_signal_dataset.parquet ^
      --out-dir prediction_model_outputs\\pre_call_model_results ^
      --test-start-quarter 2024Q1 ^
      --task both ^
      --model all
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
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input",
        default="prediction_model_outputs/pre_call_target_signal_dataset.parquet",
    )
    p.add_argument(
        "--out-dir",
        default="prediction_model_outputs/pre_call_model_results",
    )
    p.add_argument(
        "--test-start-quarter",
        default="2024Q1",
    )
    p.add_argument(
        "--task",
        default="both",
        choices=["both", "active", "direction"],
    )
    p.add_argument(
        "--model",
        default="all",
        choices=["logistic_regression", "xgboost", "all"],
        help="Which model family to train. 'all' trains both.",
    )
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--class-weight", default="balanced", choices=["balanced", "none"])
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-category-frequency", type=int, default=5)

    # XGBoost hyper-params
    p.add_argument("--xgb-n-estimators", type=int, default=400)
    p.add_argument("--xgb-max-depth", type=int, default=5)
    p.add_argument("--xgb-learning-rate", type=float, default=0.05)
    p.add_argument("--xgb-subsample", type=float, default=0.8)
    p.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    p.add_argument("--xgb-min-child-weight", type=int, default=10)
    p.add_argument("--xgb-reg-alpha", type=float, default=0.1)
    p.add_argument("--xgb-reg-lambda", type=float, default=1.0)

    return p.parse_args()


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


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


def make_one_hot_encoder(min_frequency: int) -> OneHotEncoder:
    try:
        if min_frequency and min_frequency > 1:
            return OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=min_frequency,
                sparse_output=True,
            )
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
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


# ============================================================
# Data split and feature selection
# ============================================================

def time_split(df: pd.DataFrame, test_start_quarter: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "target_quarter_index" in df.columns:
        qidx = pd.to_numeric(df["target_quarter_index"], errors="coerce")
    else:
        qidx = df["target_quarter"].map(quarter_to_index)

    test_start_idx = quarter_to_index(test_start_quarter)
    if pd.isna(test_start_idx):
        raise ValueError(f"Invalid test-start-quarter: {test_start_quarter}")

    valid = qidx.notna()
    df = df.loc[valid].copy()
    qidx = qidx.loc[valid]

    train = df.loc[qidx < test_start_idx].copy()
    test = df.loc[qidx >= test_start_idx].copy()

    if len(train) == 0:
        raise ValueError("Empty train set.")
    if len(test) == 0:
        raise ValueError("Empty test set.")

    return train, test


def leakage_columns() -> set[str]:
    return {
        "target_active_label",
        "target_direction_label",
        "target_direction_code",
        "target_label_raw_mode",
        "label_event_rows",
    }


def select_feature_columns(df: pd.DataFrame, specification: str) -> list[str]:
    blocked = leakage_columns()
    base = [c for c in df.columns if c not in blocked]
    base = [c for c in base if c != "target_quarter"]

    if specification == "full_pre_call":
        selected = base
    elif specification == "history_only":
        selected = [
            c for c in base
            if c in {"target_company", "signal", "target_community", "target_quarter_index"}
            or c.startswith("hist_")
        ]
    elif specification == "source_only":
        selected = [
            c for c in base
            if c in {"target_company", "signal", "target_community", "target_quarter_index"}
            or c.startswith("prevq_")
            or c.startswith("sameq_pre_")
        ]
        selected = [c for c in selected if not c.startswith("hist_")]
    elif specification == "source_plus_history":
        selected = [
            c for c in base
            if c in {"target_company", "signal", "target_community", "target_quarter_index"}
            or c.startswith("prevq_")
            or c.startswith("sameq_pre_")
            or c.startswith("hist_")
        ]
    else:
        raise ValueError(f"Unknown specification: {specification}")

    cleaned = []
    for c in selected:
        if c not in df.columns:
            continue
        if df[c].notna().sum() == 0:
            continue
        if df[c].astype(str).nunique(dropna=True) <= 1:
            continue
        cleaned.append(c)

    seen = set()
    out = []
    for c in cleaned:
        if c not in seen:
            out.append(c)
            seen.add(c)

    if not out:
        raise ValueError(f"No features selected for {specification}")

    return out


def split_feature_types(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[list[str], list[str]]:
    numeric, categorical = [], []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            numeric.append(c)
        else:
            categorical.append(c)
    return numeric, categorical


# ============================================================
# Pipeline builders
# ============================================================

def build_lr_pipeline(
    df: pd.DataFrame,
    feature_cols: list[str],
    args: argparse.Namespace,
    task: str,
) -> Pipeline:
    numeric, categorical = split_feature_types(df, feature_cols)
    transformers = []
    if numeric:
        transformers.append((
            "num",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "cat",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_one_hot_encoder(args.min_category_frequency)),
            ]),
            categorical,
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers, remainder="drop", sparse_threshold=0.3,
    )
    class_weight = None if args.class_weight == "none" else "balanced"
    clf = LogisticRegression(
        max_iter=args.max_iter,
        class_weight=class_weight,
        random_state=args.random_state,
        solver="lbfgs",
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])


def build_xgb_pipeline(
    df: pd.DataFrame,
    feature_cols: list[str],
    args: argparse.Namespace,
    task: str,
    n_classes: int = 2,
) -> Pipeline:
    """
    XGBoost pipeline.

    Key design choices vs logistic regression:
      - No StandardScaler: XGBoost is scale-invariant (tree splits).
      - OrdinalEncoder for categoricals: encodes to integers, XGBoost handles
        them as numeric splits. Faster and often better than one-hot for trees.
      - sparse_threshold=0.0: XGBoost needs a dense matrix.
      - Class imbalance handled via scale_pos_weight (binary) or
        sample_weight at fit time (multiclass) — set in train_one().
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("xgboost is required. Install with: pip install xgboost")

    numeric, categorical = split_feature_types(df, feature_cols)
    transformers = []
    if numeric:
        transformers.append((
            "num",
            SimpleImputer(strategy="median"),
            numeric,
        ))
    if categorical:
        transformers.append((
            "cat",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )),
            ]),
            categorical,
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers, remainder="drop", sparse_threshold=0.0,
    )

    common_params = dict(
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        min_child_weight=args.xgb_min_child_weight,
        reg_alpha=args.xgb_reg_alpha,
        reg_lambda=args.xgb_reg_lambda,
        random_state=args.random_state,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
    )

    if task == "active":
        clf = XGBClassifier(objective="binary:logistic", **common_params)
    else:
        clf = XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            **common_params,
        )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])


# ============================================================
# Class imbalance helpers for XGBoost
# ============================================================

def compute_binary_scale_pos_weight(y_train: np.ndarray) -> float:
    """XGBoost convention: scale_pos_weight = neg_count / pos_count."""
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    if n_pos == 0:
        return 1.0
    return float(n_neg) / float(n_pos)


def compute_sample_weights_multiclass(y_train: np.ndarray) -> np.ndarray:
    """Per-sample inverse-frequency weights for multiclass XGBoost."""
    classes, counts = np.unique(y_train, return_counts=True)
    weight_map = {
        c: len(y_train) / (len(classes) * cnt)
        for c, cnt in zip(classes, counts)
    }
    return np.array([weight_map[y] for y in y_train])


# ============================================================
# Metrics
# ============================================================

def binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> tuple[dict[str, Any], np.ndarray]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics: dict[str, Any] = {
        "test_n": int(len(y_true)),
        "test_positive_rate": float(np.mean(y_true)),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "test_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "test_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "test_brier": float(brier_score_loss(y_true, y_prob)),
    }
    if len(np.unique(y_true)) >= 2:
        metrics["test_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["test_auc"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update({"test_tn": int(tn), "test_fp": int(fp), "test_fn": int(fn), "test_tp": int(tp)})
    return metrics, y_pred


def multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "test_n": int(len(y_true)),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "test_macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "labels": labels,
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0, output_dict=True,
        ),
    }
    try:
        metrics["test_auc_ovr_weighted"] = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted", labels=labels)
        )
    except Exception:
        metrics["test_auc_ovr_weighted"] = float("nan")
    return metrics


# ============================================================
# Training
# ============================================================

def train_one(
    train: pd.DataFrame,
    test: pd.DataFrame,
    task: str,
    specification: str,
    model_family: str,
    args: argparse.Namespace,
    model_dir: Path,
    pred_dir: Path,
) -> dict[str, Any]:
    feature_cols = select_feature_columns(train, specification)

    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(X_train[c]):
            X_train[c] = pd.to_numeric(X_train[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            X_test[c] = pd.to_numeric(X_test[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        else:
            X_train[c] = X_train[c].astype("object").where(X_train[c].notna(), None)
            X_test[c] = X_test[c].astype("object").where(X_test[c].notna(), None)

    if task == "active":
        y_train = train["target_active_label"].astype(int).to_numpy()
        y_test = test["target_active_label"].astype(int).to_numpy()
    else:
        y_train = train["target_direction_label"].astype(str).to_numpy()
        y_test = test["target_direction_label"].astype(str).to_numpy()

    fit_params: dict[str, Any] = {}

    if model_family == "logistic_regression":
        model = build_lr_pipeline(train, feature_cols, args, task=task)

    elif model_family == "xgboost":
        if task == "active":
            spw = compute_binary_scale_pos_weight(y_train)
            model = build_xgb_pipeline(train, feature_cols, args, task=task)
            model.named_steps["model"].set_params(scale_pos_weight=spw)
        else:
            classes = sorted(np.unique(y_train).tolist())
            n_classes = len(classes)
            model = build_xgb_pipeline(
                train, feature_cols, args, task=task, n_classes=n_classes,
            )
            sample_weights = compute_sample_weights_multiclass(y_train)
            fit_params["model__sample_weight"] = sample_weights
    else:
        raise ValueError(f"Unknown model_family: {model_family}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train, **fit_params)

    if task == "active":
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics, y_pred = binary_metrics(y_test, y_prob, args.threshold)
    else:
        y_prob = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        labels = list(model.named_steps["model"].classes_)
        metrics = multiclass_metrics(y_test, y_pred, y_prob, labels=labels)

    stem = (
        f"pre_call__{safe_name(task)}__{safe_name(specification)}"
        f"__{safe_name(model_family)}"
    )

    model_path = model_dir / f"{stem}.joblib"
    metadata_path = model_dir / f"{stem}.metadata.json"
    features_path = model_dir / f"{stem}.features.json"
    pred_path = pred_dir / f"{stem}.test_predictions.parquet"

    joblib.dump(model, model_path)
    save_json(feature_cols, features_path)

    result = {
        "task": task,
        "model_family": model_family,
        "specification": specification,
        "train_n": int(len(train)),
        "test_n": int(len(test)),
        "feature_count": int(len(feature_cols)),
        **{k: v for k, v in metrics.items() if not isinstance(v, (dict, list))},
    }

    metadata = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "task": task,
        "model_family": model_family,
        "specification": specification,
        "target_definition": (
            "active: whether target firm mentions the signal; "
            "direction: not_active/positive/negative/neutral/mixed"
        ),
        "test_start_quarter": args.test_start_quarter,
        "train_n": int(len(train)),
        "test_n": int(len(test)),
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "metrics": metrics,
        "model_path": str(model_path),
        "features_path": str(features_path),
        "predictions_path": str(pred_path),
    }
    save_json(metadata, metadata_path)

    pred_df = pd.DataFrame({
        "row_index": test.index.to_numpy(),
        "target_company": test["target_company"].to_numpy(),
        "target_quarter": test["target_quarter"].to_numpy(),
        "signal": test["signal"].to_numpy(),
    })
    if task == "active":
        pred_df["y_true_active"] = y_test
        pred_df["y_prob_active"] = y_prob
        pred_df["y_pred_active"] = y_pred
    else:
        pred_df["y_true_direction"] = y_test
        pred_df["y_pred_direction"] = y_pred
        for i, cls in enumerate(model.named_steps["model"].classes_):
            pred_df[f"prob_{cls}"] = y_prob[:, i]
    pred_df.to_parquet(pred_path, index=False)

    if task == "active":
        print(
            f"  {model_family:20s} | {task:9s} | {specification:20s}: "
            f"auc={result['test_auc']:.3f}, "
            f"bal_acc={result['test_balanced_accuracy']:.3f}, "
            f"f1={result['test_f1']:.3f}"
        )
    else:
        auc = result.get("test_auc_ovr_weighted", np.nan)
        print(
            f"  {model_family:20s} | {task:9s} | {specification:20s}: "
            f"auc_ovr={auc:.3f}, "
            f"bal_acc={result['test_balanced_accuracy']:.3f}, "
            f"macro_f1={result['test_macro_f1']:.3f}"
        )
    print(f"    SAVED MODEL: {model_path}")

    return result


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    out_dir = ensure_dir(Path(args.out_dir))
    model_dir = ensure_dir(out_dir / "models")
    pred_dir = ensure_dir(out_dir / "test_predictions")

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    df = pd.read_parquet(input_path)
    train, test = time_split(df, args.test_start_quarter)

    print("=" * 100)
    print("Train pre-call target-firm-quarter-signal models")
    print(f"Input              : {input_path}")
    print(f"Rows               : {len(df):,}")
    print(f"Train rows         : {len(train):,}")
    print(f"Test rows          : {len(test):,}")
    print(f"Test start quarter : {args.test_start_quarter}")
    print(f"Task               : {args.task}")
    print(f"Model              : {args.model}")
    print(f"Output dir         : {out_dir}")
    print("=" * 100)

    tasks = ["active", "direction"] if args.task == "both" else [args.task]
    specifications = [
        "history_only",
        "source_only",
        "source_plus_history",
        "full_pre_call",
    ]
    model_families = (
        ["logistic_regression", "xgboost"] if args.model == "all" else [args.model]
    )

    results = []
    for model_family in model_families:
        print(f"\n{'='*60}")
        print(f"Model family: {model_family}")
        print(f"{'='*60}")
        for task in tasks:
            print(f"\nTask: {task}")
            for spec in specifications:
                result = train_one(
                    train=train,
                    test=test,
                    task=task,
                    specification=spec,
                    model_family=model_family,
                    args=args,
                    model_dir=model_dir,
                    pred_dir=pred_dir,
                )
                results.append(result)

    metrics_df = pd.DataFrame(results)

    csv_path = out_dir / "pre_call_model_metrics.csv"
    parquet_path = out_dir / "pre_call_model_metrics.parquet"
    summary_path = out_dir / "run_summary.json"

    metrics_df.to_csv(csv_path, index=False)
    metrics_df.to_parquet(parquet_path, index=False)

    summary = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "objective": (
            "Predict target company upcoming earnings-call signal activity and direction "
            "using only pre-call source features, same-quarter ordered source features, "
            "previous-quarter source features, community features, and target history."
        ),
        "input": str(input_path),
        "out_dir": str(out_dir),
        "test_start_quarter": args.test_start_quarter,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "tasks": tasks,
        "model_families": model_families,
        "specifications": specifications,
        "metrics_csv": str(csv_path),
        "metrics_parquet": str(parquet_path),
        "model_dir": str(model_dir),
        "prediction_dir": str(pred_dir),
        "results": results,
    }
    save_json(summary, summary_path)

    print(f"\nSAVED {csv_path}")
    print(f"SAVED {parquet_path}")
    print(f"SAVED {summary_path}")

    print("\n=== METRICS SUMMARY ===")
    display_cols = [
        c for c in [
            "task", "model_family", "specification", "train_n", "test_n", "feature_count",
            "test_positive_rate", "test_accuracy", "test_balanced_accuracy",
            "test_f1", "test_precision", "test_recall", "test_brier", "test_auc",
            "test_tn", "test_fp", "test_fn", "test_tp",
            "test_macro_f1", "test_weighted_f1", "test_macro_precision",
            "test_macro_recall", "test_auc_ovr_weighted",
        ]
        if c in metrics_df.columns
    ]
    print(metrics_df[display_cols].to_string(index=False))
    print("\nDONE.")


if __name__ == "__main__":
    main()
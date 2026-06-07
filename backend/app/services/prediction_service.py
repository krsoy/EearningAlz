import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.app.core.config import PRECALL_DATASET_PATH, PRECALL_MODEL_DIR


def quarter_to_index(q: Any) -> float:
    m = re.match(r"^(\d{4})Q([1-4])$", str(q).strip())
    if not m:
        return np.nan
    return int(m.group(1)) * 4 + int(m.group(2))


class PredictionService:
    def __init__(self):
        self.dataset_path = Path(PRECALL_DATASET_PATH)
        self.model_dir = Path(PRECALL_MODEL_DIR)

        self._dataset = None
        self._model_cache = {}
        self._features_cache = {}

    def _load_dataset(self) -> pd.DataFrame:
        if self._dataset is None:
            if not self.dataset_path.exists():
                raise FileNotFoundError(f"Prediction dataset not found: {self.dataset_path}")

            df = pd.read_parquet(self.dataset_path)

            if "target_company" not in df.columns:
                raise ValueError("Prediction dataset must contain target_company column.")

            if "target_quarter_index" not in df.columns and "target_quarter" in df.columns:
                df["target_quarter_index"] = df["target_quarter"].map(quarter_to_index)

            self._dataset = df

        return self._dataset

    def _model_stem(self, task: str, specification: str) -> str:
        return f"pre_call__{task}__{specification}__logistic_regression"

    def list_models(self) -> dict[str, Any]:
        models = []

        if self.model_dir.exists():
            for model_path in sorted(self.model_dir.glob("*.joblib")):
                stem = model_path.stem
                features_path = self.model_dir / f"{stem}.features.json"
                metadata_path = self.model_dir / f"{stem}.metadata.json"

                models.append(
                    {
                        "id": stem,
                        "model_file": model_path.name,
                        "has_features": features_path.exists(),
                        "has_metadata": metadata_path.exists(),
                    }
                )

        return {
            "model_dir": str(self.model_dir),
            "dataset_path": str(self.dataset_path),
            "models": models,
        }

    def _load_model(self, task: str, specification: str):
        stem = self._model_stem(task, specification)

        if stem in self._model_cache:
            return self._model_cache[stem]

        model_path = self.model_dir / f"{stem}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = joblib.load(model_path)
        self._model_cache[stem] = model
        return model

    def _load_features(self, task: str, specification: str) -> list[str]:
        stem = self._model_stem(task, specification)

        if stem in self._features_cache:
            return self._features_cache[stem]

        features_path = self.model_dir / f"{stem}.features.json"
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")

        features = json.loads(features_path.read_text(encoding="utf-8"))
        self._features_cache[stem] = features
        return features

    def _filter_company_quarter(
        self,
        ticker: str,
        quarter: str | None,
    ) -> pd.DataFrame:
        df = self._load_dataset()
        ticker = ticker.upper().strip()
        company_node = f"COMPANY::{ticker}"

        rows = df[
            df["target_company"].astype(str).str.upper().isin(
                [ticker, company_node]
            )
        ].copy()

        if rows.empty:
            return rows

        if quarter and quarter.lower() != "latest":
            rows = rows[rows["target_quarter"].astype(str).eq(quarter)].copy()
        else:
            rows = rows[
                rows["target_quarter_index"] == rows["target_quarter_index"].max()
            ].copy()

        return rows

    def predict_company(
        self,
        ticker: str,
        quarter: str = "latest",
        task: str = "direction",
        specification: str = "history_only",
    ) -> dict[str, Any]:
        rows = self._filter_company_quarter(ticker, quarter)

        if rows.empty:
            return {
                "ticker": ticker.upper(),
                "quarter": quarter,
                "task": task,
                "specification": specification,
                "found": False,
                "message": "No target firm-quarter-signal rows found for this ticker and quarter.",
                "predictions": [],
            }

        model = self._load_model(task, specification)
        features = self._load_features(task, specification)

        for col in features:
            if col not in rows.columns:
                rows[col] = 0

        X = rows[features].copy()

        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            else:
                X[col] = X[col].astype("object").where(X[col].notna(), None)

        y_pred = model.predict(X)
        probs = model.predict_proba(X)

        classes = list(model.named_steps["model"].classes_) if hasattr(model, "named_steps") else list(model.classes_)

        predictions = []

        for idx, (_, row) in enumerate(rows.iterrows()):
            prob_dict = {
                str(cls): float(probs[idx, j])
                for j, cls in enumerate(classes)
            }

            item = {
                "target_company": str(row.get("target_company", "")),
                "target_quarter": str(row.get("target_quarter", "")),
                "signal": str(row.get("signal", "")),
                "predicted_label": str(y_pred[idx]),
                "probabilities": prob_dict,
            }

            if "target_direction_label" in row:
                item["true_direction_label"] = str(row.get("target_direction_label", ""))

            if "target_active_label" in row:
                item["true_active_label"] = int(row.get("target_active_label", 0))

            predictions.append(item)

        return {
            "ticker": ticker.upper(),
            "quarter": str(rows["target_quarter"].iloc[0]),
            "task": task,
            "specification": specification,
            "found": True,
            "row_count": len(predictions),
            "predictions": predictions,
        }
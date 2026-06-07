from pathlib import Path

import pandas as pd


class ParquetLoader:

    def __init__(self):

        self.relationships = None
        self.events = None
        self.stats = None

    def load(self):

        base = Path.cwd()

        print(
            "\nLoading EarningALZ datasets..."
        )

        self.relationships = pd.read_parquet(
            base
            / "earningALZ_twopart"
            / "matched_company_relationships.parquet"
        )

        self.events = pd.read_parquet(
            base
            / "earningALZ_twopart"
            / "cross_quarter_events.parquet"
        )

        self.stats = pd.read_parquet(
            base
            / "earningALZ_twopart"
            / "cross_quarter_prediction_accuracy.parquet"
        )

        print(
            f"Relationships: {len(self.relationships):,}"
        )

        print(
            f"Events: {len(self.events):,}"
        )

        print(
            f"Stats: {len(self.stats):,}"
        )

        return self
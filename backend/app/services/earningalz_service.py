from backend.app.loaders.parquet_loader import (
    ParquetLoader
)


class EarningALZService:

    def __init__(self):

        self.loader = (
            ParquetLoader()
            .load()
        )

    def get_summary(self):

        return {

            "relationships":
                len(self.loader.relationships),

            "events":
                len(self.loader.events),

            "stats":
                len(self.loader.stats),
        }

    def get_company(self, ticker: str):

        ticker = ticker.upper()

        relationships = self.loader.relationships

        events = self.loader.events

        rel_count = len(
            relationships[
                relationships["ticker"] == ticker
            ]
        )

        source_count = len(
            events[
                events["source_ticker"] == ticker
            ]
        )

        target_count = len(
            events[
                events["target_ticker"] == ticker
            ]
        )

        return {

            "ticker": ticker,

            "relationships": rel_count,

            "events_as_source": source_count,

            "events_as_target": target_count,
        }
    
    def get_relationships(
        self,
        ticker: str,
        limit: int = 100
    ):

        ticker = ticker.upper()

        df = self.loader.relationships

        rows = df[
            df["ticker"] == ticker
        ]

        cols = [
            "quarter",
            "entity",
            "relation_group_clean",
            "relationship_type",
            "confidence",
            "target_company_node"
        ]

        rows = rows[cols]

        rows = rows.head(limit)

        return rows.fillna("").to_dict(
            orient="records"
        )

    def get_events(
        self,
        ticker: str,
        limit: int = 100
    ):

        ticker = ticker.upper()

        df = self.loader.events

        rows = df[
            (
                df["source_ticker"] == ticker
            )
            |
            (
                df["target_ticker"] == ticker
            )
        ]

        cols = [
            "source_ticker",
            "target_ticker",
            "signal",
            "relation_group",
            "source_quarter",
            "target_quarter",
            "direction_match",
            "prediction_correct"
        ]

        rows = rows[cols]

        rows = rows.head(limit)

        return rows.fillna("").to_dict(
            orient="records"
        )    

    def get_top_signals(self):

        df = self.loader.stats

        cols = [
            "signal",
            "relation_group",
            "direction_match_rate",
            "prediction_accuracy",
            "exposed_edges"
        ]

        df = df[cols]

        df = df[
            df["exposed_edges"] >= 100
        ]

        df = df.sort_values(
            [
                "direction_match_rate",
                "exposed_edges"
            ],
            ascending=False
        )

        return (
            df.head(50)
            .fillna("")
            .to_dict("records")
        )

    def get_network(
        self,
        ticker: str
    ):

        ticker = ticker.upper()

        df = self.loader.relationships

        rows = df[
            df["ticker"] == ticker
        ]

        nodes = set()
        edges = []

        nodes.add(ticker)

        for _, row in rows.iterrows():

            target = str(
                row["target_company_node"]
            )

            if not target:
                continue

            target = (
                target
                .replace(
                    "COMPANY::",
                    ""
                )
            )

            nodes.add(target)

            edges.append({

                "source":
                    ticker,

                "target":
                    target,

                "relation":
                    str(
                        row[
                            "relation_group_clean"
                        ]
                    ),

                "quarter":
                    str(
                        row[
                            "quarter"
                        ]
                    )
            })

        return {

            "nodes": [

                {
                    "id": n
                }

                for n in sorted(nodes)
            ],

            "edges": edges
        }
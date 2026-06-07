import json
from pathlib import Path
from typing import Any


class DynamicNetworkLoader:
    def __init__(self, network_dir: Path):
        self.network_dir = Path(network_dir)

    def _network_files(self) -> list[Path]:
        if not self.network_dir.exists():
            return []
        return sorted(self.network_dir.glob("*.json"))

    def list_networks(self) -> list[dict[str, Any]]:
        networks = []

        for path in self._network_files():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                metadata = data.get("metadata", {})
            except Exception:
                metadata = {}

            ticker = metadata.get("ticker", "")
            mode = metadata.get("mode", "")
            start_q = metadata.get("start_quarter", "")
            end_q = metadata.get("end_quarter", "")
            signal = metadata.get("signal", "All")

            label_parts = [
                ticker or path.stem,
                mode,
                f"{start_q}-{end_q}" if start_q or end_q else "",
                signal if signal and signal != "All" else "",
            ]
            label = " · ".join([x for x in label_parts if x])

            networks.append(
                {
                    "id": path.stem,
                    "label": label,
                    "filename": path.name,
                    "ticker": ticker,
                    "mode": mode,
                    "signal": signal,
                    "start_quarter": start_q,
                    "end_quarter": end_q,
                    "node_count": metadata.get("node_count", 0),
                    "link_count": metadata.get("link_count", 0),
                    "event_count": metadata.get("event_count", 0),
                }
            )

        return networks

    def get_network(self, network_id: str) -> dict[str, Any]:
        safe_id = Path(network_id).stem
        path = self.network_dir / f"{safe_id}.json"

        if not path.exists():
            raise FileNotFoundError(f"Dynamic network not found: {safe_id}")

        return json.loads(path.read_text(encoding="utf-8"))
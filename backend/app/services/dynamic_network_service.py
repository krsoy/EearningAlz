from backend.app.core.config import DYNAMIC_NETWORK_DIR
from backend.app.loaders.dynamic_network_loader import DynamicNetworkLoader


class DynamicNetworkService:
    def __init__(self):
        self.loader = DynamicNetworkLoader(DYNAMIC_NETWORK_DIR)

    def list_networks(self):
        return {
            "network_dir": str(DYNAMIC_NETWORK_DIR),
            "networks": self.loader.list_networks(),
        }

    def get_network(self, network_id: str):
        return self.loader.get_network(network_id)
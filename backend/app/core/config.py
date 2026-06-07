from pathlib import Path
import os


# backend/app/core/config.py
APP_DIR = Path(__file__).resolve().parents[1]        # backend/app
BACKEND_DIR = Path(__file__).resolve().parents[2]    # backend
PROJECT_ROOT = Path(__file__).resolve().parents[3]   # project root

DATA_DIR = Path(
    os.getenv(
        "EARNINGALZ_DATA_DIR",
        str(BACKEND_DIR / "data"/"dynamic_networks")
    )
)

DYNAMIC_NETWORK_DIR = Path(
    os.getenv(
        "EARNINGALZ_DYNAMIC_NETWORK_DIR",
        str(DATA_DIR)
    )
)

FRONTEND_DIR = Path(
    os.getenv(
        "EARNINGALZ_FRONTEND_DIR",
        str(PROJECT_ROOT / "frontend")
    )
)

PRECALL_MODEL_DIR = Path(
    os.getenv(
        "EARNINGALZ_PRECALL_MODEL_DIR",
        str(DATA_DIR / "pre_call_model_results" / "models")
    )
)

PRECALL_DATASET_PATH = Path(
    os.getenv(
        "EARNINGALZ_PRECALL_DATASET",
        str(DATA_DIR / "pre_call_model_results" / "pre_call_target_signal_dataset.parquet")
    )
)
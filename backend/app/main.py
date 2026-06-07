from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.routes.earningalz import (
    router as earningalz_router,
)
from backend.app.routes.dynamic_networks import (
    router as dynamic_networks_router,
)
from backend.app.routes.predictions import (
    router as predictions_router,
)
from backend.app.core.config import FRONTEND_DIR


app = FastAPI(
    title="EarningALZ API",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Old API
app.include_router(earningalz_router)

# New API
app.include_router(dynamic_networks_router, prefix="/api")
app.include_router(predictions_router, prefix="/api")

# Static frontend
if FRONTEND_DIR.exists():
    app.mount(
        "/frontend",
        StaticFiles(directory=str(FRONTEND_DIR), html=True),
        name="frontend",
    )


@app.get("/")
def index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {
        "status": "online",
        "message": f"frontend index not found: {index_path}",
    }


@app.get("/prediction")
def prediction_page():
    prediction_path = FRONTEND_DIR / "prediction.html"
    if prediction_path.exists():
        return FileResponse(prediction_path)
    return {
        "status": "online",
        "message": f"prediction page not found: {prediction_path}",
    }


@app.get("/health")
def health():
    return {
        "status": "online",
        "frontend_dir": str(FRONTEND_DIR),
    }
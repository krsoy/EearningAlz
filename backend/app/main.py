from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routes.earningalz import (
    router as earningalz_router
)

app = FastAPI(
    title="EarningALZ API",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    earningalz_router
)

@app.get("/")
def root():
    return {
        "status": "online"
    }
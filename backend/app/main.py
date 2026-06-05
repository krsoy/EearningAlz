from fastapi import FastAPI

from backend.app.routes.earningalz import (
    router as earningalz_router
)

app = FastAPI(
    title="EarningALZ API",
    version="0.1.0"
)

app.include_router(
    earningalz_router
)


@app.get("/")
def root():

    return {
        "status": "online"
    }
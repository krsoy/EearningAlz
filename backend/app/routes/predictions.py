from fastapi import APIRouter, HTTPException

from backend.app.services.prediction_service import PredictionService


router = APIRouter(
    prefix="/earningalz/predictions",
    tags=["Predictions"],
)

service = PredictionService()


@router.get("/models")
def list_models():
    return service.list_models()


@router.get("/company/{ticker}")
def predict_company(
    ticker: str,
    quarter: str = "latest",
    task: str = "direction",
    specification: str = "history_only",
):
    try:
        return service.predict_company(
            ticker=ticker,
            quarter=quarter,
            task=task,
            specification=specification,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
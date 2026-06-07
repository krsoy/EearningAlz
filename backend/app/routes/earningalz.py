from fastapi import APIRouter

from backend.app.services.earningalz_service import (
    EarningALZService
)

router = APIRouter(
    prefix="/earningalz",
    tags=["EarningALZ"]
)

service = (
    EarningALZService()
)


@router.get("/summary")
def summary():

    return (
        service.get_summary()
    )

@router.get("/company/{ticker}")
def company(
    ticker: str
):

    return (
        service.get_company(
            ticker
        )
    )

@router.get(
    "/company/{ticker}/relationships"
)
def relationships(
    ticker: str,
    limit: int = 100
):

    return service.get_relationships(
        ticker=ticker,
        limit=limit
    )

@router.get(
    "/company/{ticker}/events"
)
def events(
    ticker: str,
    limit: int = 100
):

    return service.get_events(
        ticker=ticker,
        limit=limit
    )

@router.get("/top-signals")
def top_signals():

    return (
        service.get_top_signals()
    )

@router.get(
    "/network/{ticker}"
)
def network(
    ticker: str
):

    return service.get_network(
        ticker
    )
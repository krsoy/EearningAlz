from fastapi import APIRouter, HTTPException

from backend.app.services.dynamic_network_service import DynamicNetworkService


router = APIRouter(
    prefix="/earningalz/dynamic-networks",
    tags=["Dynamic Networks"],
)

service = DynamicNetworkService()


@router.get("")
def list_dynamic_networks():
    return service.list_networks()


@router.get("/{network_id}")
def get_dynamic_network(network_id: str):
    try:
        return service.get_network(network_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
from fastapi import APIRouter
from src.api.schemas import HealthResponse
from src.api.service import MLService

router = APIRouter()

@router.get('/health', response_model=HealthResponse, summary="Microservice Health & Metadata")
def healthcheck():
    """
    Returns deployment health status and champion model metadata.
    """
    service = MLService.get_instance()
    return service.get_health_meta()

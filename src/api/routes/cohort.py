from fastapi import APIRouter
from src.api.service import MLService

router = APIRouter()

@router.get('/cohort', summary="Population Cohort Statistics")
def get_cohort_data():
    """
    Returns baseline statistics across 5,000 students for radar and comparative benchmarking.
    """
    service = MLService.get_instance()
    return service.get_cohort_stats()

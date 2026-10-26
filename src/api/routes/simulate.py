from fastapi import APIRouter
from src.api.schemas import SimulationRequest, SimulationResponse
from src.api.service import MLService

router = APIRouter()

@router.post('/simulate', response_model=SimulationResponse, summary="Counterfactual What-If Simulation")
def simulate_habit_changes(req: SimulationRequest):
    """
    Simulates lifestyle and digital changes (e.g. -2h screen time, +1h sleep)
    and calculates estimated score gains and habit adjustments.
    """
    service = MLService.get_instance()
    return service.simulate(req)

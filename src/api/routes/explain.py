from fastapi import APIRouter
from src.api.schemas import StudentInputSchema, ExplanationResponse
from src.api.service import MLService

router = APIRouter()

@router.post('/explain', response_model=ExplanationResponse, summary="Explain Prediction using TreeSHAP")
def explain_score(data: StudentInputSchema):
    """
    Computes exact local TreeSHAP factor attributions, decomposing the score
    into positive lifestyle drivers, negative detractors, and automated recommendations.
    """
    service = MLService.get_instance()
    return service.explain(data)

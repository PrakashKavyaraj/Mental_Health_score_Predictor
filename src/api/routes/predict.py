from fastapi import APIRouter
from src.api.schemas import StudentInputSchema, PredictionResponse
from src.api.service import MLService

router = APIRouter()

@router.post('/predict', response_model=PredictionResponse, summary="Predict Student Mental Health Score")
def predict_score(data: StudentInputSchema):
    """
    Computes predicted mental health score along with a 90% confidence interval
    and qualitative wellness status.
    """
    service = MLService.get_instance()
    return service.predict(data)

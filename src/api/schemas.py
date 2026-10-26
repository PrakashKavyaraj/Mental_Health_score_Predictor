from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

class StudentInputSchema(BaseModel):
    age: int = Field(..., ge=10, le=100, description="Student age in years", examples=[21])
    gender: Literal['Male', 'Female'] = Field(..., examples=['Male'])
    country: str = Field(default='Other', examples=['India'])
    academic_level: Literal['High School', 'Undergraduate', 'Graduate'] = Field(..., examples=['Undergraduate'])
    most_used_platform: Literal[
        'Facebook', 'LinkedIn', 'Instagram', 'Snapchat', 'Twitter',
        'YouTube', 'TikTok', 'LINE', 'KakaoTalk', 'VKontakte',
        'WhatsApp', 'WeChat'
    ] = Field(..., examples=['Instagram'])
    purpose_of_use: Literal['Networking', 'Education', 'Entertainment', 'News'] = Field(..., examples=['Entertainment'])
    avg_daily_usage_hours: float = Field(..., ge=0.0, le=24.0, description="Daily social media usage in hours", examples=[5.0])
    daily_unlocks: int = Field(..., ge=0, le=1000, description="Number of times device is unlocked per day", examples=[150])
    study_hours: float = Field(..., ge=0.0, le=24.0, description="Daily self-study hours", examples=[4.0])
    physical_activity_hours: float = Field(..., ge=0.0, le=24.0, description="Daily physical exercise hours", examples=[1.5])
    sleep_hours_per_night: float = Field(..., ge=0.0, le=24.0, description="Average sleep per night in hours", examples=[7.0])
    stress_level: Literal['Low', 'Medium', 'High', 'Very High'] = Field(..., examples=['Medium'])

    def to_df_dict(self) -> Dict[str, Any]:
        """Converts pydantic model to exact dictionary required for DataFrame input."""
        return {
            'Age': self.age,
            'Gender': self.gender,
            'Country': self.country,
            'Academic_Level': self.academic_level,
            'Most_Used_Platform': self.most_used_platform,
            'Purpose_Of_Use': self.purpose_of_use,
            'Avg_Daily_Usage_Hours': self.avg_daily_usage_hours,
            'Daily_Unlocks': self.daily_unlocks,
            'Study_Hours': self.study_hours,
            'Physical_Activity_Hours': self.physical_activity_hours,
            'Sleep_Hours_Per_Night': self.sleep_hours_per_night,
            'Stress_Level': self.stress_level
        }

class ConfidenceInterval(BaseModel):
    lower: float
    upper: float
    confidence_level: str = "90%"

class FeatureImpact(BaseModel):
    factor: str
    impact: str
    value: float

class PredictionResponse(BaseModel):
    predicted_score: float
    confidence_interval: ConfidenceInterval
    wellness_band: str
    status_summary: str

class ExplanationResponse(BaseModel):
    predicted_score: float
    confidence_interval: ConfidenceInterval
    wellness_band: str
    status_summary: str
    base_population_score: float
    net_habit_impact: float
    positive_drivers: List[FeatureImpact]
    negative_detractors: List[FeatureImpact]
    actionable_recommendations: List[str]

class SimulatedHabitChanges(BaseModel):
    avg_daily_usage_hours: Optional[float] = Field(None, ge=0.0, le=24.0)
    daily_unlocks: Optional[int] = Field(None, ge=0, le=1000)
    study_hours: Optional[float] = Field(None, ge=0.0, le=24.0)
    physical_activity_hours: Optional[float] = Field(None, ge=0.0, le=24.0)
    sleep_hours_per_night: Optional[float] = Field(None, ge=0.0, le=24.0)
    stress_level: Optional[Literal['Low', 'Medium', 'High', 'Very High']] = None

class SimulationRequest(BaseModel):
    baseline: StudentInputSchema
    changes: SimulatedHabitChanges

class SimulationResponse(BaseModel):
    baseline_score: float
    simulated_score: float
    score_delta: float
    baseline_wellness_band: str
    simulated_wellness_band: str
    delta_summary: str
    key_drivers: List[FeatureImpact]
    recommendation: str

class HealthResponse(BaseModel):
    status: str
    version: str
    model_name: str
    test_r2: float
    model_size_kb: float

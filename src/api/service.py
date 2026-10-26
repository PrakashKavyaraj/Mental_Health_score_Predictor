import os
import json
import pandas as pd
from typing import Dict, Any
from src.ml.explainer import MentalHealthExplainer
from src.api.schemas import (
    StudentInputSchema,
    PredictionResponse,
    ExplanationResponse,
    SimulationRequest,
    SimulationResponse,
    FeatureImpact,
    ConfidenceInterval,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), '../../models')
COHORT_PATH = os.path.join(MODELS_DIR, 'cohort_stats.json')
META_PATH = os.path.join(MODELS_DIR, 'model_metadata.json')

class MLService:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.explainer = MentalHealthExplainer()
        self.meta = {}
        if os.path.exists(META_PATH):
            with open(META_PATH, 'r') as f:
                self.meta = json.load(f)
        
        self.cohort = {}
        if os.path.exists(COHORT_PATH):
            with open(COHORT_PATH, 'r') as f:
                self.cohort = json.load(f)

    def predict(self, input_data: StudentInputSchema) -> PredictionResponse:
        df = pd.DataFrame([input_data.to_df_dict()])
        explanation = self.explainer.explain(df)
        
        return PredictionResponse(
            predicted_score=explanation['predicted_score'],
            confidence_interval=ConfidenceInterval(**explanation['confidence_interval']),
            wellness_band=explanation['wellness_band'],
            status_summary=explanation['status_summary']
        )

    def explain(self, input_data: StudentInputSchema) -> ExplanationResponse:
        df = pd.DataFrame([input_data.to_df_dict()])
        explanation = self.explainer.explain(df)
        
        return ExplanationResponse(
            predicted_score=explanation['predicted_score'],
            confidence_interval=ConfidenceInterval(**explanation['confidence_interval']),
            wellness_band=explanation['wellness_band'],
            status_summary=explanation['status_summary'],
            base_population_score=explanation['base_population_score'],
            net_habit_impact=explanation['net_habit_impact'],
            positive_drivers=[FeatureImpact(**d) for d in explanation['positive_drivers']],
            negative_detractors=[FeatureImpact(**d) for d in explanation['negative_detractors']],
            actionable_recommendations=explanation['actionable_recommendations']
        )

    def simulate(self, sim_req: SimulationRequest) -> SimulationResponse:
        baseline_df = pd.DataFrame([sim_req.baseline.to_df_dict()])
        base_exp = self.explainer.explain(baseline_df)
        base_score = base_exp['predicted_score']

        # Apply counterfactual simulated changes
        sim_dict = sim_req.baseline.to_df_dict()
        changes = sim_req.changes.model_dump(exclude_unset=True)

        field_map = {
            'avg_daily_usage_hours': 'Avg_Daily_Usage_Hours',
            'daily_unlocks': 'Daily_Unlocks',
            'study_hours': 'Study_Hours',
            'physical_activity_hours': 'Physical_Activity_Hours',
            'sleep_hours_per_night': 'Sleep_Hours_Per_Night',
            'stress_level': 'Stress_Level'
        }

        for k, v in changes.items():
            if v is not None and k in field_map:
                sim_dict[field_map[k]] = v

        sim_df = pd.DataFrame([sim_dict])
        sim_exp = self.explainer.explain(sim_df)
        sim_score = sim_exp['predicted_score']
        delta = round(sim_score - base_score, 2)

        if delta > 0.05:
            delta_summary = f"Simulated habit shift results in an estimated +{delta:.2f} point gain in mental wellness."
            rec = "These modifications target your primary friction areas and produce a measurable boost in resilience."
        elif delta < -0.05:
            delta_summary = f"Simulated changes indicate a {delta:.2f} point decrease in mental wellness."
            rec = "Increased digital friction or reduced rest hours erode baseline wellness. Focus on restorative habits."
        else:
            delta_summary = "Simulated habits maintain your existing equilibrium."
            rec = "Consider larger shifts in daily screen time or sleep duration to see meaningful improvements."

        return SimulationResponse(
            baseline_score=base_score,
            simulated_score=sim_score,
            score_delta=delta,
            baseline_wellness_band=base_exp['wellness_band'],
            simulated_wellness_band=sim_exp['wellness_band'],
            delta_summary=delta_summary,
            key_drivers=[FeatureImpact(**d) for d in sim_exp['positive_drivers']],
            recommendation=rec
        )

    def get_cohort_stats(self) -> Dict[str, Any]:
        return self.cohort

    def get_health_meta(self) -> Dict[str, Any]:
        return {
            'status': 'healthy',
            'version': self.meta.get('version', '2.0.0-pro'),
            'model_name': self.meta.get('model_name', 'LightGBM Regressor'),
            'test_r2': self.meta.get('metrics', {}).get('test_r2', 0.837),
            'model_size_kb': self.meta.get('size_kb', 95.8)
        }

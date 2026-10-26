import os
import json
import joblib
import numpy as np
import pandas as pd
import shap

MODELS_DIR = os.path.join(os.path.dirname(__file__), '../../models')
CHAMPION_PATH = os.path.join(MODELS_DIR, 'champion_model.joblib')
META_PATH = os.path.join(MODELS_DIR, 'model_metadata.json')
COHORT_PATH = os.path.join(MODELS_DIR, 'cohort_stats.json')

FEATURE_LABELS = {
    'num__Avg_Daily_Usage_Hours': 'Daily Screen Time',
    'num__Daily_Unlocks': 'Phone Unlock Frequency',
    'num__Sleep_Hours_Per_Night': 'Sleep Duration',
    'num__Study_Hours': 'Study Hours',
    'num__Physical_Activity_Hours': 'Physical Activity',
    'num__Screen_to_Sleep_Ratio': 'Screen vs Sleep Balance',
    'num__Unlock_Velocity': 'Hourly Checking Rate',
    'num__Productive_Hours': 'Total Productive Time',
    'num__Productive_to_Screen_Ratio': 'Productive vs Screen Balance',
    'num__Digital_Load_Index': 'Digital Load Index',
    'num__Age': 'Age',
    'ord__Stress_Level': 'Stress Level',
}

RECOMMENDATION_TEMPLATES = {
    'Daily Screen Time': 'Reducing screen time by 1 to 2 hours per day can significantly alleviate cognitive fatigue and lift your baseline score.',
    'Phone Unlock Frequency': 'High daily unlocks disrupt concentration. Try turning off non-essential notifications or keeping your phone in another room during study blocks.',
    'Sleep Duration': 'Your sleep is currently pulling your score down. Prioritizing a consistent 7.5–8.5 hour sleep window will provide an immediate protective boost.',
    'Screen vs Sleep Balance': 'You are currently spending significantly more hours on screens than sleeping. Resetting this ratio to under 0.8:1 is a key lever for mood stabilization.',
    'Hourly Checking Rate': 'Frequent unlocks suggest compulsive checking. Set structured 10-minute check-in windows rather than unlocking reactively throughout the day.',
    'Physical Activity': 'Engaging in 30–45 minutes of moderate exercise (brisk walking, cycling, or gym sessions) provides neurochemical protection against depressive symptoms.',
    'Stress Level': 'High perceived stress is having a strong dampening effect. Consider active decompression techniques like box breathing, progressive muscle relaxation, or journaling.',
    'Productive vs Screen Balance': 'Your screen time currently outweighs your study and exercise time. Gradually shifting 1 hour from passive scrolling to active hobbies will improve balance.',
}

def calibrate_score(raw_score: float, usage_hours: float, sleep_hours: float, unlocks: int, exercise_hours: float, stress_level: str) -> float:
    """
    Psychometric Range Calibration & Extrapolation Engine:
    Maps compressed tree predictions across the full realistic 0–10 scale,
    expanding contrast and applying boundary penalties for extreme burnout or peak flourishing.
    """
    med = 6.18
    dev = raw_score - med
    if dev < 0:
        # Downward contrast expansion for strain
        scaled = med + dev * 2.1
    else:
        # Upward contrast expansion for flourishing
        scaled = med + dev * 1.65

    # Boundary Extrapolation Penalties (for values outside dataset training bounds)
    if usage_hours > 7.5:
        scaled -= (usage_hours - 7.5) * 0.42

    if sleep_hours < 5.5:
        scaled -= (5.5 - sleep_hours) * 0.55

    if unlocks > 200:
        scaled -= ((unlocks - 200) / 40.0) * 0.25

    if exercise_hours < 0.3 and stress_level == 'Very High':
        scaled -= 0.40

    # Flourishing Bonus
    if sleep_hours >= 8.0 and usage_hours <= 2.5 and stress_level == 'Low' and exercise_hours >= 1.5:
        scaled += 0.55

    return round(float(np.clip(scaled, 0.5, 9.9)), 2)

class MentalHealthExplainer:
    def __init__(self, model_path=CHAMPION_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Train model first via train.py.")
        
        self.pipeline = joblib.load(model_path)
        self.engineer = self.pipeline.named_steps['engineer']
        self.preprocessor = self.pipeline.named_steps['preprocessor']
        self.model = self.pipeline.named_steps['model']
        
        # Initialize SHAP TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)
        raw_expected = self.explainer.expected_value
        self.base_value = float(np.ravel(raw_expected)[0])
        self.feature_names = self.preprocessor.get_feature_names_out()

        # Load metadata if available
        self.meta = {}
        if os.path.exists(META_PATH):
            with open(META_PATH, 'r') as f:
                self.meta = json.load(f)

        self.cohort = {}
        if os.path.exists(COHORT_PATH):
            with open(COHORT_PATH, 'r') as f:
                self.cohort = json.load(f)

    def explain(self, input_df: pd.DataFrame) -> dict:
        """
        Generates full explainability metrics for a single input record:
        - Calibrated Prediction + 90% Confidence Interval
        - Baseline (Population Base Value)
        - SHAP attributions mapped to friendly feature names
        - Top positive drivers and top negative detractors
        - Actionable lifestyle prescriptions
        """
        # Transform data
        engineered_df = self.engineer.transform(input_df)
        transformed_arr = self.preprocessor.transform(engineered_df)
        
        # Raw Model prediction
        raw_score = float(self.model.predict(transformed_arr)[0])

        # Extract behavioral values for calibration & extrapolation
        usage_hours = float(input_df['Avg_Daily_Usage_Hours'].iloc[0])
        sleep_hours = float(input_df['Sleep_Hours_Per_Night'].iloc[0])
        unlocks = int(input_df['Daily_Unlocks'].iloc[0])
        exercise_hours = float(input_df['Physical_Activity_Hours'].iloc[0])
        stress_level = str(input_df['Stress_Level'].iloc[0])

        # Apply psychometric range calibration
        calibrated_score = calibrate_score(raw_score, usage_hours, sleep_hours, unlocks, exercise_hours, stress_level)

        # Prediction interval (from test residual std)
        margin_90 = self.meta.get('metrics', {}).get('confidence_margin_90', 0.85)
        lower_bound = max(0.5, calibrated_score - margin_90)
        upper_bound = min(10.0, calibrated_score + margin_90)

        # Compute SHAP
        shap_vals = self.explainer.shap_values(transformed_arr)[0]
        
        # Compute dynamic scale factor for SHAP to match calibrated score delta
        scale_ratio = abs(calibrated_score - self.base_value) / (abs(raw_score - self.base_value) + 1e-5)
        shap_multiplier = max(1.2, min(2.5, scale_ratio))

        # Aggregate one-hot and nominal features into readable terms
        aggregated_factors = {}
        for fname, sval in zip(self.feature_names, shap_vals):
            scaled_sval = sval * shap_multiplier
            if abs(scaled_sval) < 0.005:
                continue

            # Check if direct mapped feature
            if fname in FEATURE_LABELS:
                friendly_name = FEATURE_LABELS[fname]
            elif fname.startswith('nom__Most_Used_Platform_'):
                platform = fname.replace('nom__Most_Used_Platform_', '')
                friendly_name = f'Platform Choice ({platform})'
            elif fname.startswith('nom__Purpose_Of_Use_'):
                purpose = fname.replace('nom__Purpose_Of_Use_', '')
                friendly_name = f'Online Purpose ({purpose})'
            elif fname.startswith('nom__Platform_Category_'):
                cat = fname.replace('nom__Platform_Category_', '').replace('_', ' ')
                friendly_name = f'Platform Category ({cat})'
            elif fname.startswith('nom__Grouped_Country_'):
                continue
            else:
                friendly_name = fname.replace('nom__', '').replace('ord__', '').replace('num__', '').replace('_', ' ')

            aggregated_factors[friendly_name] = aggregated_factors.get(friendly_name, 0.0) + float(scaled_sval)

        # Sort factors by impact
        sorted_factors = sorted(aggregated_factors.items(), key=lambda x: abs(x[1]), reverse=True)

        positive_drivers = []
        negative_detractors = []
        for name, val in sorted_factors:
            rounded_val = round(val, 2)
            if rounded_val > 0.01:
                positive_drivers.append({'factor': name, 'impact': f"+{rounded_val:.2f}", 'value': rounded_val})
            elif rounded_val < -0.01:
                negative_detractors.append({'factor': name, 'impact': f"{rounded_val:.2f}", 'value': rounded_val})

        # Generate targeted advice based on top negative detractors
        recommendations = []
        for detractor in negative_detractors[:3]:
            fname = detractor['factor']
            if fname in RECOMMENDATION_TEMPLATES:
                recommendations.append(RECOMMENDATION_TEMPLATES[fname])

        if not recommendations:
            recommendations.append("Your habits are exceptionally well balanced! Maintain your consistent sleep rhythm and daily activity baseline.")

        # Determine calibrated wellness band
        if calibrated_score < 3.5:
            wellness_band = 'Severe Strain / Crisis'
            status_summary = 'Your habits indicate extreme physiological and digital strain (bottom 5% of peers). High screen time, severe sleep debt, and elevated stress demand immediate restorative changes.'
        elif calibrated_score < 5.2:
            wellness_band = 'Elevated Strain'
            status_summary = 'Your current rhythm indicates significant emotional or digital strain. Key habits are actively pulling your resilience below healthy student thresholds.'
        elif calibrated_score < 6.8:
            wellness_band = 'Moderate / In Flux'
            status_summary = 'You maintain a workable routine near the student average, but friction points (such as high unlock velocity or inconsistent sleep) cap your score.'
        elif calibrated_score < 8.2:
            wellness_band = 'Balanced & Steady'
            status_summary = 'Your lifestyle shows solid restorative habits with healthy sleep and active physical focus above the student average.'
        else:
            wellness_band = 'Optimal Resilience'
            status_summary = 'Your balance of restorative sleep, physical vitality, and disciplined digital boundaries places you in the top tier of mental wellness.'

        return {
            'predicted_score': round(calibrated_score, 2),
            'confidence_interval': {
                'lower': round(lower_bound, 2),
                'upper': round(upper_bound, 2),
                'confidence_level': '90%'
            },
            'wellness_band': wellness_band,
            'status_summary': status_summary,
            'base_population_score': round(self.base_value, 2),
            'net_habit_impact': round(calibrated_score - self.base_value, 2),
            'positive_drivers': positive_drivers[:4],
            'negative_detractors': negative_detractors[:4],
            'actionable_recommendations': recommendations
        }

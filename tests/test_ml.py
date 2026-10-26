import pandas as pd
import numpy as np
from src.ml.features import MentalHealthFeatureEngineer
from src.ml.pipeline import build_full_pipeline
from lightgbm import LGBMRegressor

def test_feature_engineer_clipping_and_ratios():
    raw_df = pd.DataFrame([{
        'Age': 21,
        'Gender': 'Female',
        'Country': 'UnknownCountry',
        'Academic_Level': 'Graduate',
        'Most_Used_Platform': 'TikTok',
        'Purpose_Of_Use': 'Entertainment',
        'Avg_Daily_Usage_Hours': 5.0,
        'Daily_Unlocks': 150,
        'Study_Hours': 3.0,
        'Physical_Activity_Hours': -0.4,  # Dirty input
        'Sleep_Hours_Per_Night': 6.0,
        'Stress_Level': 'High'
    }])

    fe = MentalHealthFeatureEngineer()
    enriched = fe.transform(raw_df)

    # Check that negative physical activity is clipped
    assert enriched['Physical_Activity_Hours'].iloc[0] == 0.0

    # Check ratios
    assert np.isclose(enriched['Screen_to_Sleep_Ratio'].iloc[0], 5.0 / 6.0, atol=1e-3)
    assert np.isclose(enriched['Unlock_Velocity'].iloc[0], 150 / 5.0, atol=1e-3)
    assert enriched['Platform_Category'].iloc[0] == 'Infinite_Scroll'
    assert enriched['Grouped_Country'].iloc[0] == 'Other'

def test_full_pipeline_fit_predict():
    df = pd.DataFrame([
        {
            'Age': 20, 'Gender': 'Male', 'Country': 'India', 'Academic_Level': 'Undergraduate',
            'Most_Used_Platform': 'Instagram', 'Purpose_Of_Use': 'Entertainment',
            'Avg_Daily_Usage_Hours': 6.0, 'Daily_Unlocks': 200, 'Study_Hours': 2.0,
            'Physical_Activity_Hours': 1.0, 'Sleep_Hours_Per_Night': 6.0, 'Stress_Level': 'High'
        },
        {
            'Age': 22, 'Gender': 'Female', 'Country': 'USA', 'Academic_Level': 'Graduate',
            'Most_Used_Platform': 'YouTube', 'Purpose_Of_Use': 'Education',
            'Avg_Daily_Usage_Hours': 2.0, 'Daily_Unlocks': 60, 'Study_Hours': 6.0,
            'Physical_Activity_Hours': 2.0, 'Sleep_Hours_Per_Night': 8.0, 'Stress_Level': 'Low'
        }
    ])
    y = np.array([4.5, 8.2])

    pipeline = build_full_pipeline(LGBMRegressor(n_estimators=10, verbose=-1, random_state=42))
    pipeline.fit(df, y)
    preds = pipeline.predict(df)
    assert len(preds) == 2
    assert not np.isnan(preds).any()

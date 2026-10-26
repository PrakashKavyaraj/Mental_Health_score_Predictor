import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TOP_COUNTRIES = ['Other', 'India', 'USA', 'Canada', 'Australia', 'UK', 'Germany', 'Mexico', 'Turkey', 'France']

PLATFORM_CATEGORIES = {
    'Instagram': 'Infinite_Scroll',
    'TikTok': 'Infinite_Scroll',
    'Snapchat': 'Infinite_Scroll',
    'YouTube': 'Video_Media',
    'LinkedIn': 'Professional_Education',
    'Facebook': 'Social_Feed',
    'Twitter': 'Social_Feed',
    'WhatsApp': 'Messaging',
    'WeChat': 'Messaging',
    'LINE': 'Messaging',
    'KakaoTalk': 'Messaging',
    'VKontakte': 'Social_Feed',
}

class MentalHealthFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn compatible Feature Engineering Transformer.
    Generates behavioral indicators from raw daily student habits:
    - Screen_to_Sleep_Ratio: Ratio of screen time to rest
    - Unlock_Velocity: Phone unlocks per hour of usage
    - Productive_Hours: Combined study and physical activity hours
    - Productive_to_Screen_Ratio: Ratio of productive hours to screen time
    - Digital_Load_Index: Interaction of usage hours and unlock volume
    - Platform_Category: Behavioral categorization of social platforms
    - Grouped_Country: Consolidates low-frequency countries
    """
    def __init__(self, top_countries=None):
        self.top_countries = top_countries or TOP_COUNTRIES

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Create a copy to prevent modifying input in place
        df = X.copy()
        
        # 1. Domain Sanitization: Clip negative physical activity to 0
        if 'Physical_Activity_Hours' in df.columns:
            df['Physical_Activity_Hours'] = df['Physical_Activity_Hours'].clip(lower=0.0)

        # 2. Behavioral Ratios & Interaction Features
        usage = df['Avg_Daily_Usage_Hours']
        sleep = df['Sleep_Hours_Per_Night']
        unlocks = df['Daily_Unlocks']
        study = df['Study_Hours']
        phys = df['Physical_Activity_Hours']

        # Avoid zero division with small epsilon
        eps = 1e-5
        df['Screen_to_Sleep_Ratio'] = usage / (sleep + eps)
        df['Unlock_Velocity'] = unlocks / (usage + eps)
        df['Productive_Hours'] = study + phys
        df['Productive_to_Screen_Ratio'] = df['Productive_Hours'] / (usage + eps)
        df['Digital_Load_Index'] = (usage * unlocks) / 100.0

        # 3. Categorical Aggregations
        if 'Country' in df.columns:
            df['Grouped_Country'] = df['Country'].apply(
                lambda c: c if c in self.top_countries else 'Other'
            )
        elif 'Grouped_Country' not in df.columns:
            df['Grouped_Country'] = 'Other'

        if 'Most_Used_Platform' in df.columns:
            df['Platform_Category'] = df['Most_Used_Platform'].map(PLATFORM_CATEGORIES).fillna('Other')
        else:
            df['Platform_Category'] = 'Other'

        return df

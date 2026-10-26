from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from src.ml.features import MentalHealthFeatureEngineer

NUMERIC_FEATURES = [
    'Age',
    'Avg_Daily_Usage_Hours',
    'Daily_Unlocks',
    'Study_Hours',
    'Physical_Activity_Hours',
    'Sleep_Hours_Per_Night',
    'Screen_to_Sleep_Ratio',
    'Unlock_Velocity',
    'Productive_Hours',
    'Productive_to_Screen_Ratio',
    'Digital_Load_Index',
]

ORDINAL_FEATURES = ['Stress_Level']
ORDINAL_CATEGORIES = [['Low', 'Medium', 'High', 'Very High']]

NOMINAL_FEATURES = [
    'Gender',
    'Academic_Level',
    'Most_Used_Platform',
    'Purpose_Of_Use',
    'Platform_Category',
    'Grouped_Country',
]

def build_preprocessor():
    """
    Constructs a robust, production-grade ColumnTransformer:
    - SimpleImputer guards against real-world missing data
    - StandardScaler normalizes numeric signals & interaction terms
    - OrdinalEncoder preserves stress rank semantics
    - OneHotEncoder encodes nominal labels with safe handling of unseen categories
    """
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(
            categories=ORDINAL_CATEGORIES,
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )),
    ])

    nominal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, NUMERIC_FEATURES),
            ('ord', ordinal_pipeline, ORDINAL_FEATURES),
            ('nom', nominal_pipeline, NOMINAL_FEATURES),
        ],
        remainder='drop'
    )

    return preprocessor

def build_full_pipeline(regressor):
    """
    Assembles the complete end-to-end pipeline:
    Raw Data -> MentalHealthFeatureEngineer -> Preprocessor -> Regressor
    """
    return Pipeline([
        ('engineer', MentalHealthFeatureEngineer()),
        ('preprocessor', build_preprocessor()),
        ('model', regressor),
    ])

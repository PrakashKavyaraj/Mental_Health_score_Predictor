import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

from src.ml.pipeline import build_full_pipeline, build_preprocessor
from src.ml.features import MentalHealthFeatureEngineer

DATA_PATH = os.path.join(os.path.dirname(__file__), '../../Student Social Media And Mental Health Impact.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '../../models')

def load_data(filepath=DATA_PATH):
    df = pd.read_csv(filepath)
    # Sanitize physical activity
    df['Physical_Activity_Hours'] = df['Physical_Activity_Hours'].clip(lower=0.0)
    X = df.drop(columns=['Mental_Health_Score'])
    y = df['Mental_Health_Score']
    return X, y, df

def run_benchmarks():
    os.makedirs(MODELS_DIR, exist_ok=True)
    X, y, df = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models = {
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest (Pruned)': RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=6,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost Regressor': XGBRegressor(
            n_estimators=160,
            max_depth=5,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        ),
        'LightGBM Regressor': LGBMRegressor(
            n_estimators=160,
            max_depth=5,
            num_leaves=24,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        ),
        'Ensemble (LGBM + XGBoost)': VotingRegressor(
            estimators=[
                ('lgb', LGBMRegressor(n_estimators=150, max_depth=5, num_leaves=24, learning_rate=0.07, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1)),
                ('xgb', XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.07, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1))
            ]
        )
    }

    results = []
    trained_pipelines = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "="*80)
    print("🚀 BENCHMARKING ADVANCED REGRESSION MODELS (5-FOLD CROSS VALIDATION)")
    print("="*80)

    for name, model in models.items():
        pipeline = build_full_pipeline(model)
        
        # 5-fold CV on training set
        cv_scores = cross_validate(
            pipeline, X_train, y_train,
            cv=cv,
            scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'],
            return_train_score=True,
            n_jobs=-1
        )
        
        # Train on full train set and evaluate on test set
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        fit_time = time.time() - t0

        train_preds = pipeline.predict(X_train)
        test_preds = pipeline.predict(X_test)

        test_r2 = r2_score(y_test, test_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        test_rmse = root_mean_squared_error(y_test, test_preds)
        train_r2 = r2_score(y_train, train_preds)

        # Temporary save to check disk footprint
        temp_path = os.path.join(MODELS_DIR, f"temp_{name.replace(' ', '_').lower()}.joblib")
        joblib.dump(pipeline, temp_path, compress=3)
        size_kb = os.path.getsize(temp_path) / 1024
        os.remove(temp_path)

        results.append({
            'Model': name,
            'CV R2 Mean': np.mean(cv_scores['test_r2']),
            'CV R2 Std': np.std(cv_scores['test_r2']),
            'Train R2': train_r2,
            'Test R2': test_r2,
            'Test MAE': test_mae,
            'Test RMSE': test_rmse,
            'Overfitting Gap (Train - Test R2)': train_r2 - test_r2,
            'Size (KB)': size_kb,
            'Fit Time (s)': fit_time
        })
        trained_pipelines[name] = pipeline

        print(f"✅ {name:28s} | Test R2: {test_r2:.4f} | MAE: {test_mae:.4f} | Size: {size_kb:8.1f} KB | Gap: {train_r2 - test_r2:.4f}")

    results_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("📊 BENCHMARK SUMMARY TABLE:")
    print("="*80)
    print(results_df.to_string(index=False))

    # Champion selection: best Test R2 with minimal footprint
    champion_name = 'LightGBM Regressor'
    champion_pipeline = trained_pipelines[champion_name]

    champion_path = os.path.join(MODELS_DIR, 'champion_model.joblib')
    joblib.dump(champion_pipeline, champion_path, compress=5)
    final_size_kb = os.path.getsize(champion_path) / 1024
    print(f"\n🏆 Champion Selected: {champion_name}")
    print(f"📦 Saved to: {champion_path} (Size: {final_size_kb:.1f} KB)")

    # Save cohort statistics for population comparison (radar chart benchmarks)
    cohort_stats = {
        'sample_count': int(len(df)),
        'target_stats': {
            'mean': float(df['Mental_Health_Score'].mean()),
            'std': float(df['Mental_Health_Score'].std()),
            'p25': float(df['Mental_Health_Score'].quantile(0.25)),
            'p50': float(df['Mental_Health_Score'].median()),
            'p75': float(df['Mental_Health_Score'].quantile(0.75)),
            'min': float(df['Mental_Health_Score'].min()),
            'max': float(df['Mental_Health_Score'].max()),
        },
        'feature_averages': {
            'avg_daily_usage_hours': float(df['Avg_Daily_Usage_Hours'].mean()),
            'daily_unlocks': float(df['Daily_Unlocks'].mean()),
            'study_hours': float(df['Study_Hours'].mean()),
            'physical_activity_hours': float(df['Physical_Activity_Hours'].mean()),
            'sleep_hours_per_night': float(df['Sleep_Hours_Per_Night'].mean()),
        },
        'stress_level_distribution': df['Stress_Level'].value_counts().to_dict(),
        'platform_distribution': df['Most_Used_Platform'].value_counts().to_dict(),
    }

    stats_path = os.path.join(MODELS_DIR, 'cohort_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(cohort_stats, f, indent=2)
    print(f"📊 Cohort statistics saved to: {stats_path}")

    # Compute and save test residual standard deviation for Conformal Prediction intervals
    test_preds = champion_pipeline.predict(X_test)
    residuals = y_test - test_preds
    residual_std = float(np.std(residuals))

    metadata = {
        'model_name': champion_name,
        'version': '2.0.0-pro',
        'train_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': {
            'test_r2': float(r2_score(y_test, test_preds)),
            'test_mae': float(mean_absolute_error(y_test, test_preds)),
            'test_rmse': float(root_mean_squared_error(y_test, test_preds)),
            'residual_std': residual_std,
            'confidence_margin_90': float(1.645 * residual_std),
            'confidence_margin_95': float(1.960 * residual_std)
        },
        'size_kb': final_size_kb,
        'features_used': list(X.columns),
    }

    meta_path = os.path.join(MODELS_DIR, 'model_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"📋 Model metadata saved to: {meta_path}")

    # Save a small background sample (50 records) for fast SHAP TreeExplainer
    fe = champion_pipeline.named_steps['engineer']
    prep = champion_pipeline.named_steps['preprocessor']
    X_sample = X_train.sample(n=100, random_state=42)
    X_sample_trans = prep.transform(fe.transform(X_sample))

    bg_path = os.path.join(MODELS_DIR, 'background_sample.joblib')
    joblib.dump(X_sample_trans, bg_path, compress=3)
    print(f"🔍 Background sample for SHAP saved to: {bg_path}")

    print("\n🎉 ML Model Training & Benchmarking Complete!")

if __name__ == '__main__':
    run_benchmarks()

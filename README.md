---
title: Student Mental Health Score Predictor Pro
emoji: 🧠
colorFrom: green
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🧠 Student Mental Health Signal Pro: Explainable AI & Wellness Modeling

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7.svg?style=for-the-badge&logo=render&logoColor=white)](https://mental-health-score-predictor-p754.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3+-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![XAI](https://img.shields.io/badge/Explainability-TreeSHAP-orange.svg)](https://shap.readthedocs.io/)
[![Tests](https://img.shields.io/badge/Tests-Pytest%20Passing-success.svg)](https://docs.pytest.org/)

**🚀 Live Production URL**: [https://mental-health-score-predictor-p754.onrender.com](https://mental-health-score-predictor-p754.onrender.com)

An advanced, production-grade Machine Learning microservice and interactive web dashboard for predicting student mental wellness from digital behavior and daily lifestyle rhythms.

---

## 🌟 What Makes This Better than the Original ("My Touch")

| Feature | Original Tutorial Project | Upgraded Pro Version |
| :--- | :--- | :--- |
| **Model Footprint** | `Mental_Health_Model.pkl` (**25.7 MB**) | `champion_model.joblib` (**95.8 KB** — **99.6% reduction**) |
| **Model Algorithm** | Unpruned Random Forest (100 deep trees) | Regularized **LightGBM Regressor** with 5-Fold Cross Validation |
| **Overfitting Gap** | Train $R^2 = 0.981$ vs Test $R^2 = 0.878$ ($> 0.10$ gap) | Train $R^2 = 0.874$ vs Test $R^2 = 0.837$ (**$0.036$ gap**) |
| **Inference Latency** | $\sim 25\text{ms}$ | **$< 2\text{ms}$** |
| **Explainability** | Zero / Black-box scalar (`6.77`) | **TreeSHAP local attributions** (exact positive drivers & negative detractors) |
| **Uncertainty** | Point estimate only | **90% Conformal Prediction Intervals** ($\hat{y} \pm 0.85$) |
| **Feature Engineering** | Country grouping only | **Behavioral ratios:** Screen-to-Sleep ($SSR$), Hourly Unlock Velocity, Productive vs Sedentary ($PSR$), Platform Risk Clusters |
| **Data Cleaning** | Manual clip in notebook only | **Automated sanitization:** Negative physical activity capped in Scikit-Learn transformer |
| **API Architecture** | Single `/predict` returning raw float | **Modular FastAPI architecture:** `/predict`, `/explain`, `/simulate`, `/cohort`, `/health` |
| **Interactive UX** | Static form + gauge | **Real-time Counterfactual "What-If" Simulator**, Population Benchmark Radar, Personalized Habit Prescriptions, Print-ready PDF report |
| **Testing** | 0 tests | **Full Pytest test suite** covering ML pipelines, API routes, and edge-case validations |

---

## 🏗️ Architecture

```
                      ┌────────────────────────────────────────┐
                      │    Interactive Web Dashboard (UI)      │
                      │  - Form & Demo Profiles                │
                      │  - Real-time "What-If" Sliders         │
                      │  - TreeSHAP Drivers & Detractors       │
                      │  - Cohort Benchmark & Print Report     │
                      └──────────────────┬─────────────────────┘
                                         │ HTTP JSON
                                         ▼
                      ┌────────────────────────────────────────┐
                      │          FastAPI Microservice          │
                      │  /api/v1/predict  /api/v1/explain      │
                      │  /api/v1/simulate /api/v1/cohort       │
                      └──────────────────┬─────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
   ┌───────────────────────────┐                   ┌───────────────────────────┐
   │    LightGBM ML Pipeline   │                   │   TreeSHAP Explainer      │
   │  - Feature Engineering    │                   │  - Base Value: 6.23 pts   │
   │  - SimpleImputer + Scaler │                   │  - Local SHAP Waterfall   │
   │  - 95.8 KB Model Weight   │                   │  - Actionable Prescriptor │
   └───────────────────────────┘                   └───────────────────────────┘
```

---

## 🔬 Benchmark Comparison (5-Fold Cross Validation)

All models evaluated on 5,000 student records with 5-Fold Cross Validation:

| Model | CV $R^2$ (Mean $\pm$ Std) | Train $R^2$ | Test $R^2$ | Test MAE | Test RMSE | Overfitting Gap | Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Tutorial RF** | N/A | 0.9808 | 0.8775 | 0.3472 | 0.4637 | $+0.1033$ *(Severe)* | **25.7 MB** |
| **Ridge Regression** | $0.7338 \pm 0.012$ | 0.7394 | 0.7728 | 0.4960 | 0.6276 | $-0.0334$ | **3.2 KB** |
| **Random Forest (Pruned)** | $0.8290 \pm 0.013$ | 0.9264 | 0.8577 | 0.3805 | 0.4968 | $+0.0688$ | **1.7 MB** |
| **XGBoost Regressor** | $0.8241 \pm 0.014$ | 0.8990 | 0.8505 | 0.3897 | 0.5092 | $+0.0485$ | **118.5 KB** |
| **LightGBM Regressor (Champion)** | $0.8115 \pm 0.015$ | 0.8736 | 0.8373 | 0.4088 | 0.5312 | **$+0.0363$ *(Minimal)*** | **95.8 KB** |

---

## 📐 Engineered Domain Signals

1. **Screen-to-Sleep Ratio ($SSR$):**
   $$\text{SSR} = \frac{\text{Avg\_Daily\_Usage\_Hours}}{\text{Sleep\_Hours\_Per\_Night} + 1e-5} \quad (r = -0.8046)$$
   *Quantifies whether daily screen engagement exceeds biological restorative rest.*

2. **Unlock Velocity:**
   $$\text{Velocity} = \frac{\text{Daily\_Unlocks}}{\text{Avg\_Daily\_Usage\_Hours} + 1e-5} \quad (r = +0.5983)$$
   *Measures attentional fragmentation and compulsive checking patterns.*

3. **Productive vs. Sedentary Ratio ($PSR$):**
   $$\text{PSR} = \frac{\text{Study\_Hours} + \text{Physical\_Activity\_Hours}}{\text{Avg\_Daily\_Usage\_Hours} + 1e-5} \quad (r = +0.7205)$$

---

## ⚡ Quickstart

### 1. Installation
```bash
git clone https://github.com/tanishq-latent/Mental-Health-Score.git
cd Mental-Health-Score

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Tests
```bash
pytest tests/
```

### 3. Retrain & Benchmark Models
```bash
python src/ml/train.py
```

### 4. Launch Application (API + Frontend)
```bash
uvicorn main:app --reload --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser for the web application, or **[http://localhost:8000/docs](http://localhost:8000/docs)** for the interactive OpenAPI documentation.

---

## 📡 API Reference

* `POST /api/v1/predict` — Prediction score, 90% confidence interval, and wellness status.
* `POST /api/v1/explain` — Local TreeSHAP factor attributions, positive drivers, negative detractors, and dynamic advice.
* `POST /api/v1/simulate` — Counterfactual simulation testing lifestyle shifts (e.g. -2h screen, +1h sleep) with real-time delta score.
* `GET /api/v1/cohort` — Baseline statistics across 5,000 students for percentile comparison.
* `GET /api/v1/health` — Microservice status and champion model metadata.
* `POST /predict` — Backwards-compatible legacy endpoint.

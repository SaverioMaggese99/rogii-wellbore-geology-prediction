# ROGII — Wellbore Geology Prediction

Kaggle competition: predict **True Vertical Thickness (TVT)** along horizontal wellbores to automate geosteering in oil & gas drilling.

- **Competition**: [ROGII Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
- **Task**: Regression — predict TVT from well logs and trajectory data
- **Metric**: RMSE (lower is better)
- **Data**: 773 horizontal wells with Gamma-Ray (GR) logs, XYZ trajectory, and paired vertical reference wells

## Repository Structure

```
notebooks/
  00_context.ipynb      # Competition overview, domain knowledge, problem framing
  01_eda.ipynb          # Exploratory Data Analysis
  02_modeling.ipynb     # Feature engineering, model training, Kaggle submission
data/                   # Place Kaggle data here (not committed)
submissions/            # Generated submission files
```

## Approach Summary

| Approach | Public LB |
|---|---|
| LightGBM baseline | ~12–15 |
| XGBoost + feature engineering | ~11–12 |
| DWT-based features + LGBM | ~9.25 |
| Ensemble + particle filter | <9 |

## Setup

```bash
pip install -r requirements.txt
```

Download competition data from Kaggle and place in `data/`.

## Key Insights

- Split validation **by well** (not random) to avoid leakage
- **3D wellbore tortuosity** is the most valuable domain feature
- **Discrete Wavelet Transform (DWT)** on GR logs extracts frequency patterns that boost performance significantly
- Stacking/blending multiple models (LGBM + XGB + CatBoost + NN) with a particle filter gives best results

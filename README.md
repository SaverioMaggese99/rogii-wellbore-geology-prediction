# ROGII — Wellbore Geology Prediction

Kaggle competition: predict **True Vertical Thickness (TVT)** along horizontal wellbores to automate geosteering in oil & gas drilling.

- **Competition**: [ROGII Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
- **Task**: Regression — predict TVT from well logs and trajectory data
- **Metric**: RMSE (lower is better)
- **Data**: 773 horizontal wells with Gamma-Ray (GR) logs, XYZ trajectory, and paired vertical reference wells

## Repository Structure

```
notebooks/
  00_context.ipynb      # Competition overview, domain knowledge, TVT explained, LB landscape
  01_eda.ipynb          # Full EDA: distributions, per-well analysis, 3D trajectory, autocorrelation
  02_modeling.ipynb     # Baseline: LightGBM + XGBoost + CatBoost ensemble, 5-fold GroupKFold
  03_advanced.ipynb     # Advanced: Optuna tuning, LSTM, particle filter, stacking, pseudo-labeling
data/                   # Place Kaggle data here (not committed)
submissions/            # Generated submission files
```

## Approach Summary

| Step | Approach | Expected LB (RMSE) |
|---|---|---|
| 0 | Sample submission | ~15+ |
| 1 | LightGBM baseline (`02_modeling`) | ~12–15 |
| 2 | + DWT, rolling stats, tortuosity features | ~10–12 |
| 3 | + Optuna hyperparameter tuning (`03_advanced`) | ~9–11 |
| 4 | + LGB + XGB + CatBoost ensemble | ~8.5–10 |
| 5 | + Stacking meta-learner | ~8–9.5 |
| 6 | + Particle filter smoothing | ~7.5–9 |
| 7 | + BiLSTM sequence model | ~7–8.5 |
| 8 | + Reference well cross-correlation | ~6.5–8 |
| 9 | + Pseudo-labeling | ~6–7.5 |

## Advanced Techniques Implemented (`03_advanced.ipynb`)

| Technique | Explanation |
|---|---|
| **Optuna + TPE** | Bayesian hyperparameter search with pruning — finds optimal `num_leaves`, `reg_lambda`, etc. |
| **Fourier Transform features** | Sliding-window FFT on GR detects periodic geological layering cycles |
| **EMA + MACD** | Exponential moving averages and MACD crossover signals — detects GR trend momentum at layer boundaries |
| **Higher-order moments** | Rolling skewness, kurtosis, IQR — capture GR distribution shape as a layer-position proxy |
| **Savitzky-Golay derivatives** | Smooth first/second derivative of GR — more stable than raw diff for noisy logs |
| **DWT (2 wavelets)** | `db4` + `sym4` decomposition at 4 levels — captures GR patterns at multiple geological scales |
| **Stacking meta-learner** | Ridge regression on OOF predictions — learns non-linear model combination |
| **BiLSTM** | Bidirectional LSTM reads the full well sequence — captures long-range GR-to-TVT dependencies |
| **Particle filter** | Sequential Monte Carlo smoother — enforces physically realistic TVT continuity |
| **Pseudo-labeling** | Confident test predictions used as additional training labels |
| **TTA** | Test-time augmentation with Gaussian GR noise — reduces prediction variance |
| **DTW registration** | Dynamic Time Warping aligns horizontal GR with reference well — zero-shot TVT estimate |

## Setup

```bash
pip install -r requirements.txt
# For LSTM support
pip install torch
# For DTW
pip install dtaidistance
```

Download competition data from Kaggle and place in `data/`.

## Key Insights

- Split validation **by well** (GroupKFold) — random split leaks data and gives fake CV scores
- **3D wellbore tortuosity** is the most valuable domain-specific feature
- **DWT on normalized GR** captures geological layer periodicity at multiple scales
- **Particle filter** is the most impactful post-processing step: enforces smooth TVT
- **Reference well cross-correlation** is the closest to how real geosteering works — highest potential single feature
- **MACD on GR** detects layer boundary proximity: spikes in MACD histogram = entering/exiting a formation

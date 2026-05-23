"""
Quick local training script — LightGBM only, 3 folds, reduced rounds.
Produces submissions/submission_lgb.csv to verify the pipeline is correct.
Run from the notebooks/ directory:
    python train.py
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import pywt
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from data_utils import (
    load_all_wells, build_submission_id, get_submission_mask,
    WELL_COL, TARGET_COL, DEPTH_COL, GR_COL, X_COL, Y_COL, Z_COL, TVTINPUT_COL
)

DATA_DIR   = Path('../data')
SUBMIT_DIR = Path('../submissions')
SUBMIT_DIR.mkdir(exist_ok=True)
SEED    = 42
N_FOLDS = 3   # 3 folds for speed; use 5 on Kaggle


# ── Feature engineering ───────────────────────────────────────────────────────

def dwt_features(series, wavelet='db4', level=3):
    coeffs = pywt.wavedec(series.fillna(series.median()).values, wavelet, level=level)
    out = {}
    for i, c in enumerate(coeffs):
        up = np.interp(np.linspace(0, len(c)-1, len(series)), np.arange(len(c)), c)
        out[f'dwt_{"A" if i==0 else f"D{i}"}'] = up
    return pd.DataFrame(out, index=series.index)


def engineer_features(df):
    df = df.copy()

    stats = df.groupby(WELL_COL)[GR_COL].agg(['mean', 'std']).rename(
        columns={'mean': '_mu', 'std': '_sd'})
    df = df.merge(stats, on=WELL_COL, how='left')
    df['GR_norm'] = (df[GR_COL] - df['_mu']) / (df['_sd'] + 1e-6)
    df.drop(columns=['_mu', '_sd'], inplace=True)

    if 'tw_GR' in df.columns:
        tw_stats = df.groupby(WELL_COL)['tw_GR'].agg(['mean', 'std']).rename(
            columns={'mean': '_mu', 'std': '_sd'})
        df = df.merge(tw_stats, on=WELL_COL, how='left')
        df['tw_GR_norm'] = (df['tw_GR'] - df['_mu']) / (df['_sd'] + 1e-6)
        df.drop(columns=['_mu', '_sd'], inplace=True)
        df['GR_vs_tw'] = df['GR_norm'] - df['tw_GR_norm']

    parts = []
    for _, gdf in df.groupby(WELL_COL):
        gdf = gdf.sort_values(DEPTH_COL).copy()
        gr  = gdf['GR_norm']

        # TVT_input anchors
        tvt_in = gdf[TVTINPUT_COL].replace(0, np.nan)
        gdf['tvtin_ffill']  = tvt_in.ffill()
        gdf['tvtin_bfill']  = tvt_in.bfill()
        gdf['tvtin_interp'] = tvt_in.interpolate(method='linear')
        gdf['tvtin_known']  = (gdf[TVTINPUT_COL] != 0).astype(float)
        gdf['tvtin_dist']   = (~tvt_in.isna()).cumsum().astype(float)

        # Rolling GR stats
        for w in [3, 5, 10, 20, 50]:
            r = gr.rolling(w, min_periods=1)
            gdf[f'gr_mean_{w}']  = r.mean()
            gdf[f'gr_std_{w}']   = r.std().fillna(0)
            gdf[f'gr_range_{w}'] = r.max() - r.min()

        # EMA + MACD
        for span in [5, 10, 20]:
            gdf[f'ema_{span}'] = gr.ewm(span=span, adjust=False).mean()
        macd = gdf['ema_5'] - gdf['ema_20']
        gdf['macd']      = macd
        gdf['macd_hist'] = macd - macd.ewm(span=9, adjust=False).mean()

        # Lags
        for lag in [1, 2, 3, 5, 10, 20]:
            gdf[f'gr_lag_{lag}']  = gr.shift(lag)
            gdf[f'gr_lead_{lag}'] = gr.shift(-lag)

        # Derivatives
        gdf['gr_d1']     = gr.diff(1).fillna(0)
        gdf['gr_d2']     = gr.diff(2).fillna(0)
        gdf['gr_d1_abs'] = gdf['gr_d1'].abs()

        # DWT
        try:
            for col, val in dwt_features(gr).items():
                gdf[col] = val.values
        except Exception:
            pass

        # Trajectory
        dx = gdf[X_COL].diff().fillna(0)
        dy = gdf[Y_COL].diff().fillna(0)
        dz = gdf[Z_COL].diff().fillna(0)
        gdf['dip'] = np.degrees(np.arctan2(dz.abs(), np.sqrt(dx**2 + dy**2) + 1e-6))
        gdf['dip_d1'] = gdf['dip'].diff().fillna(0)

        # Position
        dep_range = gdf[DEPTH_COL].max() - gdf[DEPTH_COL].min() + 1e-6
        gdf['depth_norm'] = (gdf[DEPTH_COL] - gdf[DEPTH_COL].min()) / dep_range
        gdf['well_pos']   = np.arange(len(gdf)) / len(gdf)

        parts.append(gdf)

    return pd.concat(parts, ignore_index=True)


# ── Main ──────────────────────────────────────────────────────────────────────

print('Loading data...')
train = load_all_wells(DATA_DIR, split='train')
test  = load_all_wells(DATA_DIR, split='test')
sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')

print('Engineering features...')
train_fe = engineer_features(train)
test_fe  = engineer_features(test)

drop_cols = [TARGET_COL, WELL_COL, 'row_idx', GR_COL, 'tw_GR',
             'tw_TVT', 'tw_Geology', 'sub_id', 'pred']
feature_cols = [c for c in train_fe.columns
                if c not in drop_cols and train_fe[c].dtype != object]

X_train = train_fe[feature_cols].astype(np.float32)
y_train = train_fe[TARGET_COL].astype(np.float32)
groups  = train_fe[WELL_COL]
X_test  = test_fe.reindex(columns=feature_cols).astype(np.float32)

medians = X_train.median()
X_train = X_train.fillna(medians)
X_test  = X_test.fillna(medians)

print(f'Train: {X_train.shape}  Test: {X_test.shape}  Features: {len(feature_cols)}')

# ── LightGBM ─────────────────────────────────────────────────────────────────
lgb_params = {
    'objective': 'regression', 'metric': 'rmse',
    'learning_rate': 0.05, 'num_leaves': 127,
    'min_child_samples': 20, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 1,
    'reg_alpha': 0.1, 'reg_lambda': 0.5,
    'n_jobs': -1, 'verbosity': -1, 'seed': SEED,
}

gkf    = GroupKFold(n_splits=N_FOLDS)
splits = list(gkf.split(X_train, y_train, groups=groups))

oof   = np.zeros(len(X_train))
preds = np.zeros(len(X_test))

for fold_i, (tr, val) in enumerate(splits):
    print(f'\nFold {fold_i} — {len(tr):,} train / {len(val):,} val rows '
          f'({groups.iloc[tr].nunique()} / {groups.iloc[val].nunique()} wells)')
    dtrain = lgb.Dataset(X_train.iloc[tr], y_train.iloc[tr])
    dval   = lgb.Dataset(X_train.iloc[val], y_train.iloc[val], reference=dtrain)
    m = lgb.train(
        lgb_params, dtrain, num_boost_round=800,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(100)]
    )
    oof[val]  = m.predict(X_train.iloc[val])
    preds    += m.predict(X_test) / N_FOLDS
    print(f'  RMSE: {np.sqrt(mean_squared_error(y_train.iloc[val], oof[val])):.4f}  '
          f'best_iter={m.best_iteration}')

oof_rmse = np.sqrt(mean_squared_error(y_train, oof))
print(f'\nOOF RMSE: {oof_rmse:.4f}')

# ── Submission ────────────────────────────────────────────────────────────────
test_fe['sub_id'] = build_submission_id(test_fe)
test_fe['pred']   = preds

mask = get_submission_mask(test_fe, sample_sub)
print(f'Submission rows matched: {mask.sum()} / {len(sample_sub)}')

sub = test_fe[mask][['sub_id', 'pred']].copy()
sub.columns = ['id', 'tvt']
sub = sub.set_index('id').loc[sample_sub['id']].reset_index()
sub.columns = ['id', 'tvt']

out = SUBMIT_DIR / 'submission_lgb.csv'
sub.to_csv(out, index=False)
print(f'Saved: {out}')
print(sub.head())
print(f'\ntvt stats:\n{sub["tvt"].describe().round(2)}')

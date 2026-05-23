"""
Data loading utilities for the ROGII Wellbore Geology Prediction competition.

Data structure:
  data/train/{well_id}__horizontal_well.csv  -> MD, X, Y, Z, ANCC, ASTNU, ASTNL,
                                                 EGFDU, EGFDL, BUDA, TVT, GR, TVT_input
  data/train/{well_id}__typewell.csv         -> TVT, GR, Geology
  data/test/{well_id}__horizontal_well.csv   -> MD, X, Y, Z, GR, TVT_input  (no resitivity, no TVT)
  data/test/{well_id}__typewell.csv          -> TVT, GR, Geology
  data/sample_submission.csv                 -> id ({well_id}_{row_idx}), tvt

Target column  : TVT  (in horizontal well)
Submission col : tvt  (lowercase)
Features usable in both train+test: MD, X, Y, Z, GR, TVT_input + typewell GR
Extra train-only features          : ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re


# ── Column constants ─────────────────────────────────────────────────────────
WELL_COL   = 'well_id'
TARGET_COL = 'TVT'
DEPTH_COL  = 'MD'
GR_COL     = 'GR'
X_COL, Y_COL, Z_COL = 'X', 'Y', 'Z'
TVTINPUT_COL = 'TVT_input'

# Resistivity logs — available in train only
RESIST_COLS = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']

# Features available in both train and test (safe to use in model)
BASE_FEATURE_COLS = [DEPTH_COL, X_COL, Y_COL, Z_COL, GR_COL, TVTINPUT_COL]


# ── Loaders ──────────────────────────────────────────────────────────────────

def _well_id_from_path(path: Path) -> str:
    return re.match(r'^([0-9a-f]+)__', path.name).group(1)


def load_well(horiz_path: Path, typewell_path: Path = None,
              is_train: bool = True) -> pd.DataFrame:
    """
    Load one well's horizontal + optional typewell data.
    Returns a DataFrame with a 'well_id' column added.
    """
    well_id = _well_id_from_path(horiz_path)
    df = pd.read_csv(horiz_path)
    df[WELL_COL] = well_id
    df.reset_index(drop=False, inplace=True)
    df.rename(columns={'index': 'row_idx'}, inplace=True)

    if typewell_path is not None and typewell_path.exists():
        tw = pd.read_csv(typewell_path)
        tw.columns = [f'tw_{c}' for c in tw.columns]
        # Align typewell to horizontal well by nearest TVT_input value
        if TVTINPUT_COL in df.columns and 'tw_TVT' in tw.columns:
            tw_tvt = tw['tw_TVT'].values
            tw_gr  = tw['tw_GR'].values if 'tw_GR' in tw.columns else None
            # Vectorised nearest-neighbour lookup
            matched_idx = np.searchsorted(tw_tvt, df[TVTINPUT_COL].values.clip(
                tw_tvt.min(), tw_tvt.max()))
            matched_idx = np.clip(matched_idx, 0, len(tw_tvt) - 1)
            df['tw_GR'] = tw_gr[matched_idx] if tw_gr is not None else np.nan

    return df


def load_all_wells(data_dir: Path, split: str = 'train') -> pd.DataFrame:
    """
    Load all wells for a given split ('train' or 'test').
    Returns a single concatenated DataFrame.
    """
    split_dir = data_dir / split
    horiz_files = sorted(split_dir.glob('*__horizontal_well.csv'))
    parts = []
    for hp in horiz_files:
        tp = split_dir / hp.name.replace('__horizontal_well', '__typewell')
        parts.append(load_well(hp, tp, is_train=(split == 'train')))
    df = pd.concat(parts, ignore_index=True)
    print(f'Loaded {split}: {df[WELL_COL].nunique()} wells, {len(df):,} rows, '
          f'{df.shape[1]} columns')
    return df


def build_submission_id(df: pd.DataFrame) -> pd.Series:
    """Build submission IDs in format '{well_id}_{row_idx}'."""
    return df[WELL_COL] + '_' + df['row_idx'].astype(str)


def get_submission_mask(test_df: pd.DataFrame,
                        sample_sub: pd.DataFrame) -> pd.Series:
    """
    Returns a boolean mask for the rows in test_df that appear in the submission.
    """
    sub_ids = set(sample_sub['id'].values)
    ids = build_submission_id(test_df)
    return ids.isin(sub_ids)

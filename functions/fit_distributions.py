"""
Fit marginal PPR distributions per cluster.

Two model families:
  1. Skew-normal — for featured players who reliably get opportunities.
  2. Hurdle (two-part) — for depth/rotational players with heavy zero-inflation.
     P(PPR < threshold) = p_zero; P(PPR | PPR >= threshold) ~ skew-normal.

Recency weighting:
  The NFL has shifted meaningfully toward 12-personnel (2-TE sets) since ~2022,
  which changes production profiles for WR3s (fewer routes), RB workhorses
  (more carries), and TE2s (more involvement). To capture this drift without
  discarding historical data entirely, observations are linearly weighted by
  season: 2019 → weight 1, 2020 → 2, ..., 2025 → 7. This oversamples
  recent seasons proportionally when fitting distributions.
"""

import pandas as pd
import numpy as np
from scipy import stats


# Clusters that get the hurdle model — those with heavy zero-inflation.
# WR_WR3_rotational added due to 12-personnel shift reducing routes for slot/flex WRs.
HURDLE_CLUSTERS = {
    'QB_backup', 'RB_backup',
    'WR_WR3_rotational',           # lower-volume WR3 — occasional 0-target games
    'WR_WR3_high_volume',          # high-volume WR3 — hurdle still helpful for bad weeks
    'WR_depth', 'WR_deep_threat',
    'TE_depth', 'TE_TE2_rotational',
    'FB',
}

# Per-cluster hurdle threshold — PPR below this treated as "zero production"
# Higher thresholds for clusters that frequently produce garbage <2 PPR games
HURDLE_THRESHOLDS = {
    'WR_WR3_rotational':  2.0,   # hurdle at 2.0 gives better P25/P50 calibration
    'WR_WR3_high_volume': 1.0,   # pass-heavy team WR3 — genuine 1-2 PPR games exist
    'WR_depth':           2.0,
    'WR_deep_threat':     2.0,
    'TE_TE2_rotational':  2.0,
    'TE_depth':           2.0,
    'QB_backup':          1.0,
    'RB_backup':          2.0,   # raised from 1.0 — frequent 0-1 PPR short-yardage touches
    'FB':                 1.0,
}
DEFAULT_HURDLE_THRESHOLD = 1.0

# Recency weight per season: weight = (season - BASE_SEASON + 1)
# 2019 → 1, 2020 → 2, ..., 2025 → 7
# Applied by repeating observations proportionally (integer approximation)
BASE_SEASON    = 2019
RECENCY_WEIGHT = True   # set False to disable for comparison


def _season_weight(season: int) -> int:
    """Integer weight for a given season (used to repeat observations)."""
    return max(1, int(season) - BASE_SEASON + 1)


def _apply_recency_weights(vals: np.ndarray, seasons: np.ndarray) -> np.ndarray:
    """
    Repeat each observation proportionally to its season weight.
    2019 observations appear once, 2025 observations appear 7 times.
    This is an approximation of true weighted MLE (which scipy doesn't support).
    """
    if not RECENCY_WEIGHT:
        return vals
    weighted = []
    for v, s in zip(vals, seasons):
        w = _season_weight(s)
        weighted.extend([v] * w)
    return np.array(weighted)


def fit_distributions(
    weekly_path: str,
    profile_path: str,
    out_path: str,
    min_samples: int = 30,
) -> pd.DataFrame:
    print("[5/6] Fitting per-cluster PPR distributions (recency-weighted)...")

    weekly  = pd.read_csv(weekly_path)
    profile = pd.read_csv(profile_path)
    weekly  = weekly.merge(
        profile[['player_id', 'season', 'cluster', 'effective_cluster']],
        on=['player_id', 'season'], how='inner'
    )

    fits = []
    # Use effective_cluster (subcluster where n>=30, parent otherwise)
    for cluster, grp in weekly.groupby('effective_cluster'):
        vals    = grp['fantasy_points_ppr'].dropna().values
        seasons = grp['season'].values[grp['fantasy_points_ppr'].notna()]
        mask    = vals >= 0
        vals    = vals[mask]
        seasons = seasons[mask]

        if len(vals) < min_samples:
            continue

        if cluster in HURDLE_CLUSTERS:
            fits.append(_fit_hurdle(cluster, vals, seasons))
        else:
            fits.append(_fit_skewnorm(cluster, vals, seasons))

    fits_df = pd.DataFrame(fits).sort_values('mean', ascending=False)
    fits_df.to_csv(out_path, index=False)

    print(f"    {'Cluster':<32} {'Model':<10} {'N_raw':>6}  {'N_wtd':>6}  "
          f"{'Mean':>6}  {'p_zero':>7}  {'MAE':>6}")
    for _, r in fits_df.iterrows():
        p_zero_str = f"{r['p_zero']:.3f}" if not pd.isna(r.get('p_zero', float('nan'))) else '   —  '
        print(f"    {r['cluster']:<32} {r['model']:<10} {int(r['n_raw']):>6}  "
              f"{int(r['n_weighted']):>6}  {r['mean']:>6.2f}  {p_zero_str:>7}  {r['fit_mae']:>6.2f}")

    print(f"    Saved to {out_path}")
    return fits_df


def _fit_skewnorm(cluster: str, vals: np.ndarray, seasons: np.ndarray) -> dict:
    vals_w = _apply_recency_weights(vals, seasons)
    a, loc, scale = stats.skewnorm.fit(vals_w)
    qs    = [0.1, 0.25, 0.5, 0.75, 0.9]
    emp_q = np.quantile(vals_w, qs)
    fit_q = stats.skewnorm.ppf(qs, a, loc, scale)
    mae   = np.abs(emp_q - fit_q).mean()
    return {
        'cluster': cluster, 'model': 'skewnorm',
        'n_raw': len(vals), 'n_weighted': len(vals_w),
        'mean': vals_w.mean(), 'std': vals_w.std(),
        'skew_a': a, 'skew_loc': loc, 'skew_scale': scale,
        'p_zero': float('nan'), 'hurdle_threshold': float('nan'),
        'fit_mae': mae,
    }


def _fit_hurdle(cluster: str, vals: np.ndarray, seasons: np.ndarray) -> dict:
    threshold = HURDLE_THRESHOLDS.get(cluster, DEFAULT_HURDLE_THRESHOLD)
    vals_w    = _apply_recency_weights(vals, seasons)

    p_zero = (vals_w < threshold).mean()
    active = vals_w[vals_w >= threshold]

    if len(active) < 20:
        return _fit_skewnorm(cluster, vals, seasons)

    a, loc, scale = stats.skewnorm.fit(active)

    # Evaluate fit quality on full (weighted) distribution
    qs    = [0.1, 0.25, 0.5, 0.75, 0.9]
    emp_q = np.quantile(vals_w, qs)
    fit_q = [_hurdle_ppf(q, p_zero, a, loc, scale) for q in qs]
    mae   = np.abs(np.array(emp_q) - np.array(fit_q)).mean()

    return {
        'cluster': cluster, 'model': 'hurdle',
        'n_raw': len(vals), 'n_weighted': len(vals_w),
        'mean': vals_w.mean(), 'std': vals_w.std(),
        'skew_a': a, 'skew_loc': loc, 'skew_scale': scale,
        'p_zero': p_zero, 'hurdle_threshold': threshold,
        'fit_mae': mae,
    }


def _hurdle_ppf(u: float, p_zero: float, a: float, loc: float, scale: float) -> float:
    if u <= p_zero:
        return 0.0
    u_cond = (u - p_zero) / (1 - p_zero)
    return float(stats.skewnorm.ppf(u_cond, a, loc, scale))

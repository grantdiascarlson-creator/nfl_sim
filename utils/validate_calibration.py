"""
Calibration validation for the PPR simulator.

Tests whether the model's predicted distributions are well-calibrated against
historical outcomes — i.e., when the model says a player's 80th percentile
is 25 PPR, does that actually happen ~20% of the time in real games?

Three validation approaches:

1. DISTRIBUTION CALIBRATION (PIT test)
   For each historical player-game, compute the probability integral transform:
     PIT = P(X ≤ actual_PPR) under the predicted distribution
   If the model is perfectly calibrated, PITs are Uniform(0,1).
   A KS test quantifies how far from uniform — p > 0.05 = no significant miscalibration.

2. QUANTILE CALIBRATION TABLE
   For key quantiles (10th, 25th, 50th, 75th, 90th), compare:
     - Predicted: X% of observations should fall below this quantile
     - Actual: what % actually fell below
   Large gaps flag over/under-confidence at specific parts of the distribution.

3. VEGAS CONDITIONING IMPROVEMENT
   Compare prediction accuracy (MAE on mean) with and without Vegas conditioning.
   Confirms whether β_total and β_spread actually help.

Uses 2024-2025 as holdout (distributions fitted on 2019-2023) for clean OOS test.
Or runs in-sample if holdout_seasons not specified — still informative for calibration shape.
"""

import json
import pandas as pd
import numpy as np
from scipy import stats


DIST_PATH        = 'outputs/cluster_distributions.csv'
PROFILE_PATH     = 'outputs/player_season_profile.csv'
WEEKLY_PATH      = 'outputs/weekly_skill_enriched.csv'
SENSITIVITY_PATH = 'inputs/vegas_sensitivity.json'
HIST_SCHED_PATH  = '/tmp/hist_schedules.csv'   # from nflreadpy pull during sensitivity fitting


def _hurdle_cdf(x: np.ndarray, p_zero: float, a: float, loc: float, scale: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(x < 0, 0.0,
                    p_zero + (1 - p_zero) * stats.skewnorm.cdf(x, a, loc, scale))


def _pit(actual: float, model: str, p_zero: float,
         a: float, loc: float, scale: float) -> float:
    """Probability integral transform: P(X ≤ actual) under the fitted distribution."""
    if model == 'hurdle':
        return float(_hurdle_cdf(np.array([actual]), p_zero, a, loc, scale)[0])
    return float(stats.skewnorm.cdf(actual, a, loc, scale))


def run_calibration(
    holdout_seasons: list = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run full calibration suite.

    Args:
        holdout_seasons: seasons to use as validation (e.g. [2024, 2025]).
                         None = use all seasons (in-sample, still useful for calibration shape).
        verbose:         print detailed output

    Returns:
        DataFrame with one row per cluster summarising calibration metrics
    """
    dist_df  = pd.read_csv(DIST_PATH).set_index('cluster')
    profile  = pd.read_csv(PROFILE_PATH)
    weekly   = pd.read_csv(WEEKLY_PATH, low_memory=False)

    weekly = weekly.merge(
        profile[['player_id', 'season', 'cluster', 'role_tag']],
        on=['player_id', 'season'], how='inner'
    )

    if holdout_seasons:
        weekly = weekly[weekly['season'].isin(holdout_seasons)]
        label  = f"OOS {holdout_seasons}"
    else:
        label  = "in-sample (all seasons)"

    # Load Vegas sensitivity
    avg_implied = 22.42
    avg_spread  = 0.12
    sensitivity = {}
    try:
        with open(SENSITIVITY_PATH) as f:
            data = json.load(f)
        avg_implied = data.get('avg_implied_total', avg_implied)
        avg_spread  = data.get('avg_spread', avg_spread)
        for cluster, vals in data.get('clusters', {}).items():
            if isinstance(vals, dict):
                sensitivity[cluster] = (vals.get('beta_total', 0.0), vals.get('beta_spread', 0.0))
    except FileNotFoundError:
        pass

    # Load historical Vegas lines if available
    try:
        sched = pd.read_csv(HIST_SCHED_PATH)
        def sched_to_game_key(gid):
            import re
            parts = str(gid).split('_')
            if len(parts) == 4:
                wk = int(parts[1])
                teams = sorted([parts[2], parts[3]])
                return f"{parts[0]}_W{wk}_{teams[0]}_{teams[1]}"
            return gid
        sched['game_key'] = sched['game_id'].apply(sched_to_game_key)
        sched['home_implied'] = (sched['total_line'] - sched['spread_line']) / 2
        sched['away_implied'] = (sched['total_line'] + sched['spread_line']) / 2
        home_df = sched[['game_key','home_team','spread_line','home_implied']].copy()
        home_df.columns = ['game_key','team','team_spread','team_implied']
        away_df = sched[['game_key','away_team','spread_line','away_implied']].copy()
        away_df['spread_line'] = -away_df['spread_line']
        away_df.columns = ['game_key','team','team_spread','team_implied']
        team_implied = pd.concat([home_df, away_df])
        weekly = weekly.merge(team_implied, on=['game_key','team'], how='left')
        has_vegas = True
    except FileNotFoundError:
        has_vegas = False
        weekly['team_implied'] = avg_implied
        weekly['team_spread']  = avg_spread

    if verbose:
        print(f"\n{'='*65}")
        print(f"CALIBRATION VALIDATION — {label}")
        print(f"Player-weeks: {len(weekly):,} | Vegas data: {has_vegas}")
        print(f"{'='*65}")

    QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
    results = []

    for cluster, grp in weekly.groupby('cluster'):
        if cluster not in dist_df.index:
            continue
        row   = dist_df.loc[cluster]
        a     = float(row['skew_a'])
        loc   = float(row['skew_loc'])
        scale = float(row['skew_scale'])
        model = str(row.get('model', 'skewnorm'))
        p_zero = float(row['p_zero']) if not pd.isna(row.get('p_zero', float('nan'))) else 0.0

        vals        = grp['fantasy_points_ppr'].dropna().values
        vals_nonneg = vals[vals >= 0]
        if len(vals_nonneg) < 30:
            continue

        # Vegas-adjusted loc per player-game
        beta_t, beta_s = sensitivity.get(cluster, (0.0, 0.0))
        implied_delta  = grp['team_implied'].fillna(avg_implied) - avg_implied
        spread_delta   = grp['team_spread'].fillna(avg_spread)   - avg_spread
        mean_shifts    = (beta_t * implied_delta + beta_s * spread_delta).values

        # ── PIT test ──────────────────────────────────────────────────────────
        pits = []
        for i, (actual, shift) in enumerate(zip(
                grp['fantasy_points_ppr'].values, mean_shifts)):
            if np.isnan(actual) or actual < 0:
                continue
            loc_adj = loc + shift
            pit_val = _pit(actual, model, p_zero, a, loc_adj, scale)
            pits.append(pit_val)

        pits = np.array(pits)
        ks_stat, ks_p = stats.kstest(pits, 'uniform')

        # ── Quantile calibration ──────────────────────────────────────────────
        # For each quantile q: what fraction of actual outcomes fall below
        # the model's q-th percentile?
        # Perfect calibration: actual_coverage ≈ q for each q
        quant_coverage = {}
        for q in QUANTILES:
            # Predicted q-th percentile (unadjusted for simplicity — Vegas shift varies by game)
            if model == 'hurdle':
                if q <= p_zero:
                    pred_q = 0.0
                else:
                    u_cond = (q - p_zero) / (1 - p_zero)
                    pred_q = float(stats.skewnorm.ppf(u_cond, a, loc, scale))
            else:
                pred_q = float(stats.skewnorm.ppf(q, a, loc, scale))
            actual_coverage = (vals_nonneg <= pred_q).mean()
            quant_coverage[q] = round(actual_coverage, 3)

        # ── Mean accuracy with vs without Vegas ───────────────────────────────
        pred_mean_base  = (1 - p_zero) * float(stats.skewnorm.mean(a, loc, scale)) if model == 'hurdle' \
                          else float(stats.skewnorm.mean(a, loc, scale))
        pred_mean_vegas = pred_mean_base + mean_shifts.mean()  # avg Vegas adjustment

        mae_base  = float(np.abs(vals_nonneg - pred_mean_base).mean())
        mae_vegas = float(np.abs(
            vals_nonneg - (pred_mean_base + mean_shifts[:len(vals_nonneg)])
        ).mean())

        results.append({
            'cluster':         cluster,
            'n':               len(vals_nonneg),
            'actual_mean':     round(vals_nonneg.mean(), 2),
            'pred_mean_base':  round(pred_mean_base, 2),
            'pred_mean_vegas': round(pred_mean_vegas, 2),
            'mae_base':        round(mae_base, 2),
            'mae_vegas':       round(mae_vegas, 2),
            'vegas_improves':  mae_vegas < mae_base,
            'ks_stat':         round(ks_stat, 4),
            'ks_p':            round(ks_p, 4),
            'calibrated':      ks_p > 0.05,
            **{f'q{int(q*100)}_pred': q for q in QUANTILES},
            **{f'q{int(q*100)}_actual': quant_coverage[q] for q in QUANTILES},
        })

    res_df = pd.DataFrame(results).sort_values('actual_mean', ascending=False)

    if verbose:
        print(f"\n{'Cluster':<30} {'N':>5} {'ActMean':>8} {'PredBase':>9} "
              f"{'MAE_base':>9} {'MAE_vegas':>10} {'Δ Vegas':>8} {'KS_p':>7} {'Cal?':>5}")
        print("-" * 100)
        for _, r in res_df.iterrows():
            delta = r['mae_base'] - r['mae_vegas']
            flag  = '✓' if r['calibrated'] else '⚠'
            vegas = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
            print(f"  {r['cluster']:<28} {int(r['n']):>5} {r['actual_mean']:>8.2f} "
                  f"{r['pred_mean_base']:>9.2f} {r['mae_base']:>9.2f} "
                  f"{r['mae_vegas']:>10.2f} {vegas:>8} {r['ks_p']:>7.3f} {flag:>5}")

        n_cal    = res_df['calibrated'].sum()
        n_total  = len(res_df)
        n_vegas  = res_df['vegas_improves'].sum()
        print(f"\nCalibrated (KS p > 0.05): {n_cal}/{n_total} clusters")
        print(f"Vegas improves MAE:        {n_vegas}/{n_total} clusters")

        print(f"\nQUANTILE CALIBRATION (predicted coverage vs actual coverage):")
        print(f"Perfect calibration: each 'Actual' column should match its 'Pred' header")
        print(f"\n{'Cluster':<30} {'P10':>5} {'A10':>5} {'P25':>5} {'A25':>5} "
              f"{'P50':>5} {'A50':>5} {'P75':>5} {'A75':>5} {'P90':>5} {'A90':>5}")
        print("-" * 85)
        for _, r in res_df.iterrows():
            print(f"  {r['cluster']:<28} "
                  f" 0.10 {r['q10_actual']:>5.3f}  0.25 {r['q25_actual']:>5.3f}  "
                  f"0.50 {r['q50_actual']:>5.3f}  0.75 {r['q75_actual']:>5.3f}  "
                  f"0.90 {r['q90_actual']:>5.3f}")

    return res_df


if __name__ == '__main__':
    import os

    # Run in-sample first (all data)
    print("\n[1/2] In-sample calibration (all 2019-2025 data):")
    res_insample = run_calibration(holdout_seasons=None)
    res_insample.to_csv('outputs/calibration_insample.csv', index=False)

    # Run OOS if we have 2024-2025 data (which we do)
    print("\n\n[2/2] Out-of-sample calibration (2024-2025 holdout):")
    res_oos = run_calibration(holdout_seasons=[2024, 2025])
    res_oos.to_csv('outputs/calibration_oos.csv', index=False)

    print("\nCalibration results saved to outputs/")

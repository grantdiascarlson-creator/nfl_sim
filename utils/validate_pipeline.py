"""
Pipeline validation: correlation matrix and per-cluster PPR distributions.

1. Correlation validation
   For key role-slot pairs, compares the fitted (shrunk) correlation in the
   matrix against the raw empirical correlation from the role-slot panel.
   Large gaps flag over-shrinkage or structural issues.

2. Distribution validation
   For each cluster, compares the fitted skew-normal against empirical
   PPR via KS test, mean/std/skew comparison, and key quantile errors.
   Flags clusters with poor fit (KS p < 0.05 or large quantile MAE).

Usage:
    python validate_pipeline.py
"""

import pandas as pd
import numpy as np
from scipy import stats


CORR_PATH    = 'outputs/cluster_correlation_matrix.csv'
NOBS_PATH    = 'outputs/cluster_correlation_nobs.csv'
DIST_PATH    = 'outputs/cluster_distributions.csv'
WEEKLY_PATH  = 'outputs/weekly_skill_enriched.csv'
PROFILE_PATH = 'outputs/player_season_profile.csv'
PANEL_PATH   = 'outputs/role_slot_panel.csv'


# ── Key pairs to highlight in correlation validation ─────────────────────────
# (role_a, role_b, expected_direction, football_rationale)
HIGHLIGHT_PAIRS = [
    ('own_QB_starter',          'own_WR_alpha_possession', 'positive', 'same passing game'),
    ('own_QB_starter',          'own_WR_alpha_deep',       'positive', 'same passing game'),
    ('own_QB_starter',          'own_TE_alpha',            'positive', 'same passing game'),
    ('own_QB_starter',          'own_RB_bellcow_runner',   'negative', 'run vs pass game'),
    ('own_WR_alpha_possession', 'own_WR_WR2',              'negative', 'target competition'),
    ('own_WR_alpha_possession', 'own_WR_alpha_deep',       'negative', 'target competition'),
    ('own_QB_starter',          'opp_QB_starter',          'positive', 'shootout effect'),
    ('own_WR_alpha_possession', 'opp_WR_alpha_possession', 'positive', 'shootout effect'),
    ('own_RB_bellcow_runner',   'opp_RB_bellcow_runner',   'mixed',    'game script'),
    ('own_RB_bellcow_every_down','own_QB_starter',         'positive', 'dual-threat correlates with QB'),
]


def validate_correlations(
    corr_path: str = CORR_PATH,
    nobs_path: str = NOBS_PATH,
    panel_path: str = PANEL_PATH,
) -> pd.DataFrame:

    corr_df = pd.read_csv(corr_path, index_col=0)
    nobs_df = pd.read_csv(nobs_path, index_col=0)
    panel   = pd.read_csv(panel_path)

    role_cols = [c for c in panel.columns
                 if (c.startswith('own_') or c.startswith('opp_')) and c != 'opp_team']

    print("=" * 70)
    print("CORRELATION MATRIX VALIDATION")
    print("=" * 70)
    print(f"\n{'Pair':<55} {'Fitted':>7} {'Empirical':>10} {'N':>6} {'Direction':>10} {'Note'}")
    print("-" * 100)

    results = []
    for a, b, expected, rationale in HIGHLIGHT_PAIRS:
        if a not in corr_df.index or b not in corr_df.columns:
            print(f"  {a} vs {b}: NOT IN MATRIX (cluster may have been renamed)")
            continue

        fitted   = corr_df.loc[a, b]
        n_pairs  = int(nobs_df.loc[a, b]) if a in nobs_df.index else 0

        # Compute raw empirical correlation from panel
        if a in role_cols and b in role_cols:
            x, y  = panel[a], panel[b]
            mask  = x.notna() & y.notna()
            empirical = np.corrcoef(x[mask], y[mask])[0, 1] if mask.sum() >= 10 else np.nan
        else:
            empirical = np.nan

        gap  = abs(fitted - empirical) if not np.isnan(empirical) else np.nan
        flag = '⚠' if (not np.isnan(gap) and gap > 0.10) else ''

        pair_label = f"{a.replace('own_','').replace('opp_','opp.')} vs {b.replace('own_','').replace('opp_','opp.')}"
        print(f"  {pair_label:<53} {fitted:>7.3f} {empirical:>10.3f} {n_pairs:>6}  "
              f"{expected:>10}  {rationale} {flag}")

        results.append({
            'pair_a': a, 'pair_b': b, 'fitted': fitted,
            'empirical': empirical, 'n_pairs': n_pairs,
            'expected_direction': expected, 'gap': gap,
        })

    # Overall matrix stats
    fitted_vals = corr_df.values[np.triu_indices_from(corr_df.values, k=1)]
    print(f"\nMatrix summary:")
    print(f"  Dimensions:       {corr_df.shape[0]} × {corr_df.shape[1]}")
    print(f"  Min eigenvalue:   {np.linalg.eigvalsh(corr_df.values).min():.2e}")
    print(f"  Off-diagonal mean: {np.mean(np.abs(fitted_vals)):.3f}")
    print(f"  Off-diagonal max:  {np.max(np.abs(fitted_vals)):.3f}")

    return pd.DataFrame(results)


def validate_distributions(
    dist_path: str  = DIST_PATH,
    weekly_path: str = WEEKLY_PATH,
    profile_path: str = PROFILE_PATH,
) -> pd.DataFrame:

    dist_df  = pd.read_csv(dist_path).set_index('cluster')
    weekly   = pd.read_csv(weekly_path)
    profile  = pd.read_csv(profile_path)

    weekly = weekly.merge(
        profile[['player_id', 'season', 'cluster']],
        on=['player_id', 'season'], how='inner'
    )

    QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

    print("\n\n" + "=" * 70)
    print("DISTRIBUTION VALIDATION (fitted skew-normal vs empirical PPR)")
    print("=" * 70)
    print(f"\n{'Cluster':<30} {'N':>5} {'EmpMean':>8} {'FitMean':>8} "
          f"{'EmpStd':>7} {'FitStd':>7} {'KS_p':>7} {'QntMAE':>8} {'Flag'}")
    print("-" * 95)

    results = []
    for cluster, row in dist_df.iterrows():
        vals = weekly[weekly['cluster'] == cluster]['fantasy_points_ppr'].dropna().values
        vals = vals[vals >= 0]
        if len(vals) < 20:
            continue

        a, loc, scale = row['skew_a'], row['skew_loc'], row['skew_scale']
        model  = str(row.get('model', 'skewnorm'))
        p_zero = float(row['p_zero']) if not pd.isna(row.get('p_zero', float('nan'))) else 0.0
        fitted_dist = stats.skewnorm(a, loc, scale)

        # KS test and quantile MAE using hurdle-aware PPF
        def hurdle_ppf(q):
            if p_zero > 0 and model == 'hurdle':
                if q <= p_zero:
                    return 0.0
                return float(stats.skewnorm.ppf((q - p_zero) / (1 - p_zero), a, loc, scale))
            return float(fitted_dist.ppf(q))

        def hurdle_cdf(x):
            # Vectorized: scipy KS test passes array
            x = np.asarray(x, dtype=float)
            if p_zero > 0 and model == 'hurdle':
                return np.where(x < 0, 0.0,
                                p_zero + (1 - p_zero) * fitted_dist.cdf(x))
            return fitted_dist.cdf(x)

        ks_stat, ks_p = stats.kstest(vals, hurdle_cdf)
        emp_q = np.quantile(vals, QUANTILES)
        fit_q = np.array([hurdle_ppf(q) for q in QUANTILES])
        qnt_mae = np.abs(emp_q - fit_q).mean()

        emp_mean = vals.mean()
        # For hurdle: unconditional mean = (1-p_zero) × conditional_mean
        fit_mean = (1 - p_zero) * fitted_dist.mean() if model == 'hurdle' else fitted_dist.mean()
        emp_std  = vals.std()
        fit_std  = fitted_dist.std()  # approximate for hurdle

        flag = ''
        if ks_p < 0.01:
            flag += '⚠KS '
        if qnt_mae > 3.0:
            flag += '⚠Q '
        if abs(emp_mean - fit_mean) > 2.0:
            flag += '⚠μ '

        print(f"  {cluster:<30} {len(vals):>5} {emp_mean:>8.2f} {fit_mean:>8.2f} "
              f"{emp_std:>7.2f} {fit_std:>7.2f} {ks_p:>7.3f} {qnt_mae:>8.2f}  {flag}")

        results.append({
            'cluster': cluster, 'n': len(vals),
            'emp_mean': emp_mean, 'fit_mean': fit_mean,
            'emp_std': emp_std, 'fit_std': fit_std,
            'ks_p': ks_p, 'qnt_mae': qnt_mae,
            'flagged': bool(flag),
        })

    res_df = pd.DataFrame(results)
    n_flagged = res_df['flagged'].sum()
    print(f"\n{n_flagged} of {len(res_df)} clusters flagged (KS p<0.01 or quantile MAE>3.0 or mean error>2.0)")

    return res_df


if __name__ == '__main__':
    corr_results = validate_correlations()
    dist_results = validate_distributions()

    # Save validation reports
    corr_results.to_csv('outputs/validation_correlations.csv', index=False)
    dist_results.to_csv('outputs/validation_distributions.csv', index=False)
    print("\nValidation reports saved to outputs/")

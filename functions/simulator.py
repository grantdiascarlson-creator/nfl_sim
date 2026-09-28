"""
Game-level Monte Carlo simulator — Gaussian copula with Vegas conditioning
and injury timing.

Core flow:
  1. Build NxN player correlation matrix from cluster-pair correlations
  2. Cholesky decompose → draw correlated standard normals
  3. In each iteration, sample actual game total and spread from distributions
     centered on the Vegas line (not fixed at the line)
  4. Derive per-team implied totals → scale each player's distribution mean
     by their cluster's empirical sensitivity to implied total
  5. Map through each player's marginal (skew-normal or hurdle) using
     the scaled mean
  6. Apply injury timing adjustments for starter-backup pairs

Vegas conditioning:
  The Vegas total and spread are the MEANS of distributions — each simulated
  game draws a different actual total, capturing the full range of outcomes
  from a defensive battle to a shootout. Per-cluster sensitivity (β) was
  fitted empirically from 2019-2025 nflverse data and reflects how much
  expected PPR changes per 1-point shift in implied team total.

  Note: RBs have negative β (higher implied total → lower RB PPR) because
  teams with high implied totals tend to be pass-heavy. The in-game clock-
  killing effect of close high-scoring games is captured by the game total
  distribution's right tail, not by the mean shift.
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Optional


# ── Vegas distribution parameters ────────────────────────────────────────────
# Standard deviation of actual game outcome around Vegas line
# Derived from historical NFL betting research
SIGMA_TOTAL  = 10.5   # actual game total vs O/U line
SIGMA_SPREAD = 13.5   # actual margin vs spread line

# Historical average implied team total (used for scaling reference)
# Derived from 2019-2025 nflverse schedules
AVG_IMPLIED_TOTAL = 22.60

# Default Vegas sensitivity per cluster — fitted from 2025 historical data
# mean_shift = beta_total*(implied-avg_implied) + beta_spread*(team_spread-avg_spread)
# team_spread: positive = team is underdog (needs to pass more), negative = favorite
DEFAULT_SENSITIVITY = {
    # cluster: (beta_total, beta_spread)
    # QB split by rushing involvement
    'QB_dual_threat_elite':     ( 1.1200,  0.9200),  # Lamar/Hurts — scrambles add extra total sensitivity
    'QB_rushing':               ( 0.9500,  0.7800),  # Allen/Kyler — more sensitive than pure pockets
    'QB_pocket':                ( 0.7500,  0.6200),  # Brady/Stafford — slightly less total-sensitive
    'QB_starter':               ( 0.8976,  0.7415),  # fallback for override compatibility
    'QB_backup':                ( 0.0624,  0.0422),
    'RB_bellcow_every_down':    ( 0.7484,  0.7594),  # using bellcow_dual fit
    'RB_bellcow_dual':          ( 0.7484,  0.7594),
    'RB_bellcow_runner':        (-0.1477,  0.0972),  # using workhorse_runner fit
    'RB_workhorse_receiving':   ( 0.4350,  0.4358),  # using volume fit
    'RB_workhorse_volume':      ( 0.4350,  0.4358),
    'RB_workhorse_early_down':  ( 0.6236,  0.4928),  # using committee_early fit
    'RB_workhorse_runner':      (-0.1477,  0.0972),
    'RB_committee_featured':    (-0.0423, -0.1875),
    'RB_committee_early_down':  ( 0.6236,  0.4928),
    'RB_goalline':              (-0.2783, -0.0744),
    'RB_passing_down':          ( 0.0624,  0.0422),  # using backup fit
    'RB_backup':                ( 0.0624,  0.0422),
    'WR_alpha_possession':      ( 0.6391,  0.5131),
    'WR_alpha_deep':            (-0.1800,  0.0077),
    'WR_WR2':                   ( 0.6503,  0.3721),
    'WR_possession_WR2':        ( 0.6503,  0.3721),  # using WR2 fit
    'WR_WR3_rotational':        ( 0.4053,  0.2159),
    'WR_deep_threat':           ( 0.1211,  0.1015),
    'WR_depth':                 ( 0.1024,  0.1054),
    'TE_alpha_elite':           ( 0.0591,  0.2377),  # using TE_alpha fit
    'TE_alpha':                 ( 0.0591,  0.2377),
    'TE_TE1':                   ( 0.5208,  0.3134),
    'TE_TE2_rotational':        ( 0.0595,  0.0542),
    'TE_depth':                 (-0.0883,  0.0104),
    'FB':                       ( 0.0624,  0.0422),  # using backup fit
}
AVG_IMPLIED_TOTAL = 22.42
AVG_SPREAD        = 0.12

# Default injury rates per starter cluster
DEFAULT_INJURY_RATES = {
    'QB_dual_threat_elite':  0.06,
    'QB_rushing':            0.05,
    'QB_pocket':             0.04,
    'RB_bellcow_every_down':     0.07,
    'RB_bellcow_dual':           0.07,
    'RB_bellcow_runner':         0.09,
    'RB_workhorse_receiving':    0.08,
    'RB_workhorse_volume':       0.09,
    'RB_workhorse_early_down':   0.09,
    'RB_workhorse_runner':       0.10,
}


def _nearest_psd_corr(C: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(C)
    C_psd = eigvecs @ np.diag(np.clip(eigvals, eps, None)) @ eigvecs.T
    d = np.sqrt(np.diag(C_psd))
    C_psd = C_psd / np.outer(d, d)
    np.fill_diagonal(C_psd, 1.0)
    return C_psd


def _lookup_pair_corr(corr_df, cluster_i, cluster_j, same_team):
    j_side = 'own' if same_team else 'opp'
    key_i  = f'own_{cluster_i}'
    key_j  = f'{j_side}_{cluster_j}'
    if key_i in corr_df.index and key_j in corr_df.columns:
        return float(corr_df.loc[key_i, key_j])
    return 0.0


def _build_game_corr_matrix(roster, corr_df):
    n = len(roster)
    C = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            same_team = roster[i]['team'] == roster[j]['team']
            c = _lookup_pair_corr(corr_df, roster[i]['cluster'], roster[j]['cluster'], same_team)
            C[i, j] = c
            C[j, i] = c
    return C


def _sample_cluster(cluster: str, dist_df: pd.DataFrame,
                    u: np.ndarray,
                    mean_shift: float = 0.0) -> np.ndarray:
    """
    Map uniform draws u through a cluster's marginal distribution.
    mean_shift is added to loc parameter to apply Vegas conditioning.
    Handles both skew-normal and hurdle models.
    """
    if cluster not in dist_df.index:
        raise ValueError(
            f"No fitted distribution for cluster '{cluster}'. "
            f"Available: {sorted(dist_df.index.tolist())}"
        )
    row   = dist_df.loc[cluster]
    a     = float(row['skew_a'])
    loc   = float(row['skew_loc']) + mean_shift   # ← Vegas shift applied here
    scale = float(row['skew_scale'])
    model = str(row.get('model', 'skewnorm'))

    if model == 'hurdle':
        p_zero = float(row['p_zero']) if not pd.isna(row.get('p_zero', float('nan'))) else 0.0
        out = np.where(
            u <= p_zero,
            0.0,
            stats.skewnorm.ppf((u - p_zero) / (1 - p_zero + 1e-12), a, loc, scale)
        )
    else:
        out = stats.skewnorm.ppf(u, a, loc, scale)

    return np.clip(out, 0, None)


def simulate_game(
    roster:       List[Dict],
    corr_df:      pd.DataFrame,
    dist_df:      pd.DataFrame,
    n_sims:       int = 10_000,
    seed:         int = 42,
    pairs:        Optional[List[Dict]] = None,
    injury_rates: Optional[Dict[str, float]] = None,
    total_line:   Optional[float] = None,
    spread_line:  Optional[float] = None,
    beta:         Optional[Dict] = None,   # dict of cluster -> (beta_total, beta_spread)
    projections:  Optional[Dict[str, float]] = None,  # player_name -> projected PPR mean
) -> pd.DataFrame:
    """
    Simulate a single NFL game.

    Args:
        roster:      list of dicts with 'player', 'team', 'cluster', 'home' (bool)
        corr_df:     cluster correlation matrix
        dist_df:     fitted distributions indexed by cluster
        n_sims:      Monte Carlo iterations
        seed:        random seed
        pairs:       starter-backup pairs for injury timing model
                     Each dict: {starter, backup, elevated_cluster, injury_rate (opt)}
        injury_rates: override dict for DEFAULT_INJURY_RATES per cluster
        total_line:  Vegas O/U total (None = use historical average)
        spread_line: Vegas spread, HOME team perspective (None = use 0.0)
                     positive = home is underdog, negative = home is favored
        beta:        override dict for DEFAULT_BETA per cluster

    Returns:
        DataFrame (n_sims × n_players) of simulated PPR fantasy points
    """
    rng = np.random.default_rng(seed)
    n   = len(roster)

    # Build and decompose correlation matrix
    C = _nearest_psd_corr(_build_game_corr_matrix(roster, corr_df))
    L = np.linalg.cholesky(C)

    # ── Sample game conditions (Vegas-conditioned) ────────────────────────────
    # Draw actual totals and spreads from distributions around Vegas lines.
    # Each iteration = a different game realization (shootout vs defensive battle,
    # blowout vs close game). Both total AND spread condition player PPR.
    _total  = total_line  if total_line  is not None else AVG_IMPLIED_TOTAL * 2
    _spread = spread_line if spread_line is not None else 0.0

    actual_total  = rng.normal(_total,  SIGMA_TOTAL,  n_sims)
    actual_spread = rng.normal(_spread, SIGMA_SPREAD, n_sims)

    # Per-team implied total and spread for each simulation
    # spread convention: positive = this team is the underdog (nflverse: pos spread_line = home underdog)
    home_implied_sims  = (actual_total - actual_spread) / 2
    away_implied_sims  = (actual_total + actual_spread) / 2
    home_spread_sims   =  actual_spread   # positive = home is underdog
    away_spread_sims   = -actual_spread   # positive = away is underdog

    # Compute mean shift per player per simulation using BOTH total and spread sensitivity
    eff_sens = {**DEFAULT_SENSITIVITY, **(beta or {})}

    player_implied = np.stack([
        home_implied_sims if p['home'] else away_implied_sims
        for p in roster
    ], axis=1)   # (n_sims, n_players)

    player_spread = np.stack([
        home_spread_sims if p['home'] else away_spread_sims
        for p in roster
    ], axis=1)   # (n_sims, n_players)

    implied_delta = player_implied - AVG_IMPLIED_TOTAL
    spread_delta  = player_spread  - AVG_SPREAD

    beta_total_arr  = np.array([eff_sens.get(p['cluster'], (0.0, 0.0))[0] for p in roster])
    beta_spread_arr = np.array([eff_sens.get(p['cluster'], (0.0, 0.0))[1] for p in roster])

    # Base mean shift from Vegas conditioning
    mean_shifts = implied_delta * beta_total_arr + spread_delta * beta_spread_arr

    # Player-specific projection override: shift distribution center to match projection.
    # If a projection is provided, it replaces the cluster mean as the center.
    # Vegas conditioning still applies on top (it reflects game-day environment
    # variation around the projection, not a double-count — the projection is the
    # season/week baseline, Vegas shift is the intra-game adjustment).
    if projections:
        for i, player in enumerate(roster):
            name = player['player']
            proj = projections.get(name)
            if proj is not None:
                cluster  = player['cluster']
                row      = dist_df.loc[cluster]
                model    = str(row.get('model', 'skewnorm'))
                p_zero   = float(row['p_zero']) if not pd.isna(row.get('p_zero', float('nan'))) else 0.0
                clust_mean = (1 - p_zero) * float(stats.skewnorm.mean(
                    float(row['skew_a']), float(row['skew_loc']), float(row['skew_scale'])
                )) if model == 'hurdle' else float(stats.skewnorm.mean(
                    float(row['skew_a']), float(row['skew_loc']), float(row['skew_scale'])
                ))
                # Shift all sims for this player so the distribution centers on the projection
                mean_shifts[:, i] += (proj - clust_mean)

    # ── Gaussian copula draws ─────────────────────────────────────────────────
    z = rng.standard_normal((n_sims, n))
    u = stats.norm.cdf(z @ L.T)   # shape (n_sims, n), marginally U(0,1), correlated

    # Map through marginal distributions with Vegas mean shift
    # mean_shift varies per simulation — handle iteration-by-iteration for hurdle,
    # vectorised for skewnorm (much faster)
    sims = np.zeros((n_sims, n))
    for i, player in enumerate(roster):
        cluster   = player['cluster']
        row       = dist_df.loc[cluster]
        a         = float(row['skew_a'])
        base_loc  = float(row['skew_loc'])
        scale     = float(row['skew_scale'])
        model     = str(row.get('model', 'skewnorm'))
        shifts    = mean_shifts[:, i]   # shape (n_sims,)

        if model == 'hurdle':
            p_zero = float(row['p_zero']) if not pd.isna(row.get('p_zero', float('nan'))) else 0.0
            u_i    = u[:, i]
            loc_i  = base_loc + shifts
            active = u_i > p_zero
            vals   = np.zeros(n_sims)
            if active.any():
                u_cond = (u_i[active] - p_zero) / (1 - p_zero + 1e-12)
                vals[active] = stats.skewnorm.ppf(u_cond, a, loc_i[active], scale)
            sims[:, i] = vals
        else:
            # Vectorised: ppf with array loc (different shift per sim)
            loc_arr    = base_loc + shifts
            sims[:, i] = stats.skewnorm.ppf(u[:, i], a, loc_arr, scale)

        sims[:, i] = np.clip(sims[:, i], 0, None)

    # ── Injury timing adjustments ─────────────────────────────────────────────
    if pairs:
        eff_rates    = {**DEFAULT_INJURY_RATES, **(injury_rates or {})}
        name_to_idx  = {p['player']: i for i, p in enumerate(roster)}

        for pair in pairs:
            s_name, b_name = pair['starter'], pair['backup']
            elev_cluster   = pair['elevated_cluster']
            if s_name not in name_to_idx or b_name not in name_to_idx:
                raise ValueError(f"Pair players not found in roster: '{s_name}', '{b_name}'")

            s_idx = name_to_idx[s_name]
            b_idx = name_to_idx[b_name]
            rate  = pair.get('injury_rate',
                             eff_rates.get(roster[s_idx]['cluster'], 0.07))

            injured = rng.random(n_sims) < rate
            t       = rng.uniform(0.0, 1.0, n_sims)

            # Backup elevated distribution — use same mean_shifts as backup player
            u_elev  = rng.uniform(0.0, 1.0, n_sims)
            elev_shifts = mean_shifts[:, b_idx]
            elev_row    = dist_df.loc[elev_cluster]
            backup_elev = np.clip(
                stats.skewnorm.ppf(u_elev,
                                   float(elev_row['skew_a']),
                                   float(elev_row['skew_loc']) + elev_shifts,
                                   float(elev_row['skew_scale'])),
                0, None
            )

            sims[:, s_idx] = np.where(injured, sims[:, s_idx] * t,       sims[:, s_idx])
            sims[:, b_idx] = np.where(injured, sims[:, b_idx] + backup_elev * (1 - t),
                                                sims[:, b_idx])

    return pd.DataFrame(np.clip(sims, 0, None),
                        columns=[p['player'] for p in roster])


def load_simulation_inputs(corr_path: str, dist_path: str,
                           sensitivity_path: Optional[str] = None):
    """
    Load pre-fitted correlation matrix, distribution parameters,
    and optionally Vegas sensitivity coefficients.
    """
    corr_df = pd.read_csv(corr_path, index_col=0)
    dist_df = pd.read_csv(dist_path).set_index('cluster')
    sensitivity = {k: v for k, v in DEFAULT_SENSITIVITY.items()}
    if sensitivity_path:
        try:
            with open(sensitivity_path) as f:
                data = json.load(f)
            # JSON stores {'cluster': {'beta_total': x, 'beta_spread': y}}
            for cluster, vals in data.get('clusters', {}).items():
                if isinstance(vals, dict):
                    sensitivity[cluster] = (vals.get('beta_total', 0.0),
                                            vals.get('beta_spread', 0.0))
                else:
                    sensitivity[cluster] = (vals, 0.0)  # legacy single-beta format
        except FileNotFoundError:
            pass
    return corr_df, dist_df, sensitivity

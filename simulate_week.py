"""
Weekly NFL fantasy simulator.

Reads projections from inputs/projections/week{N}/{platform}.csv.
  DK: QB, RB, WR, TE, DST — salary + projection in one file
  FD: QB, RB, WR, TE (no DST)

cluster assignments come from inputs/current_rosters.csv
  → run `python scripts/generate_rosters.py --week N --platform dk` first

Output: outputs/week_{N}/{platform}_sims.csv
  Columns: sim_no, player, team, position, ppr
  Rows: N_SIMS × N_players (one random draw per sim per player)

Usage:
  python simulate_week.py --week 1 --platform dk
  python simulate_week.py --week 1 --platform fd --sims 50000
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats

from functions.simulator         import _nearest_psd_corr, load_simulation_inputs
from functions.load_projections  import load_projections, projections_path

ROSTERS_PATH     = 'inputs/current_rosters.csv'
CORR_PATH        = 'outputs/cluster_correlation_matrix.csv'
DIST_PATH        = 'outputs/cluster_distributions.csv'
SENSITIVITY_PATH = 'inputs/vegas_sensitivity.json'
OUTPUT_BASE      = 'outputs'

AVG_IMPLIED      = 22.42
AVG_SPREAD       = 0.12
N_SIMS_DEFAULT   = 10_000


def _fit_player_dist(a, loc_c, scale_c, proj, ceiling, p_ceiling=0.90):
    """Fit (loc, scale) so skewnorm(a) has mean=proj and ppf(p_ceiling)=ceiling."""
    if ceiling is None or ceiling <= proj:
        return loc_c, scale_c
    try:
        delta     = a / np.sqrt(1 + a ** 2)
        mean_unit = delta * np.sqrt(2 / np.pi)
        z_ceil    = float(stats.skewnorm.ppf(p_ceiling, a, 0, 1))
        denom     = z_ceil - mean_unit
        if abs(denom) < 1e-6:
            return loc_c, scale_c
        scale_p = (ceiling - proj) / denom
        if scale_p <= 0:
            return loc_c, scale_c
        loc_p = proj - scale_p * mean_unit
        return loc_p, scale_p
    except Exception:
        return loc_c, scale_c


def _sample(cluster, dist_df, u, proj=None, ceiling=None):
    if cluster not in dist_df.index:
        return np.zeros(len(u))
    r     = dist_df.loc[cluster]
    a     = float(r["skew_a"])
    loc   = float(r["skew_loc"])
    scale = float(r["skew_scale"])
    model = str(r.get("model", "skewnorm"))
    pz    = float(r["p_zero"]) if not pd.isna(r.get("p_zero", float("nan"))) else 0.0

    if proj is not None and ceiling is not None:
        # Player-specific distribution: fit (loc, scale) from proj + ceiling.
        # The cluster's skew shape (a) is preserved — only center and spread change.
        loc, scale = _fit_player_dist(a, loc, scale, proj, ceiling)
    elif proj is not None:
        # No ceiling: just shift mean to projection, keep cluster variance
        cm    = _cluster_mean(cluster, dist_df)
        shift = proj - cm if cm != 0 else 0.0
        loc   = loc + shift

    if model == "hurdle":
        vals = np.where(u <= pz, 0.0,
                        stats.skewnorm.ppf((u - pz) / (1 - pz + 1e-12), a, loc, scale))
    else:
        vals = stats.skewnorm.ppf(u, a, loc, scale)
    floor = -15 if "DST" in cluster else 0
    return np.clip(vals, floor, None)


def _cluster_mean(cluster, dist_df):
    if cluster not in dist_df.index: return 0.0
    r   = dist_df.loc[cluster]
    pz  = float(r['p_zero']) if not pd.isna(r.get('p_zero', float('nan'))) else 0.0
    raw = float(stats.skewnorm.mean(float(r['skew_a']),
                                     float(r['skew_loc']),
                                     float(r['skew_scale'])))
    return (1 - pz) * raw if str(r.get('model', 'skewnorm')) == 'hurdle' else raw


def _get_cluster(player_name, position, rosters, available):
    """cluster_override → effective_cluster → cluster → position fallback."""
    if position == 'DST':
        row = rosters[rosters['player_name'] == player_name]
        if not row.empty:
            for col in ['cluster_override', 'effective_cluster', 'cluster']:
                c = str(row.iloc[0].get(col, '') or '').strip()
                if c and c in available and 'DST' in c:
                    return c
        return 'DST_average' if 'DST_average' in available else None

    row = rosters[rosters['player_name'] == player_name]
    if not row.empty:
        for col in ['cluster_override', 'effective_cluster', 'cluster']:
            c = str(row.iloc[0].get(col, '') or '').strip()
            if c and c in available:
                return c

    fallback = {'QB': 'QB_pocket', 'RB': 'RB_committee_featured',
                'WR': 'WR_WR2_s0', 'TE': 'TE_TE1_s1'}
    c = fallback.get(position)
    return c if c and c in available else None


def simulate_week(week: int, platform: str, n_sims: int = N_SIMS_DEFAULT) -> pd.DataFrame:
    rng = np.random.default_rng(42 + week)

    print(f"\n{'='*60}")
    print(f"NFL WEEK {week} — {platform.upper()}  ({n_sims:,} draws per player)")
    print(f"{'='*60}")

    # ── Load simulation inputs ─────────────────────────────────────────────
    corr_df, dist_df, sensitivity = load_simulation_inputs(
        CORR_PATH, DIST_PATH, SENSITIVITY_PATH)
    available = set(dist_df.index)

    # ── Load projections ───────────────────────────────────────────────────
    proj_path = projections_path(week, platform)
    print(f"\nLoading projections from {proj_path}...")
    proj = load_projections(proj_path, platform=platform)

    # ── Load rosters for cluster assignments ───────────────────────────────
    rosters = pd.read_csv(ROSTERS_PATH, dtype=str)
    for col in ['cluster_override', 'effective_cluster', 'cluster']:
        rosters[col] = rosters[col].fillna('').str.strip()
    rosters['player_name'] = rosters['player_name'].str.strip()

    proj['sim_cluster'] = proj.apply(
        lambda r: _get_cluster(r['player_name'], r['position'], rosters, available),
        axis=1
    )

    skipped = proj[proj['sim_cluster'].isna()]['player_name'].tolist()
    if skipped:
        print(f"  No cluster for {len(skipped)} players (skipped): {skipped[:5]}")
    proj = proj[proj['sim_cluster'].notna()].copy()

    # ── Build game matchups from home/away columns ─────────────────────────
    # Group players by (away_team, home_team)
    games_dict = {}
    for _, p in proj.iterrows():
        team    = p['team']
        opp     = p['opponent']
        is_home = bool(p['is_home'])
        key = (opp, team) if is_home else (team, opp)
        games_dict.setdefault(key, []).append(p.to_dict())

    games = [{'away_team': k[0], 'home_team': k[1], 'players': v}
             for k, v in games_dict.items()]
    print(f"Games: {len(games)}")

    # ── Simulate each game ─────────────────────────────────────────────────
    all_results = []

    for game in games:
        ht  = game['home_team']
        at  = game['away_team']
        gps = game['players']
        n_gp = len(gps)
        print(f"\n  {at} @ {ht}  ({n_gp} players)")

        # Correlation matrix
        C = np.eye(n_gp)
        for i in range(n_gp):
            for j in range(i + 1, n_gp):
                ci   = gps[i]['sim_cluster']
                cj   = gps[j]['sim_cluster']
                same = gps[i]['team'] == gps[j]['team']
                ki   = f"own_{ci}"
                kj   = f"{'own' if same else 'opp'}_{cj}"
                if ki in corr_df.index and kj in corr_df.columns:
                    c = float(corr_df.loc[ki, kj])
                    C[i, j] = c; C[j, i] = c

        L = np.linalg.cholesky(_nearest_psd_corr(C))
        z = rng.standard_normal((n_sims, n_gp))
        u = stats.norm.cdf(z @ L.T)

        game_sims = np.zeros((n_sims, n_gp), dtype=np.float32)
        for gi, player in enumerate(gps):
            cluster = player['sim_cluster']
            proj_   = float(player['projected_ppr'])
            cm      = _cluster_mean(cluster, dist_df)
            ceil_   = float(player.get("ceiling", 0) or 0)
            ceil_   = ceil_ if ceil_ > proj_ else None
            game_sims[:, gi] = _sample(cluster, dist_df, u[:, gi],
                                        proj=proj_,
                                        ceiling=ceil_).astype(np.float32)

        # ── QB-receiver consistency pass ──────────────────────────────────────
        # QB passing yards ≈ total team receiving yards — the copula doesn't
        # enforce this so we blend the QB's draw toward what his receivers imply.
        # Ratio: QB gets 0.04 pts/yard, receivers get 0.10 pts/yard → QB ≈ 0.40×
        # of receiver PPR from yards; add rushing/TD variance on top.
        # Blend: 60% copula draw + 40% receiver-implied QB score.
        # QB-receiver consistency: QB passing yards ≈ total receiver yards.
        # Blend alpha varies by cluster — rushing QBs get a looser constraint
        # because a big chunk of their score comes independently from their legs.
        #
        # Estimated rushing fraction by cluster (based on historical production):
        #   QB_dual_threat_elite: ~45% rushing → blend alpha = 0.15 (very loose)
        #   QB_rushing:           ~25% rushing → blend alpha = 0.25
        #   QB_pocket:            ~10% rushing → blend alpha = 0.40 (tightest)
        #   QB_backup:             any          → 0.00 (not worth constraining)
        # QB-receiver symmetric constraint:
        # Passing yards = receiving yards, so QB and receivers must be consistent.
        # We enforce: QB ∈ [rec_total × ratio - buffer, rec_total × ratio + buffer]
        #
        # ratio by cluster (reflects what fraction of receiver PPR the QB earns
        # from passing, given QB gets 0.04/yd vs receiver 0.10/yd + reception):
        #   QB_pocket:            0.35  — ~90% passing, tight constraint
        #   QB_rushing:           0.27  — ~25% rushing, looser
        #   QB_dual_threat_elite: 0.18  — ~45% rushing, loosest
        #
        # buffer = ±5 PPR: allows QB some independent variance within each scenario.
        #
        # Example (QB_pocket, receivers = 97 PPR like Chase 45 game):
        #   Center = 97 × 0.35 = 34.0
        #   Window = [29.0, 39.0]  → Burrow at 29-39, not 16
        # Example (receivers = 67 PPR, on-projection):
        #   Window = [18.5, 28.5]  → normal QB game, rarely binding
        # Example (receivers = 35 PPR, quiet day):
        #   Window = [7.25, 17.25] → QB also quiet, makes sense
        # QB-receiver symmetric constraint.
        # Center: proj × (total_rec_draw / total_rec_proj)
        #   — when receivers score at projection → center = proj (mean-preserving)
        #   — when receivers 50% above → center = 1.5 × proj
        #   — correctly handles rushing QBs since their own proj already includes rushing
        # Buffer: QB can deviate ±buffer from center within each receiver-total scenario.
        #   Pocket QBs: tight (±5) — nearly all production tied to passing
        #   Rushing QBs: moderate (±8) — more independent leg contribution
        #   Dual-threat elite: loose (±12) — major rushing independence
        QB_BUFFERS = {
            'QB_pocket':            5,
            'QB_rushing':           8,
            'QB_dual_threat_elite': 12,
        }

        for gi, player in enumerate(gps):
            if player['position'] != 'QB':
                continue
            cluster = player['sim_cluster']
            buf     = QB_BUFFERS.get(cluster)
            if buf is None:
                continue

            team    = player['team']
            proj    = float(player['projected_ppr'])
            rec_idx = [j for j, p in enumerate(gps)
                       if p['team'] == team and p['position'] in ('WR','TE','RB')]
            if not rec_idx:
                continue

            total_rec_proj = sum(float(gps[j]['projected_ppr']) for j in rec_idx)
            if total_rec_proj <= 0:
                continue

            total_rec  = game_sims[:, rec_idx].sum(axis=1)
            rec_scale  = np.clip(total_rec / total_rec_proj, 0.15, 3.0)
            center     = proj * rec_scale     # scales QB's own projection by receiver deviation
            qb_floor   = np.maximum(0.0, center - buf)
            qb_cap     = center + buf

            game_sims[:, gi] = np.clip(game_sims[:, gi], qb_floor, qb_cap).astype(np.float32)

        for gi, player in enumerate(gps):
            pprs = game_sims[:, gi]
            for sn, ppr in enumerate(pprs, 1):
                all_results.append({
                    'sim_no':   sn,
                    'player':   player['player_name'],
                    'team':     player['team'],
                    'position': player['position'],
                    'salary':   player['salary'],
                    'ppr':      round(float(ppr), 2),
                })

    if not all_results:
        print("No results generated.")
        return pd.DataFrame()

    results = pd.DataFrame(all_results)

    # ── Save to outputs/week_{N}/ ──────────────────────────────────────────
    week_dir = os.path.join(OUTPUT_BASE, f'week_{week}')
    os.makedirs(week_dir, exist_ok=True)
    out_path = os.path.join(week_dir, f'{platform}_sims.csv')
    results.to_csv(out_path, index=False)

    # ── Summary ────────────────────────────────────────────────────────────
    summary = (results.groupby(['player', 'position', 'team'])['ppr']
               .agg(mean='mean',
                    p10=lambda x: np.percentile(x, 10),
                    p50='median',
                    p90=lambda x: np.percentile(x, 90))
               .reset_index()
               .sort_values('mean', ascending=False))

    print(f"\n{'='*65}")
    print(f"WEEK {week} {platform.upper()}  |  {results['player'].nunique()} players  |  {n_sims:,} sims")
    print(f"{'='*65}")
    print(f"{'Player':<28}{'Pos':<5}{'Team':<5}{'Mean':>7}{'P10':>7}{'P50':>7}{'P90':>7}")
    print("-"*60)
    for _, r in summary.head(30).iterrows():
        print(f"  {r['player']:<26}{r['position']:<5}{r['team']:<5}"
              f"{r['mean']:>7.1f}{r['p10']:>7.1f}{r['p50']:>7.1f}{r['p90']:>7.1f}")

    print(f"\nSaved: {out_path}  ({len(results):,} rows)")
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--week',     type=int, default=1)
    p.add_argument('--platform', choices=['dk', 'fd'], default='dk')
    p.add_argument('--sims',     type=int, default=N_SIMS_DEFAULT)
    a = p.parse_args()
    simulate_week(week=a.week, platform=a.platform, n_sims=a.sims)

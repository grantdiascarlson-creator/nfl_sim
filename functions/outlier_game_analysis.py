"""
Outlier game analysis: identify games where a player significantly outperformed
their season-level cluster expectation, then check whether those games coincided
with fewer higher-tier teammates active on the same team.

The hypothesis: a WR3's 25-point game is more likely to be explained by their
team's WR1/WR2 being injured than by random variance. If true, these games are
contaminating the cluster PPR distributions with injury-fill signal that shouldn't
be there, and we should either remove them from distribution fitting or model them
separately as a distinct "injury elevation" mode.

Output:
  - outputs/outlier_games.csv: every flagged outlier game with context
  - outputs/outlier_summary.csv: per-cluster summary of injury-fill contamination rate
  - Console report with the most interesting findings
"""

import pandas as pd
import numpy as np


# ── Cluster hierarchy per position ───────────────────────────────────────────────
# For each cluster, "higher_tier_clusters" = clusters that represent a more featured
# role on the same team. If fewer of these are active in a given game, the player
# may be receiving elevated opportunity due to injury.

CLUSTER_HIERARCHY = {
    # WR: check for absence of higher-tier WRs on same team
    'WR_WR3_rotational':  ['WR_alpha_possession', 'WR_alpha_deep', 'WR_WR2', 'WR_possession_WR2'],
    'WR_deep_threat':     ['WR_alpha_possession', 'WR_alpha_deep', 'WR_WR2', 'WR_possession_WR2'],
    'WR_depth':           ['WR_alpha_possession', 'WR_alpha_deep', 'WR_WW2', 'WR_possession_WR2', 'WR_WR3_rotational'],
    'WR_WW2':             ['WR_alpha_possession', 'WR_alpha_deep'],
    'WR_possession_WR2':  ['WR_alpha_possession', 'WR_alpha_deep'],

    # RB: check for absence of higher-tier RBs on same team
    'RB_committee_early_down': ['RB_bellcow', 'RB_workhorse_dual', 'RB_workhorse_runner', 'RB_committee_featured'],
    'RB_committee_featured':   ['RB_bellcow', 'RB_workhorse_dual', 'RB_workhorse_runner'],
    'RB_goalline':             ['RB_bellcow', 'RB_workhorse_dual', 'RB_workhorse_runner'],
    'RB_passing_down':         ['RB_bellcow', 'RB_workhorse_dual', 'RB_workhorse_runner'],
    'RB_backup':               ['RB_bellcow', 'RB_workhorse_dual', 'RB_workhorse_runner',
                                'RB_committee_featured', 'RB_committee_early_down'],

    # TE: check for absence of higher-tier TEs on same team
    'TE_TE1':             ['TE_alpha'],
    'TE_TE2_rotational':  ['TE_alpha', 'TE_TE1'],
    'TE_depth':           ['TE_alpha', 'TE_TE1', 'TE_TE2_rotational'],
}


def run_outlier_analysis(
    weekly_path: str,
    profile_path: str,
    outlier_games_path: str,
    outlier_summary_path: str,
    outlier_z_threshold: float = 2.0,
    min_ppr: float = 5.0,
) -> tuple:
    """
    Identify outlier games per cluster and check for injury-fill context.

    A game is flagged as an outlier if the player's PPR output is:
      - More than outlier_z_threshold standard deviations above their cluster mean
      - AND at least min_ppr fantasy points (avoids flagging trivial high-z games
        for depth players where the mean is near zero)

    For each outlier game, we count how many players from higher-tier clusters
    on the same team were active that week (appeared in the weekly data).
    We compare this to the player's season baseline (avg higher-tier teammates
    across all their games that season) to compute a "teammate absence" score.

    Args:
        weekly_path:          path to enriched weekly skill CSV
        profile_path:         path to player-season profile CSV with cluster labels
        outlier_games_path:   path to write outlier games CSV
        outlier_summary_path: path to write per-cluster summary CSV
        outlier_z_threshold:  z-score cutoff for flagging outlier games
        min_ppr:              minimum PPR to consider a game an outlier

    Returns:
        (outlier_games_df, summary_df)
    """
    print("Loading data...")
    weekly  = pd.read_csv(weekly_path)
    profile = pd.read_csv(profile_path)

    # Attach cluster to every player-week
    weekly = weekly.merge(
        profile[['player_id', 'season', 'cluster']],
        on=['player_id', 'season'], how='inner'
    )

    # Compute cluster-level mean and std across all player-weeks (for z-score)
    cluster_stats = (
        weekly.groupby('cluster')['fantasy_points_ppr']
        .agg(cluster_mean='mean', cluster_std='std')
        .reset_index()
    )
    weekly = weekly.merge(cluster_stats, on='cluster', how='left')
    weekly['z_score'] = (
        (weekly['fantasy_points_ppr'] - weekly['cluster_mean']) / weekly['cluster_std']
    )

    # Build a lookup: for each (game_key, team), which clusters were active?
    # "active" = appeared in the weekly data (targets > 0 OR carries > 0)
    active = weekly[
        (weekly['targets'] > 0) | (weekly['carries'] > 0) | (weekly['attempts'] > 0)
    ].copy()
    game_team_clusters = (
        active.groupby(['game_key', 'team'])['cluster']
        .apply(set)
        .reset_index()
        .rename(columns={'cluster': 'active_clusters'})
    )

    # Flag outlier games: z >= threshold AND ppr >= min_ppr AND cluster has a hierarchy defined
    outlier_mask = (
        (weekly['z_score'] >= outlier_z_threshold) &
        (weekly['fantasy_points_ppr'] >= min_ppr) &
        (weekly['cluster'].isin(CLUSTER_HIERARCHY.keys()))
    )
    outliers = weekly[outlier_mask].copy()

    print(f"Flagged {len(outliers):,} outlier player-games across "
          f"{outliers['cluster'].nunique()} clusters")

    # For each outlier game, count active higher-tier teammates
    outliers = outliers.merge(game_team_clusters, on=['game_key', 'team'], how='left')

    def count_higher_tier_active(row):
        higher = CLUSTER_HIERARCHY.get(row['cluster'], [])
        active = row['active_clusters'] if isinstance(row['active_clusters'], set) else set()
        return sum(1 for c in higher if c in active)

    outliers['higher_tier_active'] = outliers.apply(count_higher_tier_active, axis=1)
    outliers['max_possible_higher_tier'] = outliers['cluster'].map(
        lambda c: len(CLUSTER_HIERARCHY.get(c, []))
    )

    # Compute each player's season baseline: avg higher-tier teammates across all games
    # (not just outlier games) so we can compare outlier vs normal games
    all_games = weekly[weekly['cluster'].isin(CLUSTER_HIERARCHY.keys())].copy()
    all_games = all_games.merge(game_team_clusters, on=['game_key', 'team'], how='left')
    all_games['higher_tier_active'] = all_games.apply(count_higher_tier_active, axis=1)

    season_baseline = (
        all_games.groupby(['player_id', 'season'])['higher_tier_active']
        .mean()
        .reset_index()
        .rename(columns={'higher_tier_active': 'season_avg_higher_tier_active'})
    )
    outliers = outliers.merge(season_baseline, on=['player_id', 'season'], how='left')
    outliers['higher_tier_deficit'] = (
        outliers['season_avg_higher_tier_active'] - outliers['higher_tier_active']
    )

    # Flag as injury-fill if higher-tier teammates notably below season average
    outliers['injury_fill_flag'] = outliers['higher_tier_deficit'] >= 0.5

    # Clean up output columns
    outlier_out = outliers[[
        'player_display_name', 'position', 'cluster', 'season', 'week',
        'team', 'opponent_team', 'fantasy_points_ppr', 'z_score',
        'higher_tier_active', 'season_avg_higher_tier_active',
        'higher_tier_deficit', 'injury_fill_flag',
    ]].sort_values(['cluster', 'z_score'], ascending=[True, False])

    outlier_out.to_csv(outlier_games_path, index=False)

    # ── Per-cluster summary ───────────────────────────────────────────────────
    summary = (
        outliers.groupby('cluster')
        .agg(
            n_outlier_games=('player_display_name', 'count'),
            pct_injury_fill=('injury_fill_flag', 'mean'),
            avg_z_score=('z_score', 'mean'),
            avg_ppr=('fantasy_points_ppr', 'mean'),
            avg_higher_tier_active_outlier=('higher_tier_active', 'mean'),
            avg_higher_tier_active_baseline=('season_avg_higher_tier_active', 'mean'),
            avg_deficit=('higher_tier_deficit', 'mean'),
        )
        .reset_index()
        .sort_values('pct_injury_fill', ascending=False)
    )
    summary['pct_injury_fill'] = summary['pct_injury_fill'].round(3)
    summary['avg_z_score'] = summary['avg_z_score'].round(2)
    summary['avg_ppr'] = summary['avg_ppr'].round(1)
    summary['avg_deficit'] = summary['avg_deficit'].round(2)

    summary.to_csv(outlier_summary_path, index=False)

    # ── Console report ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("OUTLIER GAME ANALYSIS — Injury-Fill Contamination by Cluster")
    print(f"(z >= {outlier_z_threshold}, PPR >= {min_ppr})")
    print("=" * 70)
    print(f"\n{'Cluster':<30} {'N':>5} {'%InjFill':>9} {'AvgZ':>6} "
          f"{'AvgPPR':>7} {'HigherTier(Out)':>15} {'Baseline':>9} {'Deficit':>8}")
    print("-" * 90)
    for _, r in summary.iterrows():
        print(f"  {r['cluster']:<28} {int(r['n_outlier_games']):>5} "
              f"{r['pct_injury_fill']:>9.1%} {r['avg_z_score']:>6.2f} "
              f"{r['avg_ppr']:>7.1f} {r['avg_higher_tier_active_outlier']:>15.2f} "
              f"{r['avg_higher_tier_active_baseline']:>9.2f} {r['avg_deficit']:>8.2f}")

    # Show the most interesting specific examples
    print("\n\nTOP INJURY-FILL OUTLIER GAMES (by deficit + z-score):")
    print("-" * 70)
    top = (
        outliers[outliers['injury_fill_flag']]
        .nlargest(20, 'z_score')
        [['player_display_name', 'cluster', 'season', 'week', 'team',
          'fantasy_points_ppr', 'z_score', 'higher_tier_active',
          'season_avg_higher_tier_active', 'higher_tier_deficit']]
    )
    for _, r in top.iterrows():
        print(f"  {r['player_display_name']:<25} {r['cluster']:<28} "
              f"{int(r['season'])} W{int(r['week']):<2} "
              f"PPR={r['fantasy_points_ppr']:>5.1f}  z={r['z_score']:>4.1f}  "
              f"higher_active={r['higher_tier_active']:.0f} "
              f"(baseline={r['season_avg_higher_tier_active']:.1f})")

    print(f"\nSaved outlier games to {outlier_games_path}")
    print(f"Saved summary to {outlier_summary_path}")

    return outlier_out, summary

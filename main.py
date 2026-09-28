"""
NFL Season Simulator — v1
=========================
Pipeline: raw data -> enrich (snap share + red zone) -> cluster (rule + kmeans)
       -> role panel -> correlations -> distributions -> game simulation

Usage:
    python main.py                    # full pipeline + example simulation
    python main.py --skip-pipeline    # skip to simulation (after first run)

Inputs (place in inputs/):
    player_stats_2019_2025.csv
    snap_counts_2019_2025.csv
    rz_targets_2019_2025.csv
    rz_carries_2019_2025.csv
"""

import os, sys
import pandas as pd

from functions.prep_data       import process_data
from functions.enrich_features import enrich_features
from functions.cluster_players import assign_clusters
from functions.build_role_panel    import build_role_panel
from functions.correlation_matrix  import build_correlation_matrix
from functions.fit_distributions   import fit_distributions
from functions.simulator           import simulate_game, load_simulation_inputs
from load_schedule                 import load_schedule
from functions.outlier_game_analysis import run_outlier_analysis

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_PATH      = 'inputs/player_stats_2019_2025.csv'
SNAP_PATH     = 'inputs/snap_counts_2019_2025.csv'
RZ_TGT_PATH   = 'inputs/rz_targets_2019_2025.csv'
RZ_CAR_PATH   = 'inputs/rz_carries_2019_2025.csv'

WEEKLY_PATH   = 'outputs/weekly_skill.csv'
ENRICHED_PATH = 'outputs/weekly_skill_enriched.csv'
PROFILE_PATH  = 'outputs/player_season_profile.csv'
PANEL_PATH    = 'outputs/role_slot_panel.csv'
CORR_PATH     = 'outputs/cluster_correlation_matrix.csv'
NOBS_PATH     = 'outputs/cluster_correlation_nobs.csv'
DIST_PATH     = 'outputs/cluster_distributions.csv'

OUTLIER_GAMES_PATH   = 'outputs/outlier_games.csv'
OUTLIER_SUMMARY_PATH = 'outputs/outlier_summary.csv'

os.makedirs('inputs', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ── Which cluster column to use downstream (rule or kmeans) ───────────────────
# Change to 'cluster_kmeans' to run the full pipeline on k-means labels instead
ACTIVE_CLUSTER = 'cluster_rule'


def run_pipeline():
    for path, label in [
        (RAW_PATH,    'player_stats_2019_2025.csv'),
        (SNAP_PATH,   'snap_counts_2019_2025.csv'),
        (RZ_TGT_PATH, 'rz_targets_2019_2025.csv'),
        (RZ_CAR_PATH, 'rz_carries_2019_2025.csv'),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: '{label}' not found at {path}")
            sys.exit(1)

    print("=" * 60)
    print("NFL Simulator — Data Pipeline")
    print("=" * 60)

    process_data(raw_path=RAW_PATH, out_path=WEEKLY_PATH)

    enrich_features(
        weekly_path=WEEKLY_PATH,
        snap_path=SNAP_PATH,
        rz_targets_path=RZ_TGT_PATH,
        rz_carries_path=RZ_CAR_PATH,
        out_path=ENRICHED_PATH,
    )

    profile = assign_clusters(
        weekly_path=ENRICHED_PATH,
        out_path=PROFILE_PATH,
        min_games=4,
    )

    # Use the active cluster column downstream
    if ACTIVE_CLUSTER != 'cluster_rule':
        profile = profile.rename(columns={ACTIVE_CLUSTER: 'cluster'})
    else:
        profile = profile.rename(columns={'cluster_rule': 'cluster'})
    profile.to_csv(PROFILE_PATH, index=False)

    build_role_panel(
        weekly_path=ENRICHED_PATH,
        profile_path=PROFILE_PATH,
        out_path=PANEL_PATH,
    )

    build_correlation_matrix(
        panel_path=PANEL_PATH,
        corr_out_path=CORR_PATH,
        nobs_out_path=NOBS_PATH,
    )

    fit_distributions(
        weekly_path=ENRICHED_PATH,
        profile_path=PROFILE_PATH,
        out_path=DIST_PATH,
    )

    print("\nPipeline complete.")


def run_example_simulation():
    print("\n" + "=" * 60)
    print("NFL Simulator — Example Game Simulation")
    print("=" * 60)

    corr_df, dist_df, beta = load_simulation_inputs(CORR_PATH, DIST_PATH, 'inputs/vegas_sensitivity.json')

    # ── Vegas lines — set per game ─────────────────────────────────────────────
    # total_line:  O/U total, spread_line: home team spread (+ = home underdog)
    TOTAL_LINE  = 52.5   # example: CIN vs TB week 1
    SPREAD_LINE = 3.5    # home is +3.5 underdog

    # ── Edit this roster — include 'home' True/False per player ───────────────
    roster = [
        # Team A (home)
        {'player': 'A_QB1',    'team': 'A', 'cluster': 'QB_rushing',              'home': True},
        {'player': 'A_QB_bkp', 'team': 'A', 'cluster': 'QB_backup',               'home': True},
        {'player': 'A_RB1',    'team': 'A', 'cluster': 'RB_bellcow_runner',       'home': True},
        {'player': 'A_RB_bkp', 'team': 'A', 'cluster': 'RB_backup',               'home': True},
        {'player': 'A_WR1',    'team': 'A', 'cluster': 'WR_alpha_possession',     'home': True},
        {'player': 'A_WR2',    'team': 'A', 'cluster': 'WR_WR2_s0',                  'home': True},
        {'player': 'A_TE1',    'team': 'A', 'cluster': 'TE_TE1_s0',                  'home': True},
        # Team B (away)
        {'player': 'B_QB1',    'team': 'B', 'cluster': 'QB_rushing',              'home': False},
        {'player': 'B_QB_bkp', 'team': 'B', 'cluster': 'QB_backup',               'home': False},
        {'player': 'B_RB1',    'team': 'B', 'cluster': 'RB_workhorse_runner',     'home': False},
        {'player': 'B_RB_bkp', 'team': 'B', 'cluster': 'RB_committee_early_down_s0','home': False},
        {'player': 'B_WR1',    'team': 'B', 'cluster': 'WR_alpha_deep',           'home': False},
        {'player': 'B_TE1',    'team': 'B', 'cluster': 'TE_alpha',                'home': False},
    ]

    # Starter-backup pairs for injury timing model
    pairs = [
        {'starter': 'A_QB1', 'backup': 'A_QB_bkp', 'elevated_cluster': 'QB_rushing'},
        {'starter': 'A_RB1', 'backup': 'A_RB_bkp', 'elevated_cluster': 'RB_bellcow_runner'},
        {'starter': 'B_QB1', 'backup': 'B_QB_bkp', 'elevated_cluster': 'QB_rushing'},
        {'starter': 'B_RB1', 'backup': 'B_RB_bkp', 'elevated_cluster': 'RB_workhorse_runner'},
    ]
    # ──────────────────────────────────────────────────────────────────────────

    available = set(dist_df.index)
    bad = [p for p in roster if p['cluster'] not in available]
    if bad:
        print("ERROR: Stale cluster names in roster:")
        for p in bad:
            print(f"  {p['player']}: '{p['cluster']}'")
        print(f"\nAvailable: {sorted(available)}")
        return

    print(f"\nSimulating {len(roster)} players (20,000 iterations)...")
    print(f"Vegas: total={TOTAL_LINE}, spread={SPREAD_LINE:+.1f} (home perspective)")
    sims = simulate_game(roster=roster, corr_df=corr_df, dist_df=dist_df,
                         n_sims=20_000, pairs=pairs, beta=beta,
                         total_line=TOTAL_LINE, spread_line=SPREAD_LINE)

    print("\nSimulated PPR — mean / std / P10 / P50 / P90:")
    summary = sims.describe(percentiles=[0.10, 0.50, 0.90]).T[['mean', 'std', '10%', '50%', '90%']]
    summary.columns = ['Mean', 'Std', 'P10', 'Median', 'P90']
    print(summary.round(1).to_string())

    sims.to_csv('outputs/example_game_simulation.csv', index=False)
    print("\nFull simulation saved to outputs/example_game_simulation.csv")


if __name__ == '__main__':
    skip = '--skip-pipeline' in sys.argv
    if not skip:
        run_pipeline()
    else:
        print("Skipping pipeline...")
        for p in [CORR_PATH, DIST_PATH]:
            if not os.path.exists(p):
                print(f"ERROR: {p} not found. Run without --skip-pipeline first.")
                sys.exit(1)

    run_example_simulation()

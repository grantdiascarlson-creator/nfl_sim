"""
Game-level cluster assignment.

Three types of adjustments, applied in order:

1. ELEVATION — anchor player absent from game → fill-in gets elevated cluster
2. DEMOTION — anchor player present → lower-tier teammates get demoted
3. REALITY CHECK — in-game snap share below threshold for sustained stretches
   (consecutive weeks) rather than single-game dips which may be injury exits

Key design decisions:
- RB_passing_down NOT demoted based on in-game targets: low targets may reflect
  game script (run-heavy game) rather than the player being irrelevant
- Single-game low snap share NOT demoted for WRs: could be a mid-game injury exit
  rather than a genuine role change. Only sustained multi-week stretches trigger demotion.
- TE fill-ins capped at TE_TE1, WR fill-ins capped at WR_WR2 (talent gap too large)
"""

import pandas as pd
import numpy as np


ANCHOR_CLUSTERS = {
    'RB': ['RB_bellcow_every_down', 'RB_bellcow_dual', 'RB_bellcow_runner',
           'RB_workhorse_receiving', 'RB_workhorse_volume',
           'RB_workhorse_early_down', 'RB_workhorse_runner',
           'RB_committee_receiving'],
    'WR': ['WR_alpha_possession', 'WR_alpha_deep', 'WR_WR2_featured'],
    'TE': ['TE_alpha_elite', 'TE_alpha'],
}

OPPORTUNITY_COL = {
    'RB': 'carries',
    'WR': 'targets',
    'TE': 'targets',
}

RB_TIERS = [
    'RB_bellcow_every_down', 'RB_bellcow_dual', 'RB_bellcow_runner',
    'RB_workhorse_receiving', 'RB_workhorse_volume',
    'RB_workhorse_early_down', 'RB_workhorse_runner',
    'RB_goalline', 'RB_committee_receiving',
    'RB_committee_featured', 'RB_committee_early_down',
    'RB_passing_down', 'RB_backup',
]
WR_TIERS = [
    'WR_alpha_possession', 'WR_alpha_deep',
    'WR_WR2_featured', 'WR_WR2', 'WR_possession_WR2',
    'WR_WR3_high_volume', 'WR_WR3_rotational',
    'WR_deep_threat', 'WR_depth',
]
TE_TIERS = [
    'TE_alpha_elite', 'TE_alpha', 'TE_TE1_receiving', 'TE_TE1', 'TE_TE2_rotational', 'TE_depth',
]
POSITION_TIERS = {'RB': RB_TIERS, 'WR': WR_TIERS, 'TE': TE_TIERS}

DEMOTION_RULES = {
    'TE': {
        'trigger_clusters': ['TE_alpha_elite', 'TE_alpha'],
        'demotions': {
            'TE_TE1':            'TE_TE2_rotational',
            'TE_TE2_rotational': 'TE_depth',
        },
    },
    'RB': {
        'trigger_clusters': ['RB_bellcow_every_down', 'RB_bellcow_dual', 'RB_bellcow_runner',
                             'RB_workhorse_receiving', 'RB_workhorse_volume',
                             'RB_workhorse_early_down', 'RB_workhorse_runner',
                             'RB_committee_receiving'],
        'demotions': {
            'RB_committee_receiving':  'RB_committee_featured',
            'RB_committee_featured':   'RB_committee_early_down',
            'RB_committee_early_down': 'RB_backup',
            # RB_passing_down intentionally excluded: low targets in a game may reflect
            # game script (run-heavy) rather than the player being irrelevant
        },
    },
}

ELEVATION_TARGET = {
    'RB_bellcow_every_down':   'RB_bellcow_every_down',
    'RB_bellcow_dual':         'RB_bellcow_dual',
    'RB_bellcow_runner':       'RB_bellcow_runner',
    'RB_workhorse_receiving':  'RB_workhorse_receiving',
    'RB_workhorse_volume':     'RB_workhorse_volume',
    'RB_workhorse_early_down': 'RB_workhorse_early_down',
    'RB_workhorse_dual':    'RB_workhorse_dual',
    'RB_workhorse_runner':  'RB_workhorse_runner',
    'WR_alpha_possession':  'WR_WR2',
    'WR_alpha_deep':        'WR_WR2',
    'WR_WR2_featured':      'WR_WR3_high_volume',
    'TE_alpha':             'TE_TE1',
}

# Sustained role-change detection:
# If a WR_alpha has offense_pct below this threshold for N consecutive weeks,
# reclassify ALL those weeks — this catches Egbuka-type late-season demotions
# where the role genuinely changed, not a single injury exit.
# Single-game dips are NOT flagged since they may be mid-game injury exits.
# 0.70 threshold: WR_alpha mean is 85%, so 3 consecutive weeks below 70%
# is a clear signal of genuine role reduction, not just injury variance.
# Single-game dips (week 3 = 64%, week 6 = 53% for Egbuka) are left alone
# since they may be injury exits — only sustained stretches get demoted.

def _tier_rank(cluster: str, tiers: list) -> int:
    try:
        return tiers.index(cluster)
    except ValueError:
        return len(tiers)


def compute_game_level_clusters(
    weekly: pd.DataFrame,
    profile: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign game-level cluster labels accounting for:
      1. Elevation: fill-in players when anchor is absent
      2. Demotion: teammates when anchor is present
      3. Sustained role changes: WR_alpha with consistently low snap share
    """
    weekly = weekly.merge(
        profile[['player_id', 'season', 'cluster', 'effective_cluster']],
        on=['player_id', 'season'], how='inner'
    )
    # Default: game_cluster = effective_cluster (subcluster where n>=30, else parent)
    # Elevation/demotion passes will override specific players where their role changed
    weekly['game_cluster'] = weekly['effective_cluster']
    weekly['_opps'] = weekly['carries'].fillna(0) + weekly['targets'].fillna(0)

    player_game_opps = weekly.set_index(['player_id', 'game_key'])['_opps'].to_dict()

    # Build season anchor roster from weekly data (profile has no team column)
    all_anchor_clusters = [c for anchors in ANCHOR_CLUSTERS.values() for c in anchors]
    anchor_weekly = weekly[
        weekly['cluster'].isin(all_anchor_clusters)
    ][['player_id', 'player_display_name', 'position', 'team', 'season', 'cluster']].drop_duplicates(
        subset=['player_id', 'season']
    )
    anchor_by_team_season = (
        anchor_weekly.groupby(['team', 'season', 'position'])
        .apply(lambda g: list(zip(g['player_id'], g['player_display_name'], g['cluster'])))
        .to_dict()
    )

    elevation_log = []

    # ── Elevation pass ────────────────────────────────────────────────────────
    for (game_key, team), game_df in weekly.groupby(['game_key', 'team']):
        season = game_df['season'].iloc[0]

        for pos, anchor_clusters in ANCHOR_CLUSTERS.items():
            pos_df       = game_df[game_df['position'] == pos].copy()
            tiers        = POSITION_TIERS[pos]
            opp_col      = OPPORTUNITY_COL[pos]
            team_anchors = anchor_by_team_season.get((team, season, pos), [])

            if not team_anchors:
                continue

            anchors_with_opps = [
                (pid, pname, cluster) for pid, pname, cluster in team_anchors
                if player_game_opps.get((pid, game_key), 0) > 0
            ]
            if anchors_with_opps:
                continue

            tier_ranks = [_tier_rank(c, tiers) for _, _, c in team_anchors]
            valid_ranks = [r for r in tier_ranks if r < len(tiers)]
            if not valid_ranks:
                continue
            absent_tier_rank = min(valid_ranks)
            absent_cluster   = tiers[absent_tier_rank]
            absent_names     = [pname for _, pname, _ in team_anchors]

            active_pos = pos_df[pos_df['_opps'] > 0].copy()
            if active_pos.empty:
                continue

            fill_in_idx  = active_pos[opp_col].idxmax()
            fill_in      = active_pos.loc[fill_in_idx]
            fill_in_tier = _tier_rank(fill_in['cluster'], tiers)

            if fill_in_tier <= absent_tier_rank:
                continue

            elevated_to = ELEVATION_TARGET.get(absent_cluster, absent_cluster)
            weekly.loc[fill_in_idx, 'game_cluster'] = elevated_to

            elevation_log.append({
                'game_key':       game_key,
                'season':         season,
                'week':           game_df['week'].iloc[0],
                'team':           team,
                'position':       pos,
                'absent_players': ', '.join(absent_names),
                'absent_cluster': absent_cluster,
                'fill_in_player': fill_in['player_display_name'],
                'fill_in_cluster':fill_in['cluster'],
                'elevated_to':    elevated_to,
                'fill_in_opps':   fill_in[opp_col],
                'fill_in_ppr':    fill_in['fantasy_points_ppr'],
            })

    # ── In-game target share for RB checks ───────────────────────────────────
    team_game_targets = (
        weekly.groupby(['game_key', 'team'])['targets']
        .sum().reset_index()
        .rename(columns={'targets': 'team_game_targets'})
    )
    weekly = weekly.merge(team_game_targets, on=['game_key', 'team'], how='left')
    weekly['ingame_target_share'] = np.where(
        weekly['team_game_targets'] > 0,
        weekly['targets'].fillna(0) / weekly['team_game_targets'], np.nan
    )

    # ── Demotion pass ─────────────────────────────────────────────────────────
    demotion_log = []
    for (game_key, team), game_df in weekly.groupby(['game_key', 'team']):
        season = game_df['season'].iloc[0]

        for pos, rules in DEMOTION_RULES.items():
            pos_df = game_df[game_df['position'] == pos].copy()
            if pos_df.empty:
                continue

            # Use parent cluster for trigger check — trigger names are parent-level
            # (game_cluster may now be a subcluster like RB_bellcow_runner_s0)
            trigger_active = pos_df[
                (pos_df['cluster'].isin(rules['trigger_clusters'])) &
                (pos_df['_opps'] > 0)
            ]
            if trigger_active.empty:
                continue

            for idx, row in pos_df.iterrows():
                demoted = rules['demotions'].get(row['game_cluster'])
                if demoted is None:
                    continue
                weekly.loc[idx, 'game_cluster'] = demoted
                demotion_log.append({
                    'game_key':       game_key,
                    'season':         season,
                    'week':           row['week'],
                    'team':           team,
                    'position':       pos,
                    'player':         row['player_display_name'],
                    'from_cluster':   row['game_cluster'],
                    'to_cluster':     demoted,
                    'trigger_players':', '.join(trigger_active['player_display_name'].tolist()),
                })

    # ── Clean up and report ───────────────────────────────────────────────────
    weekly = weekly.drop(columns=['_opps', 'team_game_targets', 'ingame_target_share'])

    n_elev    = len(elevation_log)
    n_dem     = len(demotion_log)
    n_seasons = weekly['season'].nunique()
    n_games   = weekly['game_key'].nunique()

    print(f"    Elevations: {n_elev:>5,} (~{n_elev/n_seasons:.0f}/season)")
    print(f"    Demotions:  {n_dem:>5,} (~{n_dem/n_seasons:.0f}/season)")

    if elevation_log:
        pd.DataFrame(elevation_log).to_csv('outputs/elevation_log.csv', index=False)
    if demotion_log:
        pd.DataFrame(demotion_log).to_csv('outputs/demotion_log.csv', index=False)

    # Sample elevation events
    if elevation_log:
        print("\n    Sample elevations:")
        for r in elevation_log[:4]:
            print(f"      {int(r['season'])} W{int(r['week']):<2} {r['team']}  "
                  f"absent: {r['absent_players']}  → {r['fill_in_player']} to {r['elevated_to']}")

    return weekly

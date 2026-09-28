"""
Projection engine: converts season-long stat projections to per-game PPR
estimates for use in game simulation.

Two input files:
  inputs/season_projections.csv  — cumulative season totals per player
  inputs/week_projections.csv    — single-week PPR overrides (takes priority)

Week priority (highest to lowest):
  1. week_projections.csv entry for this player (exact PPR, no conversion)
  2. season_projections.csv entry converted to weekly PPR
  3. Cluster mean (default if no projection provided)

Season → game distribution logic:
  Each player's season total is distributed across their 16/17 game schedule
  proportionally to:
    game_weight = timing_weight(week) × vegas_weight(week)
  where:
    timing_weight captures when production arrives in the season
    vegas_weight is the team's implied total that week / season avg implied total

Weighting options:
  uniform       — equal timing weight across all games (Vegas still adjusts)
  frontweighted — veteran/declining: week 1 = 1.5×, week 17 = 0.5× (linear)
  backweighted  — rookie/developing: week 1 = 0.5×, week 17 = 1.5× (linear)
  strong_front  — steep front: week 1 = 2.0×, week 17 = 0.25×
  strong_back   — steep back: week 1 = 0.25×, week 17 = 2.0×

PPR scoring (standard):
  +1.0  per reception
  +0.1  per receiving yard
  +6.0  per receiving TD
  +0.1  per rushing yard
  +6.0  per rushing TD
  +0.04 per passing yard
  +4.0  per passing TD
  -2.0  per interception
"""

import os
import pandas as pd
import numpy as np


SEASON_PROJ_PATH = 'inputs/season_projections.csv'
WEEK_PROJ_PATH   = 'inputs/week_projections.csv'
SCHEDULE_PATH    = 'inputs/schedule_processed.csv'


# ── PPR scoring ───────────────────────────────────────────────────────────────

def stats_to_ppr(
    rec: float = 0, rec_yards: float = 0, rec_tds: float = 0,
    rush_att: float = 0, rush_yards: float = 0, rush_tds: float = 0,
    pass_yards: float = 0, pass_tds: float = 0, pass_int: float = 0,
    **kwargs
) -> float:
    return (
        rec * 1.0 +
        rec_yards * 0.1 +
        rec_tds * 6.0 +
        rush_yards * 0.1 +
        rush_tds * 6.0 +
        pass_yards * 0.04 +
        pass_tds * 4.0 -
        pass_int * 2.0
    )


# ── Timing weight curves ──────────────────────────────────────────────────────

def timing_weight(week: int, total_weeks: int, mode: str) -> float:
    """
    Returns the timing multiplier for a given week.
    Normalisation is done at the schedule level (not here) after Vegas weighting.
    """
    t = (week - 1) / max(total_weeks - 1, 1)  # 0.0 at week 1, 1.0 at final week

    if mode == 'uniform':
        return 1.0
    elif mode == 'frontweighted':
        return 1.5 - t          # 1.5 → 0.5
    elif mode == 'backweighted':
        return 0.5 + t          # 0.5 → 1.5
    elif mode == 'strong_front':
        return 2.0 - 1.75 * t  # 2.0 → 0.25
    elif mode == 'strong_back':
        return 0.25 + 1.75 * t  # 0.25 → 2.0
    else:
        return 1.0  # fallback: uniform


# ── Season → weekly projection ────────────────────────────────────────────────

def season_to_weekly(
    season_proj: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert season cumulative projections to per-game PPR estimates.

    Returns a DataFrame with columns:
        player_name, team, week, projected_ppr, projected_ppr_source
    """
    rows = []

    for _, player in season_proj.iterrows():
        name    = player['player_name']
        team    = player['team']
        mode    = str(player.get('weighting', 'uniform') or 'uniform').strip().lower()
        season_ppr = stats_to_ppr(
            rec       = float(player.get('rec', 0) or 0),
            rec_yards = float(player.get('rec_yards', 0) or 0),
            rec_tds   = float(player.get('rec_tds', 0) or 0),
            rush_att  = float(player.get('rush_att', 0) or 0),
            rush_yards= float(player.get('rush_yards', 0) or 0),
            rush_tds  = float(player.get('rush_tds', 0) or 0),
            pass_yards= float(player.get('pass_yards', 0) or 0),
            pass_tds  = float(player.get('pass_tds', 0) or 0),
            pass_int  = float(player.get('pass_int', 0) or 0),
        )

        # Get this player's games (skip bye weeks)
        team_games = schedule[
            (schedule['home_team'] == team) | (schedule['away_team'] == team)
        ].copy()
        team_games = team_games.sort_values('week')

        if team_games.empty:
            continue

        # Per-game implied total for this team
        team_games['team_implied'] = team_games.apply(
            lambda r: r['home_implied'] if r['home_team'] == team else r['away_implied'],
            axis=1
        )

        # Timing weights
        weeks      = team_games['week'].tolist()
        total_wks  = len(weeks)
        t_weights  = [timing_weight(w, total_wks, mode) for w in weeks]

        # Vegas weights: proportional to implied total
        v_weights  = team_games['team_implied'].tolist()

        # Combined weight = timing × Vegas
        combined   = [t * v for t, v in zip(t_weights, v_weights)]
        total_w    = sum(combined)

        if total_w <= 0:
            continue

        # Distribute season PPR proportionally
        for i, (_, game) in enumerate(team_games.iterrows()):
            game_ppr = season_ppr * combined[i] / total_w
            rows.append({
                'player_name':          name,
                'team':                 team,
                'week':                 game['week'],
                'projected_ppr':        round(game_ppr, 2),
                'projected_ppr_source': 'season',
            })

    return pd.DataFrame(rows)


# ── Load week override projections ────────────────────────────────────────────

def load_week_projections(week: int) -> pd.DataFrame:
    """
    Load single-week PPR overrides. Returns empty DataFrame if file not found.
    """
    # Support both a shared week_projections.csv and week-specific files
    paths_to_try = [
        f'inputs/week_{week}_projections.csv',
        WEEK_PROJ_PATH,
    ]
    for path in paths_to_try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['projected_ppr_source'] = f'week_override ({path})'
            # Filter to this week if there's a week column
            if 'week' in df.columns:
                df = df[df['week'] == week]
            return df
    return pd.DataFrame()


# ── Unified projection lookup ─────────────────────────────────────────────────

class ProjectionEngine:
    """
    Resolves the projected PPR for any player in any week.

    Priority:
      1. Week-level override (week_projections.csv)
      2. Season projection converted to weekly (season_projections.csv)
      3. None (simulator uses cluster mean)
    """

    def __init__(self, week: int):
        self.week = week
        self._schedule = None
        self._season_weekly = None   # player_name → projected_ppr for this week
        self._week_override = {}     # player_name → projected_ppr

        self._load()

    def _load(self):
        # Load schedule
        if os.path.exists(SCHEDULE_PATH):
            self._schedule = pd.read_csv(SCHEDULE_PATH)

        # Load week overrides
        week_df = load_week_projections(self.week)
        if not week_df.empty and 'player_name' in week_df.columns:
            self._week_override = dict(
                zip(week_df['player_name'].str.strip(),
                    week_df['projected_ppr'])
            )
            print(f"  Loaded {len(self._week_override)} week-{self.week} projection overrides")

        # Load and convert season projections
        if os.path.exists(SEASON_PROJ_PATH) and self._schedule is not None:
            season_df = pd.read_csv(SEASON_PROJ_PATH)
            season_df = season_df[season_df['player_name'].notna()].copy()
            season_df['player_name'] = season_df['player_name'].str.strip()

            weekly = season_to_weekly(season_df, self._schedule)
            this_week = weekly[weekly['week'] == self.week]

            self._season_weekly = dict(
                zip(this_week['player_name'], this_week['projected_ppr'])
            )
            n_season = len(self._season_weekly)
            if n_season > 0:
                print(f"  Loaded {n_season} season projections → week {self.week} PPR estimates")

    def get(self, player_name: str) -> tuple:
        """
        Returns (projected_ppr, source) or (None, None) if no projection.
        """
        name = str(player_name).strip()

        if name in self._week_override:
            return float(self._week_override[name]), 'week_override'

        if self._season_weekly and name in self._season_weekly:
            return float(self._season_weekly[name]), 'season_proj'

        return None, None

    def summary(self) -> dict:
        return {
            'week':              self.week,
            'week_overrides':    len(self._week_override),
            'season_projections': len(self._season_weekly) if self._season_weekly else 0,
        }

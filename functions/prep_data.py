import pandas as pd
import numpy as np


def process_data(raw_path: str, out_path: str) -> pd.DataFrame:
    """
    Load raw weekly player stats, filter to skill positions and regular season,
    build a reliable game key, and compute derived features (aDOT, carry share).

    Args:
        raw_path: path to raw player_stats CSV from nflverse
        out_path: path to write cleaned weekly skill player CSV

    Returns:
        Cleaned DataFrame
    """
    print("[1/6] Loading and preparing data...")

    df = pd.read_csv(raw_path, low_memory=False)

    # Regular season only, weeks 1-17 (exclude week 18 garbage time)
    df = df[(df['season_type'] == 'REG') & (df['week'] <= 17)].copy()

    # Skill positions only
    skill_positions = ['QB', 'RB', 'WR', 'TE', 'FB']
    df = df[df['position'].isin(skill_positions)].copy()

    # Build reliable game key (the provided game_id has too many nulls)
    df['game_key'] = (
        df['season'].astype(str) + '_W' + df['week'].astype(str) + '_' +
        df[['team', 'opponent_team']].min(axis=1) + '_' +
        df[['team', 'opponent_team']].max(axis=1)
    )

    # avg depth of target per player-week
    df['adot'] = np.where(df['targets'] > 0, df['receiving_air_yards'] / df['targets'], np.nan)

    # Carry share (not provided in source, derive from team totals)
    team_game_totals = (
        df.groupby(['game_key', 'team'])
          .agg(team_carries=('carries', 'sum'),
               team_targets=('targets', 'sum'))
          .reset_index()
    )
    df = df.merge(team_game_totals, on=['game_key', 'team'], how='left')
    df['carry_share'] = np.where(df['team_carries'] > 0, df['carries'] / df['team_carries'], 0.0)

    keep_cols = [
        'player_id', 'player_display_name', 'position', 'season', 'week',
        'team', 'opponent_team', 'game_key',
        'carries', 'rushing_yards', 'rushing_tds', 'carry_share',
        'targets', 'receptions', 'receiving_yards', 'receiving_tds',
        'target_share', 'air_yards_share', 'wopr', 'adot',
        'passing_yards', 'passing_tds', 'passing_interceptions', 'attempts',
        'fantasy_points_ppr',
    ]
    df = df[keep_cols].copy()

    df.to_csv(out_path, index=False)

    print(f"    Rows: {len(df):,} | Players: {df['player_id'].nunique():,} | Games: {df['game_key'].nunique():,}")
    print(f"    Saved to {out_path}")

    return df

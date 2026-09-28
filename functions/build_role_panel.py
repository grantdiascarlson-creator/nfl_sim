import pandas as pd
import numpy as np
from functions.game_level_clusters import compute_game_level_clusters


def build_role_panel(weekly_path: str, profile_path: str, out_path: str) -> pd.DataFrame:
    """
    Build the own-team / opponent role-slot panel used to compute cross-cluster correlations.

    For each game and each team, this produces a row with:
      - own_<cluster>: summed PPR points from players of that cluster on this team
      - opp_<cluster>: summed PPR points from players of that cluster on the opposing team

    Summing within a team-cluster-game slot (rather than taking one player) handles weeks
    where multiple players share a role (e.g. committee backfield).

    Args:
        weekly_path:  path to cleaned weekly skill CSV (output of process_data)
        profile_path: path to player-season profiles with cluster labels (output of assign_clusters)
        out_path:     path to write role-slot panel CSV

    Returns:
        Panel DataFrame with own/opp cluster fantasy totals per team-game
    """
    print("[3/6] Building role-slot panel...")

    weekly = pd.read_csv(weekly_path)
    profile = pd.read_csv(profile_path)

    # Attach effective_cluster and compute game-level elevations
    print("    Computing game-level cluster elevations for injured starters...")
    weekly = compute_game_level_clusters(weekly, profile)

    # Use game_cluster (not season cluster) for role-slot correlation:
    # a fill-in RB should correlate like the bellcow role they're filling,
    # not like the backup they normally are
    team_cluster_game = (
        weekly.groupby(['game_key', 'team', 'game_cluster'])['fantasy_points_ppr']
        .sum()
        .reset_index()
        .rename(columns={'game_cluster': 'cluster'})
    )

    # Pivot to wide: rows = (game_key, team), columns = cluster
    wide = team_cluster_game.pivot_table(
        index=['game_key', 'team'],
        columns='cluster',
        values='fantasy_points_ppr'
    ).reset_index()

    # Map each game to its two teams
    game_teams = (
        weekly.groupby('game_key')['team']
        .unique()
        .apply(sorted)
        .reset_index()
    )
    game_teams = game_teams[game_teams['team'].apply(len) == 2].copy()
    game_teams['team_a'] = game_teams['team'].str[0]
    game_teams['team_b'] = game_teams['team'].str[1]

    # Build own + opp prefixed views
    own = wide.add_prefix('own_').rename(columns={'own_game_key': 'game_key', 'own_team': 'team'})
    opp = wide.add_prefix('opp_').rename(columns={'opp_game_key': 'game_key', 'opp_team': 'opp_team'})

    panel = wide[['game_key', 'team']].merge(
        game_teams[['game_key', 'team_a', 'team_b']], on='game_key'
    )
    panel['opp_team'] = np.where(panel['team'] == panel['team_a'], panel['team_b'], panel['team_a'])

    panel = panel.merge(own, on=['game_key', 'team'], how='left')
    panel = panel.merge(opp.rename(columns={'opp_team': 'opp_team'}), on=['game_key', 'opp_team'], how='left')

    panel.to_csv(out_path, index=False)

    print(f"    Panel rows: {len(panel):,} | Games: {panel['game_key'].nunique():,}")
    print(f"    Saved to {out_path}")

    return panel

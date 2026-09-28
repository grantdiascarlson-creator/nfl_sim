import re
import unicodedata
import pandas as pd
import numpy as np


# ── Name alias map ────────────────────────────────────────────────────────────
# Maps weekly stats display names -> snap count file names
# for cases where normalization alone can't resolve the mismatch
# (nicknames, accented characters, abbreviated first names, etc.)
# Add new entries here as you find them in snap_unmatched.csv
NAME_ALIASES = {
    # Nickname -> full name
    'chig okonkwo':        'chigoziem okonkwo',
    'gabe davis':          'gabriel davis',
    'dee eskridge':        "d'wayne eskridge",
    'bisi johnson':        'olabisi johnson',
    'benjamin watson':     'ben watson',
    'ben victor':          'binjimen victor',
    'jalen cropper':       'jalen moreno-cropper',
    'joseph fortson':      'jody fortson',
    'josh perkins':        'joshua perkins',
    'john shenker':        'john samuel shenker',
    'mike woods':          'mike woods ii',
    'rod williams':        'rodarius williams',

    # Accented / encoding issues (handled by unicode strip, listed for documentation)
    # 'audric estim√©' -> strips to 'audric estime' which matches snap 'Audric Estime'

    # Suffix differences
    'willie snead':        'willie snead iv',
    'john franklin':       'john franklin iii',
}


def _normalize_name(name: str) -> str:
    """
    Normalize player names for joining across datasets.
    Steps:
      1. Strip accents (Estimé -> Estime)
      2. Lowercase + strip whitespace
      3. Remove suffixes (Jr./Sr./II/III/IV)
      4. Remove periods (D.J. -> dj)
      5. Collapse whitespace
      6. Apply alias map for nickname/full-name mismatches
    """
    # Strip unicode accents
    name = unicodedata.normalize('NFD', str(name))
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
    name = name.lower().strip()
    name = re.sub(r'\b(jr\.?|sr\.?|ii|iii|iv)\b', '', name)
    name = name.replace('.', '')
    name = re.sub(r'\s+', ' ', name).strip()
    # Apply alias overrides
    return NAME_ALIASES.get(name, name)


def enrich_features(
    weekly_path: str,
    snap_path: str,
    rz_targets_path: str,
    rz_carries_path: str,
    out_path: str,
) -> pd.DataFrame:
    """
    Join snap share and red zone stats onto the weekly player skill DataFrame.

    Join strategy:
      - RZ targets/carries: direct join on (player_id, season, week) — same GSIS ID format
      - Snap counts: name-normalized join on (name_norm, team, season, week) — PFR IDs differ
        Uses alias map to handle nickname/full-name mismatches (e.g. Chig -> Chigoziem)

    Missing values:
      - rz_targets / rz_carries: filled with 0 (player had no RZ opportunity that week)
      - offense_pct: left as NaN where unmatched (not fabricated)

    Outputs:
      - Enriched weekly CSV at out_path
      - Diagnostic unmatched CSV at out_path.replace('.csv', '_snap_unmatched.csv')
    """
    print("[1.5/6] Enriching weekly data with snap share and red zone stats...")

    weekly = pd.read_csv(weekly_path, low_memory=False)
    snaps  = pd.read_csv(snap_path, low_memory=False)
    rz_tgt = pd.read_csv(rz_targets_path, low_memory=False)
    rz_car = pd.read_csv(rz_carries_path, low_memory=False)

    # ── Red zone targets (GSIS join) ─────────────────────────────────────────
    rz_tgt = rz_tgt.rename(columns={'receiver_player_id': 'player_id'})
    weekly = weekly.merge(
        rz_tgt[['player_id', 'season', 'week', 'rz_targets']],
        on=['player_id', 'season', 'week'], how='left'
    )
    weekly['rz_targets'] = weekly['rz_targets'].fillna(0).astype(int)

    # ── Red zone carries (GSIS join) ─────────────────────────────────────────
    rz_car = rz_car.rename(columns={'rusher_player_id': 'player_id'})
    weekly = weekly.merge(
        rz_car[['player_id', 'season', 'week', 'rz_carries']],
        on=['player_id', 'season', 'week'], how='left'
    )
    weekly['rz_carries'] = weekly['rz_carries'].fillna(0).astype(int)

    # ── Red zone carry share per player-game ─────────────────────────────────
    # Useful for identifying goal-line backs in committee situations:
    # a RB with 20% overall carry share but 60% of his team's RZ carries is the TD-scorer
    team_rz_carries = (
        weekly.groupby(['game_key', 'team'])['rz_carries']
        .sum()
        .reset_index()
        .rename(columns={'rz_carries': 'team_rz_carries'})
    )
    weekly = weekly.merge(team_rz_carries, on=['game_key', 'team'], how='left')
    weekly['rz_carry_share'] = np.where(
        weekly['team_rz_carries'] > 0,
        weekly['rz_carries'] / weekly['team_rz_carries'],
        0.0
    )

    # ── Snap share (name-normalized join) ────────────────────────────────────
    snaps = snaps[snaps['game_type'] == 'REG'].copy()

    # Normalize team abbreviations: snap counts keep historical names, weekly stats use current
    TEAM_ABBREV_MAP = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LAR'}
    snaps['team'] = snaps['team'].replace(TEAM_ABBREV_MAP)

    snaps['name_norm']  = snaps['player'].apply(_normalize_name)
    weekly['name_norm'] = weekly['player_display_name'].apply(_normalize_name)

    # Position filter uses prefix match to capture all snap-count variants:
    # HB, RB/F, RB/W, FB/D, FB/R, FB/T, TE/D, WR/R etc.
    SKILL_PREFIXES = ('QB', 'RB', 'HB', 'WR', 'TE', 'FB')
    snap_skill = snaps[snaps['position'].str.startswith(SKILL_PREFIXES)].copy()

    snap_slim = (
        snap_skill
        [['name_norm', 'team', 'season', 'week', 'offense_snaps', 'offense_pct']]
        .drop_duplicates(subset=['name_norm', 'team', 'season', 'week'])
    )

    weekly = weekly.merge(snap_slim, on=['name_norm', 'team', 'season', 'week'], how='left')

    # ── Diagnostics ──────────────────────────────────────────────────────────
    skill_positions = ['QB', 'RB', 'WR', 'TE', 'FB']
    snap_match_rate = weekly['offense_pct'].notna().mean()
    print(f"    Snap share match rate:    {snap_match_rate:.1%}")
    print(f"    RZ targets > 0:           {(weekly['rz_targets'] > 0).sum():,} player-weeks")
    print(f"    RZ carries > 0:           {(weekly['rz_carries'] > 0).sum():,} player-weeks")

    # Write unmatched diagnostic (skill positions only)
    unmatched = (
        weekly[weekly['offense_pct'].isna() & weekly['position'].isin(skill_positions)]
        .groupby(['player_display_name', 'position', 'team', 'season'])
        .agg(unmatched_weeks=('week', 'count'))
        .reset_index()
        .sort_values(['season', 'unmatched_weeks'], ascending=[True, False])
    )
    unmatched_path = out_path.replace('.csv', '_snap_unmatched.csv')
    unmatched.to_csv(unmatched_path, index=False)
    print(f"    Unmatched player-seasons: {len(unmatched):,} → {unmatched_path}")
    if len(unmatched) > 0:
        top = unmatched.nlargest(5, 'unmatched_weeks')[['player_display_name', 'season', 'unmatched_weeks']]
        print(f"    Top unmatched:")
        for _, r in top.iterrows():
            print(f"      {r['player_display_name']} ({int(r['season'])}) — {int(r['unmatched_weeks'])} weeks")

    weekly = weekly.drop(columns=['name_norm'])
    weekly.to_csv(out_path, index=False)
    print(f"    Saved enriched data to {out_path}")

    return weekly

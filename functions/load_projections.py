"""
Load and normalise DK or FD weekly projection files.

Both files share the same structure but with platform-prefixed columns.
DK file has DST entries; FD file does not.

Normalised output columns:
  player_name   - clean player name; DST = "{TEAM}_DST"
  team          - team abbreviation
  position      - QB / RB / WR / TE / DST
  salary        - integer salary
  projected_ppr - projected fantasy points (the distribution mean)
  floor         - P10 projection
  ceiling       - P90 projection
  is_home       - bool; True if team is home this week
  opponent      - opponent team abbreviation
  small_pct     - small-field ownership projection (%)
  large_pct     - large-field ownership projection (%)
  platform      - 'dk' or 'fd'

Usage:
  from functions.load_projections import load_projections
  df = load_projections('inputs/projections/week1/dk.csv', platform='dk')
  df = load_projections('inputs/projections/week1/fd.csv', platform='fd')
"""

import re
import os
import pandas as pd


def _parse_salary(val) -> int:
    """'$8,000' → 8000"""
    try:
        return int(re.sub(r'[^\d]', '', str(val)))
    except (ValueError, TypeError):
        return 0


def _parse_pct(val) -> float:
    """'37.6%' → 37.6"""
    try:
        return float(str(val).replace('%', '').strip())
    except (ValueError, TypeError):
        return 0.0


def _home_away(team: str, opp: str):
    """
    Determine is_home and clean opponent from the Opp column.
    '@HOU' → is_home=False, opponent='HOU'  (team is away)
    'HOU'  → is_home=True,  opponent='HOU'  (team is home)
    """
    opp = str(opp).strip()
    if opp.startswith('@'):
        return False, opp[1:].strip()
    return True, opp.strip()


def load_projections(path: str, platform: str = None) -> pd.DataFrame:
    """
    Load a DK or FD projection file and return a normalised DataFrame.

    Args:
        path:     path to the CSV (dk.csv or fd.csv)
        platform: 'dk' or 'fd'. Auto-detected from filename if not provided.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Projection file not found: {path}")

    # Auto-detect platform from filename
    if platform is None:
        fname = os.path.basename(path).lower()
        if 'draftkings' in fname or fname.startswith('dk'):
            platform = 'dk'
        elif 'fanduel' in fname or fname.startswith('fd'):
            platform = 'fd'
        else:
            raise ValueError(f"Cannot auto-detect platform from {path}. Pass platform='dk' or 'fd'.")

    platform = platform.lower()
    pfx = 'DK' if platform == 'dk' else 'FD'

    raw = pd.read_csv(path)

    # Validate expected columns
    expected = ['Player', f'{pfx} Pos', 'Team', 'Opp',
                f'{pfx} Salary', f'{pfx} Proj']
    missing = [c for c in expected if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    rows = []
    for _, r in raw.iterrows():
        pos      = str(r[f'{pfx} Pos']).strip()
        team     = str(r['Team']).strip()
        raw_name = str(r['Player']).strip()

        # DST: DK uses 'DST', FD uses 'D' — both map to TEAM_DST
        if pos in ('DST', 'D'):
            pos = 'DST'
            player_name = f"{team}_DST"
        else:
            player_name = raw_name

        is_home, opponent = _home_away(team, r['Opp'])

        salary = _parse_salary(r.get(f'{pfx} Salary', 0))
        proj   = float(r.get(f'{pfx} Proj',    0) or 0)
        floor_ = float(r.get(f'{pfx} Floor',   0) or 0)
        ceil_  = float(r.get(f'{pfx} Ceiling', 0) or 0)
        sm_pct = _parse_pct(r.get('Small Field', 0))
        lg_pct = _parse_pct(r.get('Large Field', 0))

        rows.append({
            'player_name':   player_name,
            'team':          team,
            'position':      pos,
            'salary':        salary,
            'projected_ppr': round(proj, 2),
            'floor':         round(floor_, 2),
            'ceiling':       round(ceil_, 2),
            'is_home':       is_home,
            'opponent':      opponent,
            'small_pct':     sm_pct,
            'large_pct':     lg_pct,
            'platform':      platform,
        })

    df = pd.DataFrame(rows)

    # Filter out players with no projection or salary
    df = df[(df['salary'] > 0) | (df['position'] == 'DST')].copy()
    df = df[df['projected_ppr'] >= 0].copy()

    print(f"  {platform.upper()}: {len(df)} players loaded from {os.path.basename(path)}")
    pos_counts = df['position'].value_counts()
    for pos, n in pos_counts.items():
        print(f"    {pos:<5} {n}")

    return df.reset_index(drop=True)


def projections_path(week: int, platform: str, base: str = 'inputs/projections') -> str:
    """Return the canonical path for a week's projection file."""
    fname = f"{platform.lower()}.csv"
    return os.path.join(base, f"week{week}", fname)

"""
Generate current_rosters.csv from a weekly projection file.

Logic:
  1. Load projection file (all players expected to play this week)
  2. For each player, look up cluster from current_rosters_base.csv
     — this is the master file with all start-of-year cluster assignments
       and manual overrides (e.g. Lamar → QB_dual_threat_elite)
     — cluster_override takes priority over effective_cluster over cluster
  3. Fall back to player_season_profile.csv for anyone not in the base
  4. K → K_starter, DST → DST cluster from recent season average
  5. Output inputs/current_rosters.csv

Run once per week before simulate_week.py:
  python scripts/generate_rosters.py --week 1 --platform dk
  python scripts/generate_rosters.py --week 1 --platform fd
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from functions.load_projections import load_projections, projections_path

BASE_ROSTERS_PATH = 'inputs/current_rosters_base.csv'
PROFILE_PATH      = 'outputs/player_season_profile.csv'
DST_PATH          = 'outputs/dst_weekly.csv'
ROSTERS_OUT       = 'inputs/current_rosters.csv'


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]", '', str(name).lower().strip())


def get_dst_clusters(dst_path: str) -> dict:
    if not os.path.exists(dst_path):
        return {}
    dst = pd.read_csv(dst_path)
    latest = dst['season'].max()
    avg    = dst[dst['season'] == latest].groupby('team')['fantasy_points'].mean()
    return {team: ('DST_strong' if v > 7.5 else 'DST_average' if v >= 5.0 else 'DST_weak')
            for team, v in avg.items()}


def generate_rosters(proj_path: str,
                     base_path:    str = BASE_ROSTERS_PATH,
                     profile_path: str = PROFILE_PATH,
                     dst_path:     str = DST_PATH,
                     out_path:     str = ROSTERS_OUT) -> pd.DataFrame:

    print(f"\nGenerating rosters from {proj_path}...")

    # ── Load projection file ───────────────────────────────────────────────
    platform = 'dk' if 'dk' in os.path.basename(proj_path).lower() else 'fd'
    proj = load_projections(proj_path, platform=platform)
    print(f"  Projection players: {len(proj)}")

    # ── Load base roster (primary cluster source) ──────────────────────────
    base = pd.read_csv(base_path, dtype=str)
    base['player_name']      = base['player_name'].str.strip()
    base['cluster_override'] = base['cluster_override'].fillna('').str.strip()
    base['effective_cluster']= base['effective_cluster'].fillna('').str.strip()
    base['cluster']          = base['cluster'].fillna('').str.strip()
    base['role_tag']         = base.get('role_tag', pd.Series('', index=base.index)).fillna('')
    base['name_norm']        = base['player_name'].apply(normalize)
    base_lookup              = base.set_index('name_norm').to_dict('index')
    print(f"  Base roster players: {len(base)}")

    # ── Load profile for fallback ──────────────────────────────────────────
    profile_lookup = {}
    if os.path.exists(profile_path):
        profile = pd.read_csv(profile_path, dtype=str)
        profile['player_display_name'] = profile['player_display_name'].str.strip()
        recent = (profile.sort_values('season', ascending=False)
                  .drop_duplicates('player_display_name'))
        for _, r in recent.iterrows():
            key = normalize(r['player_display_name'])
            profile_lookup[key] = {
                'cluster':          str(r.get('cluster','') or ''),
                'effective_cluster':str(r.get('effective_cluster','') or ''),
                'role_tag':         str(r.get('role_tag','') or ''),
            }

    dst_clusters = get_dst_clusters(dst_path)

    # ── Build output rows ──────────────────────────────────────────────────
    # Players not found in base get depth clusters if projected < 4.5 PPR,
    # starter-level fallback if >= 4.5 (likely a meaningful contributor)
    LOW_PROJ_THRESHOLD  = 4.5
    FALLBACK_DEPTH      = {'QB':'QB_backup', 'RB':'RB_backup',
                           'WR':'WR_depth',  'TE':'TE_depth'}
    FALLBACK_STARTER    = {'QB':'QB_pocket', 'RB':'RB_committee_featured',
                           'WR':'WR_WR2_s0', 'TE':'TE_TE1_s1'}

    rows      = []
    no_base   = []
    no_profile= []

    for _, player in proj.iterrows():
        name  = player['player_name']
        team  = player['team']
        pos   = player['position']
        proj_ = float(player.get('projected_ppr', 0) or 0)

        if pos == 'DST':
            cluster = dst_clusters.get(team, 'DST_average')
            rows.append({'player_name': name, 'team': team, 'position': 'DST',
                         'cluster': cluster, 'effective_cluster': cluster,
                         'cluster_override': '', 'role_tag': '',
                         'projected_ppr': proj_})
            continue

        if pos == 'K':
            rows.append({'player_name': name, 'team': team, 'position': 'K',
                         'cluster': 'K_starter', 'effective_cluster': 'K_starter',
                         'cluster_override': '', 'role_tag': '',
                         'projected_ppr': proj_})
            continue

        # ── Skill positions: look up in base first ─────────────────────────
        norm   = normalize(name)
        brow   = base_lookup.get(norm)

        if brow is None:
            # Try first-two-words match for suffix differences (Jr., III, etc.)
            first2 = ' '.join(norm.split()[:2])
            matches = {k: v for k, v in base_lookup.items()
                       if k.startswith(first2)}
            if len(matches) == 1:
                brow = list(matches.values())[0]
            elif len(matches) > 1:
                team_match = {k: v for k, v in matches.items()
                              if str(v.get('team','')).strip() == team}
                brow = list((team_match or matches).values())[0]

        if brow is None:
            # Last-name + same team fallback — handles nickname vs full name
            # (Kenny Gainwell → Kenneth Gainwell, CJ → C.J., etc.)
            # Team must match — update current_rosters_base.csv if a player changes teams
            last = norm.split()[-1] if norm.split() else ''
            if last:
                same_team = {k: v for k, v in base_lookup.items()
                             if k.split()[-1] == last
                             and str(v.get('team','')).strip() == team}
                if len(same_team) == 1:
                    brow = list(same_team.values())[0]

        if brow:
            override  = str(brow.get('cluster_override','') or '').strip()
            eff       = str(brow.get('effective_cluster','') or '').strip()
            base_clus = str(brow.get('cluster','') or '').strip()
            role_tag  = str(brow.get('role_tag','') or '').strip()
        else:
            no_base.append(name)
            # Fall back to historical profile
            prow = profile_lookup.get(norm)
            if prow:
                override  = ''
                eff       = prow['effective_cluster']
                base_clus = prow['cluster']
                role_tag  = prow['role_tag']
            else:
                no_profile.append(name)
                override  = ''
                fb = FALLBACK_DEPTH if proj_ < LOW_PROJ_THRESHOLD else FALLBACK_STARTER
                eff       = fb.get(pos, 'WR_depth')
                base_clus = fb.get(pos, 'WR_depth')
                role_tag  = ''

        rows.append({
            'player_name':       name,
            'team':              team,
            'position':          pos,
            'cluster':           base_clus,
            'effective_cluster': eff,
            'cluster_override':  override,
            'role_tag':          role_tag,
            'projected_ppr':     proj_,
        })

    roster_df = pd.DataFrame(rows)
    roster_df.to_csv(out_path, index=False)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"ROSTER: {len(roster_df)} players → {out_path}")
    print(f"{'='*55}")
    for pos, grp in roster_df.groupby('position'):
        n_ov = (grp['cluster_override'].str.strip() != '').sum()
        print(f"  {pos:<5} {len(grp):>4}  ({n_ov} overrides)")

    if no_base:
        print(f"\nNot in base — used profile/fallback ({len(no_base)}):")
        for p in no_base[:15]:
            r = roster_df[roster_df['player_name']==p].iloc[0]
            print(f"  {p:<30} → {r['effective_cluster'] or r['cluster']}")
        if len(no_base) > 15:
            print(f"  ... and {len(no_base)-15} more")

    if no_profile:
        print(f"\nNo profile found — used position fallback ({len(no_profile)}):")
        for p in no_profile[:10]:
            print(f"  {p}")

    print(f"\nTop 20 by projection:")
    print(f"{'Player':<28}{'Pos':<5}{'Team':<5}{'Cluster':<32}{'Override':<22}{'Proj':>6}")
    print("-"*100)
    for _, r in roster_df.sort_values('projected_ppr', ascending=False).head(20).iterrows():
        ov = r['cluster_override'] or ''
        cl = r['cluster_override'] if ov else r['effective_cluster'] or r['cluster']
        print(f"  {r['player_name']:<26}{r['position']:<5}{r['team']:<5}{cl:<32}{ov:<22}{r['projected_ppr']:>6.1f}")

    return roster_df


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Build current_rosters.csv from weekly projections')
    p.add_argument('--week',     type=int, default=1)
    p.add_argument('--platform', default='dk', choices=['dk','fd'])
    p.add_argument('--projections', default=None,
                   help='Override projection file path directly')
    p.add_argument('--base',     default=BASE_ROSTERS_PATH)
    p.add_argument('--profile',  default=PROFILE_PATH)
    p.add_argument('--dst',      default=DST_PATH)
    p.add_argument('--out',      default=ROSTERS_OUT)
    a = p.parse_args()
    proj_path = a.projections or f'inputs/projections/week{a.week}/{a.platform}.csv'
    generate_rosters(proj_path, a.base, a.profile, a.dst, a.out)

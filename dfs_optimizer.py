import os
"""
DFS Lineup Optimizer

Two functions:

1. optimal_lineups(sims_path, salaries_path)
   For each simulation, solve the salary-capped ILP to find the optimal lineup.
   Output: which players are in the optimal lineup for each sim, and what the
   optimal score was. Shows which players appear most often in optimal lineups.

2. generate_field_lineups(projections, salaries, n_lineups, n_sims)
   Generate N realistic "field" lineups based on noisy projections + stacking.
   Evaluate each lineup against all simulations. Find which lineup wins most
   often and which has the best average score.

Lineup format (DraftKings-style):
  QB(1) + RB(2) + WR(3) + TE(1) + FLEX(RB/WR/TE)(1) + DST(1) = 9 players
  Salary cap: $50,000

Usage:
  python dfs_optimizer.py --sims outputs/week_1_sims.csv
                          --salaries inputs/week_salaries.csv
                          --mode optimal
  python dfs_optimizer.py --sims outputs/week_1_sims.csv
                          --salaries inputs/week_salaries.csv
                          --mode field --n-lineups 2000
"""

import argparse
import numpy as np
import pandas as pd
from pulp import (LpProblem, LpVariable, LpMaximize, lpSum,
                  LpBinary, PULP_CBC_CMD, value)


# ── DFS constants ──────────────────────────────────────────────────────────
SALARY_CAP = 50_000
# DraftKings main slate: QB + 2RB + 3WR + TE + FLEX(RB/WR/TE) + DST = 9 players
# K is NOT in the main slate lineup
ROSTER = {
    'QB':  {'min': 1, 'max': 1},
    'RB':  {'min': 2, 'max': 3},   # 2 starters + possibly 1 in FLEX
    'WR':  {'min': 3, 'max': 4},   # 3 starters + possibly 1 in FLEX
    'TE':  {'min': 1, 'max': 2},   # 1 starter  + possibly 1 in FLEX
    'DST': {'min': 1, 'max': 1},
}
TOTAL_PLAYERS = 9


def solve_lineup(ppr_dict: dict, salary_dict: dict, pos_dict: dict,
                 team_dict: dict = None,
                 locked: list = None,
                 excluded: list = None,
                 stacking: bool = False,
                 salary_cap: int = SALARY_CAP) -> tuple:
    """
    Solve the DFS lineup ILP.

    Args:
        ppr_dict:   {player_name: projected_ppr}
        salary_dict:{player_name: salary}
        pos_dict:   {player_name: position}
        team_dict:  {player_name: team}  — required for stacking
        locked:     players forced into the lineup
        excluded:   players excluded from the lineup
        stacking:   enforce QB+teammate stacking (at least 1 WR/TE same team as QB)
        salary_cap: default 50000

    Returns:
        (selected_players: list, total_ppr: float, total_salary: int)
        Returns (None, 0, 0) if infeasible.
    """
    # K is not in the main slate lineup format (QB/RB/WR/TE/FLEX/DST)
    DFS_POSITIONS = {'QB', 'RB', 'WR', 'TE', 'DST'}
    players = [p for p in ppr_dict
               if p in salary_dict and p in pos_dict
               and pos_dict[p] in DFS_POSITIONS]
    locked   = set(locked or [])
    excluded = set(excluded or [])

    if not players:
        return None, 0, 0

    prob = LpProblem("DFS_lineup", LpMaximize)
    x = {p: LpVariable(f"x_{i}", cat=LpBinary) for i, p in enumerate(players)}

    # Objective
    prob += lpSum(ppr_dict[p] * x[p] for p in players)

    # Total players
    prob += lpSum(x[p] for p in players) == TOTAL_PLAYERS

    # Salary cap
    prob += lpSum(salary_dict[p] * x[p] for p in players) <= salary_cap

    # Positional constraints
    for pos, limits in ROSTER.items():
        pos_players = [p for p in players if pos_dict[p] == pos]
        if pos_players:
            prob += lpSum(x[p] for p in pos_players) >= limits['min']
            prob += lpSum(x[p] for p in pos_players) <= limits['max']

    # Locked/excluded
    for p in locked:
        if p in x:
            prob += x[p] == 1
    for p in excluded:
        if p in x:
            prob += x[p] == 0

    # Stacking: if QB from team T is selected, at least 1 WR/TE from team T
    if stacking and team_dict:
        teams = set(team_dict.values())
        for team in teams:
            qbs  = [p for p in players if pos_dict[p] == 'QB'  and team_dict.get(p) == team]
            pass_ = [p for p in players if pos_dict[p] in ('WR','TE') and team_dict.get(p) == team]
            if qbs and pass_:
                # For each QB on this team selected, at least 1 pass catcher from same team
                prob += lpSum(x[p] for p in pass_) >= lpSum(x[p] for p in qbs)

    prob.solve(PULP_CBC_CMD(msg=0))

    if prob.status != 1:   # infeasible
        return None, 0, 0

    selected = [p for p in players if value(x[p]) > 0.5]
    total_ppr = sum(ppr_dict[p] for p in selected)
    total_sal = sum(salary_dict[p] for p in selected)
    return selected, total_ppr, total_sal


# ── Optimal lineup per simulation ─────────────────────────────────────────

def optimal_lineups(sims_path: str, salaries_path: str,
                    output_path: str = None) -> pd.DataFrame:
    """
    For each simulation, find the salary-capped optimal lineup.
    """
    print(f"\nLoading simulations from {sims_path}...")
    sims = pd.read_csv(sims_path)

    # Salary, position and team come from the sims file (populated from projection file)
    # salaries_path is optional override (e.g. to add missing players)
    player_info = sims.groupby('player').first().reset_index()
    salary_dict = dict(zip(player_info['player'], player_info['salary'])) \
                  if 'salary' in sims.columns else {}
    pos_dict    = dict(zip(player_info['player'], player_info['position']))
    team_dict   = dict(zip(player_info['player'], player_info['team']))

    # Optional salary override file
    if salaries_path and os.path.exists(salaries_path):
        sal = pd.read_csv(salaries_path)
        sal['player_name'] = sal['player_name'].str.strip()
        salary_dict.update(dict(zip(sal['player_name'], sal['salary'])))
        pos_dict.update(dict(zip(sal['player_name'], sal['position'])))

    n_sims   = sims['sim_no'].nunique()
    n_players = sims['player'].nunique()
    print(f"Simulations: {n_sims:,}  |  Players: {n_players}")

    results = []
    for sim_no, sim_grp in sims.groupby('sim_no'):
        ppr_dict = dict(zip(sim_grp['player'], sim_grp['ppr']))

        selected, total_ppr, total_sal = solve_lineup(
            ppr_dict, salary_dict, pos_dict, team_dict
        )
        if selected is None:
            continue

        # Build one row per lineup: sim_no, QB, RB1, RB2, WR1, WR2, WR3, TE, FLEX, DST
        # Assign players to positional slots
        by_pos = {}
        for p in selected:
            pos = pos_dict.get(p, '')
            by_pos.setdefault(pos, []).append(p)

        # Determine FLEX: the position with one extra player
        qb  = by_pos.get('QB', [None])[0]
        dst = by_pos.get('DST', [None])[0]
        rbs = by_pos.get('RB', [])
        wrs = by_pos.get('WR', [])
        tes = by_pos.get('TE', [])

        flex = None
        if len(rbs) == 3:
            flex = rbs.pop()
        elif len(wrs) == 4:
            flex = wrs.pop()
        elif len(tes) == 2:
            flex = tes.pop()

        # Sort by PPR desc within each position
        rbs = sorted(rbs, key=lambda p: ppr_dict.get(p, 0), reverse=True)
        wrs = sorted(wrs, key=lambda p: ppr_dict.get(p, 0), reverse=True)

        row = {
            'sim_no':        sim_no,
            'QB':            qb,
            'RB1':           rbs[0] if len(rbs) > 0 else None,
            'RB2':           rbs[1] if len(rbs) > 1 else None,
            'WR1':           wrs[0] if len(wrs) > 0 else None,
            'WR2':           wrs[1] if len(wrs) > 1 else None,
            'WR3':           wrs[2] if len(wrs) > 2 else None,
            'TE':            tes[0] if len(tes) > 0 else None,
            'FLEX':          flex,
            'DST':           dst,
            'total_ppr':     round(total_ppr, 2),
            'total_salary':  total_sal,
        }
        results.append(row)

        if sim_no % 500 == 0:
            print(f"  Solved {sim_no:,}/{n_sims:,} sims...")

    results_df = pd.DataFrame(results)
    if output_path:
        results_df.to_csv(output_path, index=False)

    # Summary: player appearance rate across all optimal lineups
    lineup_cols = ['QB','RB1','RB2','WR1','WR2','WR3','TE','FLEX','DST']
    player_counts = {}
    for col in lineup_cols:
        for name in results_df[col].dropna():
            player_counts[name] = player_counts.get(name, 0) + 1

    summary_rows = []
    for player, count in sorted(player_counts.items(), key=lambda x: -x[1]):
        summary_rows.append({
            'player':         player,
            'position':       pos_dict.get(player, ''),
            'team':           team_dict.get(player, ''),
            'appearance_pct': round(count / n_sims * 100, 1),
            'appearances':    count,
        })
    summary = pd.DataFrame(summary_rows)

    print(f"\n{'='*65}")
    print(f"OPTIMAL LINEUP ANALYSIS — {n_sims:,} simulations")
    print(f"{'='*65}")
    print(f"\n{'Player':<26}{'Pos':<5}{'Team':<5}{'% in optimal':>13}")
    print("-"*52)
    for _, r in summary.head(20).iterrows():
        print(f"  {r['player']:<24}{r['position']:<5}{r['team']:<5}"
              f"{r['appearance_pct']:>12.1f}%")

    print(f"\nSample lineups (first 5):")
    print(results_df.head(5)[['sim_no','QB','RB1','RB2','WR1','WR2','WR3',
                               'TE','FLEX','DST','total_ppr']].to_string(index=False))

    return results_df, summary


# ── Field lineup generator ─────────────────────────────────────────────────

def generate_field_lineups(
    sims_path: str,
    salaries_path: str,
    n_lineups: int = 2000,
    noise_sigma: float = 0.35,
    enforce_stacking: bool = True,
    output_path: str = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate N realistic field lineups, evaluate against all simulations,
    and rank by performance.

    Args:
        sims_path:        path to week_X_sims.csv
        salaries_path:    path to salary file
        n_lineups:        number of field lineups to generate
        noise_sigma:      std of multiplicative noise on projections
                          (0.35 = 35% noise, creates diverse lineups)
        enforce_stacking: QB must have at least 1 WR/TE from same team
        output_path:      where to save results

    Output columns:
        lineup_id, player (repeated 9x per lineup), position, team, salary,
        mean_score, p10_score, p90_score, win_rate (% of sims this lineup wins)
    """
    print(f"\nLoading data...")
    sims = pd.read_csv(sims_path)
    player_info = sims.groupby('player').first().reset_index()
    salary_dict = dict(zip(player_info['player'], player_info['salary'])) \
                  if 'salary' in sims.columns else {}
    pos_dict    = dict(zip(player_info['player'], player_info['position']))
    team_dict   = dict(zip(player_info['player'], player_info['team']))
    if salaries_path and os.path.exists(salaries_path):
        sal = pd.read_csv(salaries_path)
        sal['player_name'] = sal['player_name'].str.strip()
        salary_dict.update(dict(zip(sal['player_name'], sal['salary'])))

    # Mean projection from simulations (use as baseline, then add noise)
    mean_proj = sims.groupby('player')['ppr'].mean().to_dict()

    # Only keep players who have salary and sim data
    eligible = [p for p in mean_proj if p in salary_dict and p in pos_dict]
    print(f"Eligible players: {len(eligible)}  |  Generating {n_lineups:,} field lineups...")

    rng = np.random.default_rng(seed)
    lineups = []   # list of lists of player names

    attempts = 0
    while len(lineups) < n_lineups and attempts < n_lineups * 5:
        attempts += 1

        # Add multiplicative noise to projections
        noisy_proj = {
            p: max(0.1, mean_proj[p] * rng.lognormal(0, noise_sigma))
            for p in eligible
        }

        # Randomly exclude a few players to create diversity (simulate fades)
        n_fades = rng.integers(0, min(8, len(eligible) // 4))
        faded   = list(rng.choice(eligible, size=n_fades, replace=False))

        selected, _, _ = solve_lineup(
            noisy_proj, salary_dict, pos_dict, team_dict,
            excluded=faded,
            stacking=enforce_stacking,
        )
        if selected and len(selected) == TOTAL_PLAYERS:
            lineups.append(tuple(sorted(selected)))

    # Deduplicate
    lineups = list(set(lineups))
    print(f"  Generated {len(lineups):,} unique lineups ({attempts} attempts)")

    # ── Evaluate each lineup against all simulations ───────────────────────
    print(f"Evaluating {len(lineups):,} lineups against {sims['sim_no'].nunique():,} sims...")

    # Build PPR matrix: (n_sims × n_players)
    sim_pivot = sims.pivot(index='sim_no', columns='player', values='ppr').fillna(0)
    n_sims    = len(sim_pivot)

    lineup_scores = np.zeros((len(lineups), n_sims))
    for li, lineup in enumerate(lineups):
        cols = [p for p in lineup if p in sim_pivot.columns]
        lineup_scores[li] = sim_pivot[cols].sum(axis=1).values

    # Win rate: which lineup scores highest in each sim
    winners = lineup_scores.argmax(axis=0)   # (n_sims,) — index of winning lineup per sim
    win_counts = np.bincount(winners, minlength=len(lineups))

    # Build results
    def _assign_slots(lineup):
        by_pos = {}
        for p in lineup:
            pos = pos_dict.get(p, '')
            by_pos.setdefault(pos, []).append(p)
        rbs = sorted(by_pos.get('RB', []), key=lambda p: mean_proj.get(p, 0), reverse=True)
        wrs = sorted(by_pos.get('WR', []), key=lambda p: mean_proj.get(p, 0), reverse=True)
        tes = by_pos.get('TE', [])
        flex = None
        if len(rbs) == 3: flex = rbs.pop()
        elif len(wrs) == 4: flex = wrs.pop()
        elif len(tes) == 2: flex = tes.pop()
        return {
            'QB':  by_pos.get('QB', [None])[0],
            'RB1': rbs[0] if len(rbs) > 0 else None,
            'RB2': rbs[1] if len(rbs) > 1 else None,
            'WR1': wrs[0] if len(wrs) > 0 else None,
            'WR2': wrs[1] if len(wrs) > 1 else None,
            'WR3': wrs[2] if len(wrs) > 2 else None,
            'TE':  tes[0] if tes else None,
            'FLEX': flex,
            'DST': by_pos.get('DST', [None])[0],
        }

    rows = []
    for li, lineup in enumerate(lineups):
        scores = lineup_scores[li]
        slots  = _assign_slots(lineup)
        rows.append({
            'lineup_id':    li + 1,
            **slots,
            'salary':       sum(salary_dict.get(p, 0) for p in lineup),
            'mean_score':   round(scores.mean(), 2),
            'p10_score':    round(np.percentile(scores, 10), 2),
            'p50_score':    round(np.percentile(scores, 50), 2),
            'p90_score':    round(np.percentile(scores, 90), 2),
            'win_count':    int(win_counts[li]),
            'win_rate_pct': round(win_counts[li] / n_sims * 100, 2),
        })

    results_df = pd.DataFrame(rows)
    if output_path:
        results_df.to_csv(output_path, index=False)

    # Summary: top lineups by mean score and by win rate
    SLOT_COLS = ['QB','RB1','RB2','WR1','WR2','WR3','TE','FLEX','DST']
    results_df['players'] = results_df[SLOT_COLS].apply(
        lambda r: ', '.join(str(p) for p in r if p), axis=1)
    lineup_summary = results_df[['lineup_id','mean_score','p10_score','p50_score',
                                  'p90_score','win_rate_pct','salary','players']].copy()
    lineup_summary = lineup_summary.rename(columns={'salary':'total_salary'})

    print(f"\n{'='*70}")
    print(f"FIELD LINEUP EVALUATION — {len(lineups):,} lineups vs {n_sims:,} sims")
    print(f"{'='*70}")

    print(f"\nTOP 10 BY MEAN SCORE:")
    print(f"{'LID':>4} {'Mean':>7} {'P10':>7} {'P50':>7} {'P90':>7} {'Win%':>6}  Players")
    print("-"*80)
    for _, r in lineup_summary.nlargest(10, 'mean_score').iterrows():
        print(f"{int(r['lineup_id']):>4} {r['mean_score']:>7.1f} {r['p10_score']:>7.1f} "
              f"{r['p50_score']:>7.1f} {r['p90_score']:>7.1f} {r['win_rate_pct']:>5.1f}%  "
              f"{r['players'][:60]}")

    print(f"\nTOP 10 BY WIN RATE:")
    print(f"{'LID':>4} {'Mean':>7} {'P10':>7} {'P50':>7} {'P90':>7} {'Win%':>6}  Players")
    print("-"*80)
    for _, r in lineup_summary.nlargest(10, 'win_rate_pct').iterrows():
        print(f"{int(r['lineup_id']):>4} {r['mean_score']:>7.1f} {r['p10_score']:>7.1f} "
              f"{r['p50_score']:>7.1f} {r['p90_score']:>7.1f} {r['win_rate_pct']:>5.1f}%  "
              f"{r['players'][:60]}")

    return results_df, lineup_summary


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sims',      default='outputs/week_1/dk_sims.csv')
    p.add_argument('--salaries',  default=None,
                        help='Optional salary override CSV. Salary is taken from sims file by default.')
    p.add_argument('--mode',      choices=['optimal','field'], default='optimal')
    p.add_argument('--n-lineups', type=int, default=2000)
    p.add_argument('--noise',     type=float, default=0.35)
    p.add_argument('--no-stack',  action='store_true')
    p.add_argument('--output',    default=None)
    a = p.parse_args()

    week_dir = os.path.dirname(a.sims)
    if a.mode == 'optimal':
        out = a.output or os.path.join(week_dir, os.path.basename(a.sims).replace('_sims.csv', '_optimal_lineups.csv'))
        optimal_lineups(a.sims, a.salaries, output_path=out)
    else:
        out = a.output or os.path.join(week_dir, os.path.basename(a.sims).replace('_sims.csv', '_field_lineups.csv'))
        generate_field_lineups(
            a.sims, a.salaries,
            n_lineups=a.n_lineups,
            noise_sigma=a.noise,
            enforce_stacking=not a.no_stack,
            output_path=out,
        )

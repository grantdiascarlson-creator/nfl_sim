# NFL Weekly Simulator

A Monte Carlo engine that simulates a full NFL week of fantasy scoring, with realistic correlation between players, and uses the simulations to build and evaluate DraftKings and FanDuel lineups.

This simulator models correlations bewteen players in the same game from seven seasons of historical data and draws thousands of joint outcomes for every game on the slate.

**Stack:** Python, NumPy, SciPy, pandas, PuLP (CBC solver)

## How it works

### 1. Group players by role

Players with similar roles produce similar scoring patterns. A pass-catching committee back behaves differently from a bellcow runner, even at the same projection. Each player-season from 2019–2025 is assigned a role cluster:

- **Rule-based clusters** use usage thresholds: snap share, target share, carry share, and red-zone usage.
- **K-means subclusters** split the larger groups further, using per-game volume (targets and carries per game) so a high-volume WR3 separates from a rotational one.

The result is about 50 clusters across QB, RB, WR, TE, DST and K. Examples include `QB_dual_threat_elite`, `RB_committee_receiving`, `WR_alpha_possession` and `TE_TE2_rotational`. Current-season assignments live in `inputs/current_rosters_base.csv` and can be overridden by hand when a role changes.

### 2. Estimate correlations between roles

For every game, each player is mapped to a role slot, either on his own team or on the opponent's (for example `own_WR_alpha_possession` or `opp_QB_pocket`). Pairwise correlations between slots give a 106 × 106 matrix.

Some examples from the fitted matrix:

| Pair | r | Interpretation |
|---|---|---|
| Pocket QB ↔ own WR1 (possession) | 0.30 | Tight passing link |
| Pocket QB ↔ own featured WR2 | 0.25 | Moderate coupling |
| Elite dual-threat QB ↔ own WR | ~0.15 | Rushing production is independent |
| Same-team RB ↔ WR | ~0.09 | Weak, driven by game script |
| DST ↔ opposing QB | negative | Sacks and turnovers against scoring |

Sparse pairs (fewer than 20 shared games) are set to zero, and the matrix is projected to the nearest positive semi-definite matrix (Higham, 2002). This brings the condition number down from about 5 million to about 600, so the Cholesky decomposition is stable.

Recent seasons are weighted more heavily (2019 = 1×, rising to 2025 = 7×) to reflect changes in how offenses are operating in 2026.

### 3. Fit each player's distribution

Fantasy scores are right-skewed: most games land near the median, with a long tail of big games. Each cluster gets a **skew-normal** shape. Depth players use a **hurdle model**, which adds an explicit probability of a near-zero game, since a backup often gets no meaningful touches.

Each player's distribution is then fitted to his own weekly projection and ceiling, while keeping the cluster's skew. In plain terms, the cluster decides the *shape* of the distribution and the projection file decides *where it sits and how wide it is*:

```
δ         = a / √(1 + a²)
mean_unit = δ · √(2/π)                    # mean of skewnorm(a, 0, 1)
z_ceil    = skewnorm.ppf(0.90, a, 0, 1)

scale = (ceiling − proj) / (z_ceil − mean_unit)
loc   = proj − scale · mean_unit
```

So a player projected for 21.9 with a ceiling of 37.3 has a simulated mean of about 21.9 and a 90th percentile of about 37.3.

### 4. Simulate each game with a Gaussian copula

For every game on the slate:

1. Pull the correlation submatrix for the players involved.
2. Draw correlated standard normals using its Cholesky factor.
3. Convert them to uniforms with the normal CDF.
4. Feed each uniform through the player's own distribution (inverse CDF) to get a fantasy score.

This keeps every player's individual distribution exactly as fitted while reproducing the historical correlation structure between them.

### 5. Keep QBs consistent with their receivers

A copula alone can produce impossible games, like a quarterback scoring 16 while his top receiver scores 45. Passing yards have to go somewhere, so the QB's score is kept inside a window around his projection, scaled by how his receivers did:

```
rec_scale = clip(total_receiver_score / total_receiver_projection, 0.15, 3.0)
center    = qb_projection × rec_scale
QB score  ∈ [max(0, center − buffer), center + buffer]
```

The buffer depends on how much of the QB's production comes from running: ±5 for pocket passers, ±8 for rushing QBs, and ±12 for elite dual-threat QBs. When receivers hit their projections, the window is centered on the QB's projection, so the constraint doesn't shift his mean. It only binds in the tails.

### 6. Condition on Vegas lines

Game-level scaling uses sensitivity coefficients for the game total and spread (`inputs/vegas_sensitivity.json`), so a high-total shootout produces more fantasy points across the board than a low-total game.

## Lineup optimization

`dfs_optimizer.py` works from the simulation output in two modes:

- **Optimal:** solves an integer program for each simulation to find the highest-scoring lineup under the salary cap, then reports how often each player appears in optimal lineups.
- **Field:** generates a set of realistic contest lineups (projection noise, QB stacking, random fades), scores every lineup against every simulation, and ranks them by win rate and score distribution.

DraftKings lineup: QB, 2 RB, 3 WR, TE, FLEX (RB/WR/TE), DST. $50,000 salary cap.

## Validation

Checks from a 10,000-simulation run of a Week 1 DraftKings slate:

| Player | Metric | Target | Simulated |
|---|---|---|---|
| Ja'Marr Chase | 90th percentile | 37.3 (ceiling) | 37.1 |
| Jahmyr Gibbs | 90th percentile | 40.6 (ceiling) | 40.4 |
| Josh Allen | Mean | 19.5 (projection) | 19.6 |
| Aaron Rodgers | Mean | 15.2 (projection) | 14.9 |
| Rodgers, when no receiver tops 10 | Max | < 20 | 16.6 |
| Burrow, when receivers are in top 10% | Average | > 28 | 30.1 |
| Ben Skowronek | 99th percentile | < 5 | 3.4 |

`utils/validate_calibration.py` runs probability-integral-transform (PIT) and quantile calibration tests against historical actuals. `utils/validate_pipeline.py` compares simulated correlations and distributions with historical ones.

## Project structure

```
nfl_sim/
├── simulate_week.py          Run the weekly simulation
├── dfs_optimizer.py          Optimal and field lineup modes
├── main.py                   Rebuild clusters, correlations and distributions from historical data
├── functions/                Pipeline modules (prep, clustering, correlation, distributions, simulator)
├── scripts/
│   └── generate_rosters.py   Assign clusters to this week's players
├── utils/                    Validation and roster archive scripts
├── inputs/                   Rosters, red-zone usage, Vegas sensitivity
└── examples/
    └── sample_dk_sims.csv    Sample simulation output
```

## Running it

### Setup

```bash
pip install numpy scipy pandas pulp
```

The historical player stats and snap counts (2019–2025) come from [nflverse](https://github.com/nflverse) and aren't included because of their size. Download the weekly player stats and snap counts, for example with `nflreadpy`, and save them as:

```
inputs/player_stats_2019_2025.csv
inputs/snap_counts_2019_2025.csv
```

Then build the seasonal model outputs (clusters, correlation matrix, fitted distributions):

```bash
python main.py
```

This only needs to run again when new historical data is added.

### Weekly workflow

Weekly projections aren't included. Add your own CSV with player, position, team, opponent, salary, projection, floor and ceiling columns at `inputs/projections/week{N}/dk.csv` (or `fd.csv`).

```bash
python scripts/generate_rosters.py --week 1 --platform dk
python simulate_week.py --week 1 --platform dk --sims 10000
python dfs_optimizer.py --mode optimal --sims outputs/week_1/dk_sims.csv
python dfs_optimizer.py --mode field --sims outputs/week_1/dk_sims.csv --n-lineups 2000
```

10,000 simulations of a full slate take about 30 seconds. See `examples/sample_dk_sims.csv` for the output format: one row per simulation × player, with `sim_no, player, team, position, salary, ppr`.

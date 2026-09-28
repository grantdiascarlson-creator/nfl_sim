"""
Scenario engine: injury returns (with cascade elevation) and role competition.

INJURY RETURNS (inputs/injury_returns.csv)
  Columns: player_name, team, position,
           elevated_player, elevated_cluster,   ← who fills in when this player is out
           w1..w17, notes

  Weekly probabilities: p=0 = out, p=1 = healthy, p=0.4 = 40% chance active.

  Cascade: when the injured player is inactive in a simulation, the elevated_player
  is re-sampled from elevated_cluster for that sim. Same Bernoulli draw links them.

  Season permanence: probabilities encode the full timeline. For a season-ending
  injury, set all remaining weeks to 0. For a mid-season injury with return, the
  weekly p-values encode the expected availability. At season-start, draw a single
  structured "injury state" per simulation (see draw_season_injury_states).

ROOKIE ASCENSION (inputs/rookie_ascension.csv)
  Columns: rookie_name, veteran_name (blank=no incumbent), team,
           rookie_base_cluster, rookie_ascended_cluster,
           veteran_base_cluster, veteran_demoted_cluster,
           w1..w17, notes

  veteran_name can be blank — rookie just develops, no one gets demoted.
  Weekly probabilities are CUMULATIVE (p_w = P(ascension has happened by week w)).

  Season permanence: draw a single ascension week per simulation.
  Once ascended, stays ascended for the rest of the season.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats


INJURY_PATH    = 'inputs/injury_returns.csv'
ASCENSION_PATH = 'inputs/rookie_ascension.csv'


def _get_week_prob(row: pd.Series, week: int) -> float:
    val = row.get(f'w{week}')
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 1.0
    return float(val)


def _sample_cluster(cluster: str, dist_df: pd.DataFrame,
                    n: int, rng: np.random.Generator) -> np.ndarray:
    if not cluster or cluster not in dist_df.index:
        return np.zeros(n)
    row    = dist_df.loc[cluster]
    a      = float(row['skew_a'])
    loc    = float(row['skew_loc'])
    scale  = float(row['skew_scale'])
    model  = str(row.get('model', 'skewnorm'))
    p_zero = float(row['p_zero']) if not pd.isna(row.get('p_zero', float('nan'))) else 0.0
    u      = rng.uniform(0, 1, n)
    if model == 'hurdle':
        vals = np.where(u <= p_zero, 0.0,
                        stats.skewnorm.ppf((u - p_zero) / (1 - p_zero + 1e-12), a, loc, scale))
    else:
        vals = stats.skewnorm.ppf(u, a, loc, scale)
    return np.clip(vals, 0, None)


def _transition_probs(cumulative: list) -> list:
    """Cumulative → transition probabilities for season draw."""
    prev, trans = 0.0, []
    for p in cumulative:
        p = min(float(p), 1.0)
        trans.append(max(0.0, p - prev))
        prev = p
    trans.append(max(0.0, 1.0 - prev))  # never
    total = sum(trans)
    return [t / total for t in trans] if total > 0 else trans


class ScenarioEngine:

    def __init__(self, week: int = None):
        self.week             = week
        self._injuries        = []   # list of injury dicts (all 17 week probs)
        self._ascension_pairs = []   # list of ascension pair dicts
        self._load()

    def _load(self):
        week = self.week

        # ── Injury returns ─────────────────────────────────────────────────
        # Simple binary active/inactive probability per week.
        # When inactive, player gets PPR = 0. No cascade or cluster elevation —
        # use rookie_ascension.csv if you want to model another player stepping up.
        if os.path.exists(INJURY_PATH):
            df = pd.read_csv(INJURY_PATH)
            for _, row in df.iterrows():
                name = str(row['player_name']).strip()
                week_probs = {w: _get_week_prob(row, w) for w in range(1, 18)}
                self._injuries.append({
                    'player':     name,
                    'team':       str(row.get('team', '') or '').strip(),
                    'week_probs': week_probs,
                })

            if self._injuries and week is not None:
                active = [i for i in self._injuries if i['week_probs'].get(week, 1.0) < 1.0]
                if active:
                    print(f"  Injury return scenarios: {len(active)} players")
                    for inj in active:
                        p = inj['week_probs'][week]
                        status = "OUT" if p == 0 else f"{p*100:.0f}% chance active"
                        print(f"    {inj['player']:<30} {status}")

        # ── Rookie ascension ───────────────────────────────────────────────
        if os.path.exists(ASCENSION_PATH):
            df = pd.read_csv(ASCENSION_PATH)
            for _, row in df.iterrows():
                veteran = str(row.get('veteran_name', '') or '').strip()
                cumulative = [min(float(row.get(f'w{w}', 0) or 0), 1.0) for w in range(1, 18)]
                self._ascension_pairs.append({
                    'rookie_name':     str(row['rookie_name']).strip(),
                    'veteran_name':    veteran if veteran else None,
                    'team':            str(row['team']).strip(),
                    'rookie_base':     str(row['rookie_base_cluster']).strip(),
                    'rookie_ascended': str(row['rookie_ascended_cluster']).strip(),
                    'veteran_base':    str(row.get('veteran_base_cluster', '') or '').strip(),
                    'veteran_demoted': str(row.get('veteran_demoted_cluster', '') or '').strip(),
                    'cumulative':      cumulative,
                    'transition':      _transition_probs(cumulative),
                })

            if week is not None:
                active = [p for p in self._ascension_pairs
                          if p['cumulative'][week-1] > 0]
                if active:
                    print(f"  Ascension scenarios: {len(active)} pairs")
                    for pair in active:
                        pct = f"{pair['cumulative'][week-1]*100:.0f}%"
                        vs  = f" (vs {pair['veteran_name']})" if pair['veteran_name'] else " (solo dev)"
                        print(f"    {pair['rookie_name']:<28} {pct} chance of starter role{vs}")

    # ── Single-week application ────────────────────────────────────────────

    def apply(self, sims: pd.DataFrame, dist_df: pd.DataFrame,
              rng: np.random.Generator, week: int = None) -> pd.DataFrame:
        """Apply all scenarios to a single-week simulation matrix."""
        week   = week or self.week
        sims   = sims.copy()
        n      = len(sims)

        # Injury returns: zero out players who are inactive this sim
        for inj in self._injuries:
            p_active = inj['week_probs'].get(week, 1.0)
            player   = inj['player']
            if player not in sims.columns or p_active >= 1.0:
                continue

            if p_active <= 0.0:
                inactive = np.ones(n, dtype=bool)
            else:
                inactive = rng.random(n) >= p_active

            sims[player] = np.where(inactive, 0.0, sims[player].fillna(0.0))

        # Rookie ascension
        for pair in self._ascension_pairs:
            p = pair['cumulative'][week - 1]
            if p <= 0:
                continue
            self._apply_ascension(sims, dist_df, rng, pair,
                                   rng.random(n) < p, n)

        return sims

    def _apply_ascension(self, sims, dist_df, rng, pair,
                          ascended_mask, n):
        rookie  = pair['rookie_name']
        veteran = pair['veteran_name']

        if rookie in sims.columns and pair['rookie_ascended'] != pair['rookie_base']:
            if pair['rookie_ascended'] in dist_df.index:
                asc_vals = _sample_cluster(pair['rookie_ascended'], dist_df, n, rng)
                sims[rookie] = np.where(ascended_mask, asc_vals, sims[rookie].values)

        # Veteran side is optional (blank = solo development)
        if (veteran and veteran in sims.columns and
                pair['veteran_demoted'] and pair['veteran_demoted'] != pair['veteran_base']):
            if pair['veteran_demoted'] in dist_df.index:
                dem_vals = _sample_cluster(pair['veteran_demoted'], dist_df, n, rng)
                sims[veteran] = np.where(ascended_mask, dem_vals, sims[veteran].values)

    # ── Season-long state draws ────────────────────────────────────────────

    def draw_season_states(self, n_sims: int, rng: np.random.Generator) -> dict:
        """
        Pre-draw season-long states for all scenarios.

        Returns a dict with:
          'injury_states':    {player_name: (n_sims,) bool array per week}
          'ascension_weeks':  list of (n_sims,) int arrays, one per pair
                              values 1-17 = ascension week, 18 = never

        For injuries: active_state[player][week][sim] = True if playing.
        The weekly probability curve is used to derive the likely injury
        and return week per simulation, preserving the overall distribution
        while making the timeline coherent (can't be healthy in W3, out in W4,
        healthy in W5 for the same injury).
        """
        states = {'injury_states': {}, 'ascension_weeks': []}

        # ── Injury states: coherent per-sim timelines ──────────────────────
        for inj in self._injuries:
            probs = [inj['week_probs'][w] for w in range(1, 18)]

            # Detect injury/return week ranges from probability curve
            # Segments: p=0 = definitely out, p=1 = definitely healthy,
            # 0<p<1 = uncertain (use midpoint of segment as transition point)
            # Simple model: for each sim, draw a single active/inactive array
            # that respects the probability curve but stays coherent
            per_sim = np.ones((n_sims, 17), dtype=bool)  # default: healthy

            for w_idx, p in enumerate(probs):
                if p >= 1.0:
                    per_sim[:, w_idx] = True
                elif p <= 0.0:
                    per_sim[:, w_idx] = False
                else:
                    # Uncertain week: draw independently (brief window)
                    per_sim[:, w_idx] = rng.random(n_sims) < p

            # Enforce coherence: if a sim has the player out in week W and in
            # in week W+2 but out again in week W+3, it's likely two separate
            # injuries. For simplicity, enforce that recovery is monotonic once
            # the probability starts rising (no re-injury in this model)
            # We do this by finding the "recovery start" week per sim
            # Find the last week with p=0 (definite injury) and first week with p=1
            last_definite_out = max((w for w, p in enumerate(probs) if p <= 0), default=-1)
            first_definite_in = next((w for w, p in enumerate(probs) if p >= 1.0), 17)

            # In sims where player is "active" before last_definite_out week,
            # they should be inactive (corrects the uncertain pre-return weeks)
            if last_definite_out >= 0:
                per_sim[:, :last_definite_out+1] = False

            states['injury_states'][inj['player']] = per_sim

        # ── Ascension weeks: permanent once triggered ──────────────────────
        outcomes = list(range(1, 18)) + [18]  # weeks 1-17 + never
        for pair in self._ascension_pairs:
            draws = rng.choice(outcomes, size=n_sims, p=pair['transition'])
            states['ascension_weeks'].append(draws)

        return states

    def apply_season_week(self, sims: pd.DataFrame, dist_df: pd.DataFrame,
                          rng: np.random.Generator, week: int,
                          season_states: dict) -> pd.DataFrame:
        """Apply pre-drawn season states for one week in a season simulation."""
        sims  = sims.copy()
        n     = len(sims)
        w_idx = week - 1

        # Injuries
        for inj in self._injuries:
            player = inj['player']
            if player not in sims.columns:
                continue
            active_arr = season_states['injury_states'].get(player)
            if active_arr is None:
                continue

            inactive = ~active_arr[:, w_idx]
            sims[player] = np.where(~active_arr[:, w_idx], 0.0, sims[player].fillna(0.0))

            elev_p = inj['elevated_player']
            elev_c = inj['elevated_cluster']
            if elev_p and elev_c and elev_p in sims.columns and inactive.any():
                elevated_vals = _sample_cluster(elev_c, dist_df, n, rng)
                sims[elev_p] = np.where(inactive, elevated_vals, sims[elev_p].values)

        # Ascension
        for i, pair in enumerate(self._ascension_pairs):
            if i >= len(season_states['ascension_weeks']):
                continue
            ascended_mask = season_states['ascension_weeks'][i] <= week
            self._apply_ascension(sims, dist_df, rng, pair, ascended_mask, n)

        return sims

    def has_scenarios(self, week: int = None) -> bool:
        week = week or self.week or 1
        has_injury = any(i['week_probs'].get(week, 1.0) < 1.0 for i in self._injuries)
        has_ascension = any(p['cumulative'][week-1] > 0 for p in self._ascension_pairs)
        return has_injury or has_ascension

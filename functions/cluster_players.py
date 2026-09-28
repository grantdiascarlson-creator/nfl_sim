import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ── Season-level profile aggregation ────────────────────────────────────────────

def build_profiles(weekly_path: str, min_games: int = 4) -> pd.DataFrame:
    df = pd.read_csv(weekly_path)
    active = df[(df['targets'] > 0) | (df['carries'] > 0) | (df['attempts'] > 0)].copy()

    profile = (
        active.groupby(['player_id', 'player_display_name', 'position', 'season'])
        .agg(
            games=('game_key', 'nunique'),
            target_share=('target_share', 'mean'),
            air_yards_share=('air_yards_share', 'mean'),
            wopr=('wopr', 'mean'),
            adot=('adot', 'mean'),
            carry_share=('carry_share', 'mean'),
            carries_pg=('carries', 'mean'),
            targets_pg=('targets', 'mean'),
            attempts_pg=('attempts', 'mean'),
            offense_pct=('offense_pct', 'mean'),
            rz_targets_pg=('rz_targets', 'mean'),
            rz_carries_pg=('rz_carries', 'mean'),
            rz_carry_share=('rz_carry_share', 'mean'),
            ppr_pg=('fantasy_points_ppr', 'mean'),
        )
        .reset_index()
    )

    return profile[profile['games'] >= min_games].copy()


# ── Rule-based clustering ────────────────────────────────────────────────────────

def _rule_cluster(row: pd.Series) -> str:
    pos       = row['position']
    wopr      = row['wopr']
    adot      = row['adot']
    cs        = row['carry_share']
    ts        = row['target_share']
    attpg     = row['attempts_pg']
    snap      = row['offense_pct']
    rz_cs     = row['rz_carry_share']

    snap_known = not (snap is None or (isinstance(snap, float) and np.isnan(snap)))
    adot_known = not (adot is None or (isinstance(adot, float) and np.isnan(adot)))

    if pos == 'QB':
        if attpg < 20:
            return 'QB_backup'
        cpg = row['carries_pg']
        cpg_known = not (cpg is None or (isinstance(cpg, float) and np.isnan(cpg)))
        # Split by rushing involvement — carries/game is the cleanest signal
        # Elite dual-threat: Lamar, peak Hurts, peak Allen — 7+ carries/game, 20-28 PPR
        # Rushing: Allen 2020-25, Kyler, Daniels, Maye — 3-7 carries/game
        # Pocket: Brady, Stafford, Flacco, Brees — <3 carries/game
        # Thresholds calibrated to separate designed runners from scramble QBs:
        # >= 8.0: Lamar/peak Hurts/peak Fields — primary rushing weapon
        # >= 5.0: Allen/Kyler/Daniels/Maye — mobile, use legs meaningfully
        # <  5.0: Dak/Herbert/Mahomes/Burrow — scramble occasionally but pocket QBs
        if cpg_known and cpg >= 8.0:
            return 'QB_dual_threat_elite'
        elif cpg_known and cpg >= 5.0:
            return 'QB_rushing'
        else:
            return 'QB_pocket'

    if pos == 'WR':
        # Low snap — split by aDOT: deep threat package player vs true depth
        if snap_known and snap < 0.45:
            if adot_known and adot >= 14:
                return 'WR_deep_threat'
            return 'WR_depth'
        # On-field players — WOPR differentiates role tier
        if wopr >= 0.60:
            # Split alpha WRs by aDOT: possession alphas get more RZ targets,
            # deep alphas have fatter right tail but lower floor
            if adot_known and adot < 11:
                return 'WR_alpha_possession'   # Kupp, Adams, Chase
            return 'WR_alpha_deep'             # Hill, Diggs, Jefferson
        elif wopr >= 0.40:
            # Split WR2s by absolute volume: featured WR2 (7.5+ tpg) vs standard WR2
            tpg = row['targets_pg']
            tpg_known = not (tpg is None or (isinstance(tpg, float) and np.isnan(tpg)))
            if adot_known and adot <= 8:
                # Possession WR2 — further split by volume
                if tpg_known and tpg >= 8.0:
                    return 'WR_possession_WR2'   # Rashee Rice 2025, Keenan Allen — elite slot
                return 'WR_possession_WR2'       # keep as one for now, n=32 is thin
            else:
                if tpg_known and tpg >= 6.5:
                    return 'WR_WR2_featured'     # Tyreek 2020, Godwin, Pickens — 7.5 tpg
                return 'WR_WR2'                  # standard WR2 — 5.7 tpg
        elif wopr >= 0.22:
            if snap_known and snap < 0.40:
                return 'WR_depth'
            # Split on absolute target volume — same WOPR means different things
            # on a 35-pass team vs a 22-pass team
            tpg = row['targets_pg']
            tpg_known = not (tpg is None or (isinstance(tpg, float) and np.isnan(tpg)))
            if tpg_known and tpg >= 4.5:
                return 'WR_WR3_high_volume'   # JuJu, Rashee, Godwin — pass-heavy offenses
            return 'WR_WR3_rotational'         # Hardman, Boutte — true rotational
        else:
            return 'WR_depth'

    if pos == 'TE':
        if wopr >= 0.38:
            # Elite tier: true target hogs (Kelce, McBride, Waller) — 7+ targets/game
            # Standard: featured TE1 but not a primary receiver (Kittle, Gronk, Andrews)
            tpg = row['targets_pg']
            tpg_known = not (tpg is None or (isinstance(tpg, float) and np.isnan(tpg)))
            # Both conditions required: high volume AND high opportunity share
            # LaPorta/Engram/Njoku get 7+ targets but lower WOPR — they're standard alpha
            if tpg_known and tpg >= 7.0 and wopr >= 0.50:
                return 'TE_alpha_elite'
            return 'TE_alpha'
        elif wopr >= 0.22:
            # Split TE1 by absolute target volume:
            # receiving TE1 (Hooper, Kraft, Schultz — 5+ tpg) vs
            # situational/blocking TE1 (Taysom, Tonyan, Tonges — 4 tpg)
            tpg = row['targets_pg']
            tpg_known = not (tpg is None or (isinstance(tpg, float) and np.isnan(tpg)))
            if tpg_known and tpg >= 5.0:
                return 'TE_TE1_receiving'
            return 'TE_TE1'
        elif wopr >= 0.10:
            return 'TE_TE2_rotational'
        else:
            return 'TE_depth'

    if pos == 'RB':
        # Low snap = depth regardless of carry share
        if snap_known and snap < 0.30:
            return 'RB_backup'

        # True bellcow: dominates total touches — carry share + target share >= 0.75
        total_touch_share = cs + ts
        if total_touch_share >= 0.75:
            # 3-way split confirmed by k-means subcluster analysis:
            # every_down = high snap + receiving (peak McCaffrey, Kyren)
            # dual       = high receiving but comes off field sometimes (Cook, Bijan)
            # runner     = pure workhorse, limited passing-down role (Henry, Taylor)
            snap_known = not (snap is None or (isinstance(snap, float) and np.isnan(snap)))
            if snap_known and snap >= 0.80 and ts >= 0.12:
                return 'RB_bellcow_every_down'
            elif ts >= 0.12:
                return 'RB_bellcow_dual'
            else:
                return 'RB_bellcow_runner'

        # Workhorse RB1: clear lead back by carry share (50%+) but not a bellcow
        # 3-way split based on receiving role and volume:
        #   receiving  — featured in both phases (Gibbs, Ekeler, McCaffrey): ts >= 0.13
        #   volume     — ground-and-pound lead back (Barkley, Cook, Jacobs): cs >= 0.57
        #   early_down — limited on passing downs (Dobbins, Etienne, Zeke): ts < 0.13, cs < 0.57
        if cs >= 0.50:
            if ts < 0.08:
                return 'RB_workhorse_runner'
            elif ts >= 0.13:
                return 'RB_workhorse_receiving'
            elif cs >= 0.57:
                return 'RB_workhorse_volume'
            else:
                return 'RB_workhorse_early_down'

        # Committee comes BEFORE goalline/passing_down:
        # a committee back with high RZ share is tagged 'goalline' via role_tag,
        # not reclassified — the cluster stays committee
        if cs >= 0.25:
            if snap_known and snap >= 0.50:
                # Split featured committee: receiving back vs volume runner
                # Kamara, Ekeler, Achane at 5+ targets/game are genuinely different
                # from Gibbs 2024 or Sanders who run more than they catch
                tpg = row['targets_pg']
                tpg_known = not (tpg is None or (isinstance(tpg, float) and np.isnan(tpg)))
                if tpg_known and tpg >= 4.5:
                    return 'RB_committee_receiving'   # Kamara, Ekeler, Achane — 5.1 tpg, 15.5 ppr
                return 'RB_committee_featured'
            return 'RB_committee_early_down'

        # Goalline specialist: true specialty role, very low overall carry share
        # but dominates red zone (Zach Line types, pure short-yardage)
        if rz_cs >= 0.40:
            return 'RB_goalline'

        # Passing down: low carry share, used mainly as a receiver
        if ts >= 0.10 and cs < 0.30:
            return 'RB_passing_down'

        return 'RB_backup'

    if pos == 'FB':
        return 'FB'

    return f'{pos}_other'


# Clusters where a goalline tag is meaningful (low-tier backs getting RZ work)
_GOALLINE_TAG_CLUSTERS = {
    "RB_committee_featured", "RB_committee_early_down",
    "RB_passing_down", "RB_backup",
}
_GOALLINE_TAG_RZ_THRESHOLD = 0.40


def _assign_role_tags(row: pd.Series) -> str:
    tags = []
    rz_cs = row["rz_carry_share"]
    rz_known = not (rz_cs is None or (isinstance(rz_cs, float) and np.isnan(rz_cs)))
    if row["cluster"] in _GOALLINE_TAG_CLUSTERS and rz_known and rz_cs >= _GOALLINE_TAG_RZ_THRESHOLD:
        tags.append("goalline")
    return ",".join(tags)


def assign_rule_clusters(profile: pd.DataFrame) -> pd.DataFrame:
    out = profile.copy()
    out["cluster"]  = out.apply(_rule_cluster, axis=1)
    out["role_tag"] = out.apply(_assign_role_tags, axis=1)
    return out


# ── Within-cluster k-means sub-analysis ──────────────────────────────────────────
# Features to use when subclustering each rule-based cluster
# Chosen to capture the within-cluster variation most likely to be meaningful

SUBCLUSTER_FEATURES = {
    'WR_alpha_possession':['wopr', 'adot', 'offense_pct', 'rz_targets_pg'],
    'WR_alpha_deep':      ['wopr', 'adot', 'offense_pct', 'rz_targets_pg'],
    'WR_WR2':             ['wopr', 'adot', 'offense_pct', 'rz_targets_pg'],
    'WR_possession_WR2':  ['wopr', 'adot', 'offense_pct', 'rz_targets_pg'],
    'WR_WR3_rotational':  ['wopr', 'adot', 'offense_pct'],
    'WR_deep_threat':     ['adot', 'offense_pct', 'rz_targets_pg'],
    'TE_alpha_elite':     ['wopr', 'adot', 'offense_pct', 'rz_targets_pg'],
    'TE_alpha':           ['wopr', 'adot', 'offense_pct', 'rz_targets_pg'],
    'TE_TE1':             ['wopr', 'adot', 'offense_pct', 'rz_targets_pg'],
    'TE_TE2_rotational':  ['wopr', 'adot', 'offense_pct'],
    'RB_bellcow_every_down': ['carry_share', 'target_share', 'rz_carry_share', 'offense_pct'],
    'RB_bellcow_dual':       ['carry_share', 'target_share', 'rz_carry_share', 'offense_pct'],
    'RB_bellcow_runner':     ['carry_share', 'rz_carry_share', 'offense_pct'],
    'RB_workhorse_receiving':  ['carry_share', 'target_share', 'rz_carry_share', 'offense_pct'],
    'RB_workhorse_volume':     ['carry_share', 'target_share', 'rz_carry_share', 'offense_pct'],
    'RB_workhorse_early_down': ['carry_share', 'target_share', 'rz_carry_share', 'offense_pct'],
    'RB_workhorse_runner':['carry_share', 'rz_carry_share', 'offense_pct'],
    'RB_committee_featured':   ['carry_share', 'target_share', 'rz_carry_share', 'offense_pct'],
    'RB_committee_early_down': ['carry_share', 'target_share', 'rz_carry_share', 'offense_pct'],
    'RB_goalline':        ['rz_carry_share', 'carry_share', 'offense_pct'],
    'RB_passing_down':    ['target_share', 'carry_share', 'offense_pct'],
}


def _interpret_subcluster_split(grp_a: pd.DataFrame, grp_b: pd.DataFrame, features: list) -> str:
    """
    Auto-generate a plain-English description of what distinguishes two subclusters
    by finding the feature with the largest standardized mean difference.
    """
    diffs = {}
    for f in features:
        a_mean = grp_a[f].mean()
        b_mean = grp_b[f].mean()
        pooled_std = np.sqrt((grp_a[f].std()**2 + grp_b[f].std()**2) / 2 + 1e-9)
        diffs[f] = abs(a_mean - b_mean) / pooled_std

    top_feature = max(diffs, key=diffs.get)
    a_val = grp_a[top_feature].mean()
    b_val = grp_b[top_feature].mean()
    hi_grp = 'sub_0' if a_val > b_val else 'sub_1'
    lo_grp = 'sub_1' if hi_grp == 'sub_0' else 'sub_0'

    label_map = {
        'wopr':          'target opportunity (WOPR)',
        'adot':          'depth of target (aDOT)',
        'offense_pct':   'snap share',
        'rz_targets_pg': 'red zone targets',
        'carry_share':   'carry share',
        'target_share':  'target share',
        'rz_carry_share':'red zone carry share',
    }
    feature_label = label_map.get(top_feature, top_feature)
    return (f"primary split on {feature_label}: "
            f"{hi_grp} higher ({a_val if hi_grp=='sub_0' else b_val:.3f}) "
            f"vs {lo_grp} ({b_val if hi_grp=='sub_0' else a_val:.3f})")


def run_subcluster_analysis(
    profile: pd.DataFrame,
    min_cluster_size: int = 40,
    min_silhouette: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run k-means (k=2,3) within each rule-based cluster to surface sub-structure
    that the rule-based logic didn't capture.

    Only reports clusters where k-means finds meaningful separation
    (silhouette score > min_silhouette). Skips clusters too small to subcluster.

    Args:
        profile:           player-season profile with 'cluster' column already assigned
        min_cluster_size:  minimum player-seasons to attempt subclustering
        min_silhouette:    minimum silhouette score to report (below = no real sub-structure)
        seed:              random seed

    Returns:
        profile DataFrame with 'subcluster' column added
    """
    print("\n" + "=" * 70)
    print("WITHIN-CLUSTER K-MEANS SUB-ANALYSIS")
    print("=" * 70)

    profile = profile.copy()
    profile['subcluster'] = profile['cluster']  # default: same as parent cluster

    for cluster_name, cluster_df in profile.groupby('cluster'):
        features = SUBCLUSTER_FEATURES.get(cluster_name)
        if features is None or len(cluster_df) < min_cluster_size:
            continue

        X = cluster_df[features].copy().fillna(cluster_df[features].median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        best_k, best_score, best_labels = None, -1, None
        for k in [2, 3]:
            if len(X_scaled) < k * 10:  # need at least 10 per sub-cluster
                continue
            km = KMeans(n_clusters=k, random_state=seed, n_init=20)
            labels = km.fit_predict(X_scaled)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_k, best_score, best_labels = k, score, labels

        if best_score < min_silhouette:
            print(f"\n  {cluster_name} (n={len(cluster_df)}): "
                  f"no meaningful sub-structure (best silhouette={best_score:.3f})")
            continue

        # Report findings
        print(f"\n  {cluster_name} (n={len(cluster_df)}) — "
              f"k={best_k}, silhouette={best_score:.3f}")

        sub_groups = {}
        for sub_id in range(best_k):
            mask = best_labels == sub_id
            sub_df = cluster_df.iloc[mask]
            sub_groups[f'sub_{sub_id}'] = sub_df
            top = sub_df.nlargest(4, 'ppr_pg')
            examples = ', '.join(
                f"{r['player_display_name']} ({r['season']})" for _, r in top.iterrows()
            )
            feature_summary = '  '.join(
                f"{f}={sub_df[f].mean():.3f}" for f in features
            )
            print(f"    sub_{sub_id} (n={len(sub_df)}): {examples}")
            print(f"           {feature_summary}")
            # Write subcluster label back to profile
            profile.loc[cluster_df.index[mask], 'subcluster'] = f'{cluster_name}_s{sub_id}'

        if best_k == 2:
            interp = _interpret_subcluster_split(
                sub_groups['sub_0'], sub_groups['sub_1'], features
            )
            print(f"    → {interp}")

    return profile


# ── Public entry point ───────────────────────────────────────────────────────────

def assign_clusters(
    weekly_path: str,
    out_path: str,
    min_games: int = 4,
    run_subclusters: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build player-season profiles, apply rule-based clustering, optionally run
    within-cluster k-means sub-analysis, and save the combined profile CSV.

    Args:
        weekly_path:      path to enriched weekly skill CSV
        out_path:         path to write player-season profile CSV
        min_games:        minimum active games to include a player-season
        run_subclusters:  if True, run k-means within each rule-based cluster
        seed:             random seed
    """
    print("[2/6] Building player profiles and clustering...")

    profile = build_profiles(weekly_path, min_games=min_games)
    print(f"    Player-seasons: {len(profile):,} across {profile['season'].nunique()} seasons")

    print("\n  Rule-based clustering...")
    profile = assign_rule_clusters(profile)

    print("\n  Cluster distribution:")
    for cluster, count in profile['cluster'].value_counts().items():
        print(f"    {cluster:<30s} {count:>4d}")

    if run_subclusters:
        profile = run_subcluster_analysis(profile, seed=seed)

    # Sanity check: show where key archetype players landed
    print("\n  Sanity check — key players:")
    checks = [
        ('Derrick Henry',        2021),
        ('Christian McCaffrey',  2023),
        ('Alvin Kamara',         2020),
        ('Adrian Peterson',      2019),
        ('Josh Jacobs',          2022),
        ('Austin Ekeler',        2022),
        ('James White',          2019),
        ('Cooper Kupp',          2021),   # expect WR_alpha_possession
        ('Tyreek Hill',          2023),   # expect WR_alpha_deep
        ('Travis Kelce',         2022),
        ('Marquise Goodwin',     2019),
    ]
    for name, season in checks:
        row = profile[
            (profile['player_display_name'] == name) &
            (profile['season'] == season)
        ]
        if not row.empty:
            r = row.iloc[0]
            print(f"    {name} ({season}): {r['cluster']}"
                  f"  cs={r['carry_share']:.2f}  ts={r['target_share']:.2f}"
                  f"  snap={r['offense_pct']:.2f}  rz_cs={r['rz_carry_share']:.2f}"
                  f"  ppr={r['ppr_pg']:.1f}")
        else:
            print(f"    {name} ({season}): not found (below min_games threshold)")

    # ── Effective cluster: subcluster where n>=30, parent cluster otherwise ──
    MIN_SUBCLUSTER_SAMPLES = 30
    sub_counts = profile['subcluster'].value_counts().to_dict()

    def _effective_cluster(row) -> str:
        sub = row['subcluster']
        # If subcluster is null, empty, or identical to parent (no k-means ran)
        if not sub or pd.isna(sub) or sub == row['cluster']:
            return row['cluster']
        # Use subcluster only if it has sufficient sample size
        if sub_counts.get(sub, 0) >= MIN_SUBCLUSTER_SAMPLES:
            return sub
        return row['cluster']

    profile['effective_cluster'] = profile.apply(_effective_cluster, axis=1)

    n_using_sub = (profile['effective_cluster'] != profile['cluster']).sum()
    n_sub_clusters = profile['effective_cluster'].nunique()
    print(f"    Using subcluster as effective cluster: {n_using_sub} player-seasons")
    print(f"    Total effective clusters: {n_sub_clusters}")

    profile.to_csv(out_path, index=False)
    print(f"\n    Saved to {out_path}")

    return profile

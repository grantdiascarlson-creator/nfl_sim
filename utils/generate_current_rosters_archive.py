"""
Generate current roster cluster assignments for manual review.

Sources:
  1. Underdog ADP CSV (from bbmdb.com export) — player list with current position
  2. Footballguys ADP page — team info (scraped above)
  3. Historical profile (outputs/player_season_profile.csv) — inherited clusters

Output: inputs/current_rosters.csv with one row per player, sorted by ADP.
Edit the cluster_override column to reassign roles for the upcoming season.
"""

import pandas as pd
import numpy as np
import re


ADP_PATH     = 'inputs/Underdog_Draft_Table_2026-07-22.csv'
PROFILE_PATH = 'outputs/player_season_profile.csv'
WEEKLY_PATH  = 'outputs/weekly_skill_enriched.csv'
OUT_PATH     = 'inputs/current_rosters.csv'

# Team lookup scraped from footballguys.com/adp — player name -> (team, bye)
TEAM_LOOKUP_RAW = """Jahmyr Gibbs|DET|6
Bijan Robinson|ATL|11
Ja'Marr Chase|CIN|6
Puka Nacua|LAR|11
Jaxon Smith-Njigba|SEA|11
Christian McCaffrey|SF|8
Amon-Ra St. Brown|DET|6
Jonathan Taylor|IND|13
Ashton Jeanty|LV|13
James Cook III|BUF|7
Justin Jefferson|MIN|6
CeeDee Lamb|DAL|14
De'Von Achane|MIA|6
Saquon Barkley|PHI|10
Omarion Hampton|LAC|7
Chase Brown|CIN|6
Drake London|ATL|11
Ken Walker III|KC|5
Brock Bowers|LV|13
Derrick Henry|BAL|13
A.J. Brown|NE|11
Nico Collins|HOU|8
Jeremiyah Love|ARI|14
Trey McBride|ARI|14
George Pickens|DAL|14
Josh Allen|BUF|7
Rashee Rice|KC|5
Chris Olave|NO|8
DeVonta Smith|PHI|10
Kyren Williams|LAR|11
Breece Hall|NYJ|13
Josh Jacobs|GB|11
Malik Nabers|NYG|8
Javonte Williams|DAL|14
Zay Flowers|BAL|13
Tee Higgins|CIN|6
Emeka Egbuka|TB|10
Travis Etienne Jr.|NO|8
Tetairoa McMillan|CAR|5
Garrett Wilson|NYJ|13
Ladd McConkey|LAC|7
Colston Loveland|CHI|10
Cam Skattebo|NYG|8
Jaylen Waddle|DEN|10
Luther Burden III|CHI|10
Davante Adams|LAR|11
Lamar Jackson|BAL|13
Terry McLaurin|WAS|7
TreVeyon Henderson|NE|11
Bucky Irving|TB|10
Quinshon Judkins|CLE|11
David Montgomery|HOU|8
D'Andre Swift|CHI|10
Jameson Williams|DET|6
DJ Moore|BUF|7
Mike Evans|SF|8
Drake Maye|NE|11
Tyler Warren|IND|13
Joe Burrow|CIN|6
Bhayshul Tuten|JAX|7
Rome Odunze|CHI|10
Christian Watson|GB|11
Carnell Tate|TEN|9
Jayden Daniels|WAS|7
Jadarian Price|SEA|11
Marvin Harrison Jr.|ARI|14
Jalen Hurts|PHI|10
Jaylen Warren|PIT|9
Brian Thomas Jr.|JAX|7
Chuba Hubbard|CAR|5
Tony Pollard|TEN|9
Jordyn Tyson|NO|8
DK Metcalf|PIT|9
Caleb Williams|CHI|10
Tucker Kraft|GB|11
Alec Pierce|IND|13
Courtland Sutton|DEN|10
Parker Washington|JAX|7
RJ Harvey|DEN|10
Rhamondre Stevenson|NE|11
Harold Fannin Jr.|CLE|11
Kyle Pitts Sr.|ATL|11
Justin Herbert|LAC|7
Dak Prescott|DAL|14
Rico Dowdle|PIT|9
Chris Godwin Jr.|TB|10
Sam LaPorta|DET|6
Jaxson Dart|NYG|8
Trevor Lawrence|JAX|7
Makai Lemon|PHI|10
Michael Wilson|ARI|14
Kyle Monangai|CHI|10
Michael Pittman Jr|PIT|9
Kenny Gainwell|TB|10
Quentin Johnston|LAC|7
Matthew Stafford|LAR|11
Blake Corum|LAR|11
J.K. Dobbins|DEN|10
Ricky Pearsall|SF|8
Brock Purdy|SF|8
Patrick Mahomes II|KC|5
George Kittle|SF|8
Jakobi Meyers|JAX|7
Jordan Addison|MIN|6
Jayden Reed|GB|11
Rachaad White|WAS|7
Jonathon Brooks|CAR|5
Josh Downs|IND|13
Wan'Dale Robinson|TEN|9
Bo Nix|DEN|10
Travis Kelce|KC|5
Jared Goff|DET|6
Aaron Jones Sr.|MIN|6
Jake Ferguson|DAL|14
Xavier Worthy|KC|5
Jacory Croskey-Merritt|WAS|7
Jordan Mason|MIN|6
Isaiah Likely|NYG|8
Dalton Kincaid|BUF|7
Kyler Murray|MIN|6
Dallas Goedert|PHI|10
Matthew Golden|GB|11
Romeo Doubs|NE|11
Mark Andrews|BAL|13
Jordan Love|GB|11
Baker Mayfield|TB|10
Tyler Shough|NO|8
KC Concepcion|CLE|11
Khalil Shakir|BUF|7
Jayden Higgins|HOU|8
Zach Charbonnet|SEA|11
Rashid Shaheed|SEA|11
Tyrone Tracy Jr.|NYG|8
Jalen Coker|CAR|5
Woody Marks|HOU|8
Chris Rodriguez Jr.|JAX|7
Malik Willis|MIA|6
Sam Darnold|SEA|11
Isiah Pacheco|DET|6
Tyler Allgeier|ARI|14
Tyjae Spears|TEN|9
Juwan Johnson|NO|8
Brenton Strange|JAX|7
Oronde Gadsden|LAC|7
C.J. Stroud|HOU|8
Hunter Henry|NE|11
Chig Okonkwo|WAS|7
Kenyon Sadiq|NYJ|13
Cam Ward|TEN|9
Omar Cooper Jr.|NYJ|13
Alvin Kamara|NO|8
Jalen McMillan|TB|10
Denzel Boston|CLE|11
Travis Hunter|JAX|7
Keaton Mitchell|LAC|7
Jauan Jennings|MIN|6
Bryce Young|CAR|5
Jonah Coleman|DEN|10
T.J. Hockenson|MIN|6
Brian Robinson Jr|ATL|11
Daniel Jones|IND|13
Jalen Nailor|LV|13
Dylan Sampson|CLE|11
Tank Bigsby|PHI|10
Dalton Schultz|HOU|8
Tre Tucker|LV|13
Jerry Jeudy|CLE|11
Fernando Mendoza|LV|13
Aaron Rodgers|PIT|9
AJ Barner|SEA|11
Antonio Williams|WAS|7
Emmett Johnson|KC|5
Adonai Mitchell|NYJ|13
Braelon Allen|NYJ|13
Terrance Ferguson|LAR|11
Nicholas Singleton|TEN|9
Kaytron Allen|WAS|7
Ray Davis|BUF|7
Greg Dulcich|MIA|6
Gunnar Helm|TEN|9
Kimani Vidal|LAC|7
Kayshon Boutte|NE|11
Geno Smith|NYJ|13
Zachariah Branch|ATL|11
MarShawn Lloyd|GB|11
Brandon Aiyuk|SF|8
Tank Dell|HOU|8
James Conner|ARI|14
Cade Otton|TB|10
Emanuel Wilson|SEA|11
Darnell Mooney|NYG|8
Dontayvion Wicks|PHI|10
Germie Bernard|PIT|9
George Holani|SEA|11
Cooper Kupp|SEA|11
Pat Bryant|DEN|10
Keon Coleman|BUF|7
Mike Gesicki|CIN|6
Eli Stowers|PHI|10
Chris Bell|MIA|6
Jaylin Noel|HOU|8
Shedeur Sanders|CLE|11
Deshaun Watson|CLE|11
Elijah Sarratt|BAL|13
Rashod Bateman|BAL|13
Evan Engram|DEN|10
Jayden Wright|MIA|6
Kirk Cousins|LV|13
Chimere Dike|TEN|9
J.J. McCarthy|MIN|6
Samaje Perine|CIN|6
Mason Taylor|NYJ|13
Darnell Washington|PIT|9
Jake Tonges|SF|8
Tre' Harris|LAC|7
Malik Washington|MIA|6
Calvin Ridley|TEN|9
Mike Washington Jr.|LV|13
Isaac TeSlaa|DET|6
Adonai Mitchell|NYJ|13
Jalan Tolbert|MIA|6
Jordan James|SF|8
Chris Brooks|GB|11
Jaydon Blue|DAL|14
Sean Tucker|TB|10
Jacoby Brissett|ARI|14
Kaelon Black|SF|8
Troy Franklin|DEN|10
Justice Hill|BAL|13
Demond Claiborne|MIN|6
Tua Tagovailoa|ATL|11
Trey Benson|ARI|14
Emari Demercado|KC|5
Tory Horton|SEA|11
De'Zhaun Stribling|SF|8
Eli Stowers|PHI|10
Will Howard|PIT|9
Andrei Iosivas|CIN|6
Jaylen Wright|MIA|6
Roman Hemby|LV|13
Tyquan Thornton|KC|5
Darius Slayton|NYG|8
Adam Randall|BAL|13
Devin Singletary|NYG|8
Devaughn Vele|NO|8
Jack Bech|LV|13
Ryan Flournoy|DAL|14
Elic Ayomanor|TEN|9
LeQuint Allen Jr.|JAX|7
Hollywood Brown|PHI|10
Malachi Fields|NYG|8
Skyler Bell|BUF|7
Pat Freiermuth|PIT|9
David Njoku|LAC|7
Colby Parkinson|LAR|11
Terrance Ferguson|LAR|11
AJ Barner|SEA|11
Gunnar Helm|TEN|9
Mason Taylor|NYJ|13
Theo Johnson|NYG|8
Darnell Washington|PIT|9
Jake Tonges|SF|8
Chris Brooks|GB|11
MarShawn Lloyd|GB|11
Emmett Johnson|KC|5
Jonah Coleman|DEN|10
Zachariah Branch|ATL|11
Isaac TeSlaa|DET|6
Jauan Jennings|MIN|6
Omar Cooper Jr.|NYJ|13
Jalen Nailor|LV|13
Dylan Sampson|CLE|11
Antonio Williams|WAS|7
Wa'Dale Robinson|TEN|9
Jacory Croskey-Merritt|WAS|7
Jordan Mason|MIN|6
Denzel Boston|CLE|11
Kayshon Boutte|NE|11
Nicholas Singleton|TEN|9
Brian Robinson Jr.|ATL|11
Kaytron Allen|WAS|7
Braelon Allen|NYJ|13
Jonathon Brooks|CAR|5
Demond Claiborne|MIN|6
Mike Washington Jr.|LV|13
Chris Rodriguez Jr.|JAX|7
Bhayshul Tuten|JAX|7
George Holani|SEA|11
Emanuel Wilson|SEA|11
KC Concepcion|CLE|11
Jayden Higgins|HOU|8
Woody Marks|HOU|8
Jalen Coker|CAR|5
Tyrone Tracy Jr.|NYG|8
Darnell Mooney|NYG|8
Germie Bernard|PIT|9
Isiah Pacheco|DET|6
Tank Dell|HOU|8
Rashid Shaheed|SEA|11
Isaiah Likely|NYG|8
Jacob Croskey-Merritt|WAS|7
Dalton Schultz|HOU|8
Ollie Gordon II|MIA|6
DJ Giddens|IND|13
Ty Johnson|BUF|7
Adam Randall|BAL|13
Jordan James|SF|8
Kaelon Black|SF|8
Troy Franklin|DEN|10
Justice Hill|BAL|13
Trey Benson|ARI|14
Emari Demercado|KC|5
Tory Horton|SEA|11
Andrei Iosivas|CIN|6
Darius Slayton|NYG|8
Devaughn Vele|NO|8
Jack Bech|LV|13
Elic Ayomanor|TEN|9
LeQuint Allen Jr.|JAX|7
Hollywood Brown|PHI|10
Malachi Fields|NYG|8
Skyler Bell|BUF|7
Jaydon Blue|DAL|14"""

def build_team_lookup():
    import unicodedata
    def norm(name):
        name = unicodedata.normalize('NFD', str(name))
        name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
        name = name.lower().strip()
        name = re.sub(r'\b(jr\.?|sr\.?|ii|iii|iv)\b', '', name)
        name = name.replace('.', '').replace("\u2019", "'")
        name = re.sub(r'\s+', ' ', name).strip()
        return name

    lookup = {}
    for line in TEAM_LOOKUP_RAW.strip().split('\n'):
        parts = line.split('|')
        if len(parts) >= 2:
            player = parts[0].strip()
            team   = parts[1].strip()
            lookup[norm(player)] = team

    # Manual aliases for genuine name variants between sources
    aliases = {
        'kenneth walker':  lookup.get('ken walker', 'KC'),
        'marvin harrison': lookup.get('marvin harrison', 'ARI'),
        'brian thomas':    lookup.get('brian thomas', 'JAX'),
        'kyle pitts':      lookup.get('kyle pitts', 'ATL'),
        'aaron jones':     lookup.get('aaron jones', 'MIN'),
    }
    lookup.update(aliases)
    return lookup

def normalize_name(name):
    name = str(name).lower().strip()
    # Remove suffixes
    import re
    name = re.sub(r'\b(jr\.?|sr\.?|ii|iii|iv)\b', '', name)
    name = name.replace('.', '').replace("'", "'")
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def generate_current_rosters():
    adp_df  = pd.read_csv(ADP_PATH)
    profile = pd.read_csv(PROFILE_PATH)
    weekly  = pd.read_csv(WEEKLY_PATH, low_memory=False)
    print(f"ADP file: {len(adp_df)} players")

    # Filter to skill positions only
    skill_pos = ['QB', 'RB', 'WR', 'TE']
    adp_df = adp_df[adp_df['Position'].isin(skill_pos)].copy()
    adp_df = adp_df.rename(columns={
        'Player': 'player_name',
        'Position': 'position',
        'ADP on July 21': 'adp',
        'Position Rank': 'position_rank',
    })
    adp_df['adp'] = pd.to_numeric(adp_df['adp'], errors='coerce')
    adp_df = adp_df.sort_values('adp').reset_index(drop=True)

    # Build team lookup: primary = nflverse 2025 data, fallback = footballguys page
    # nflverse is more complete; footballguys catches FA signings and trades
    fbg_lookup  = build_team_lookup()
    adp_df['name_norm'] = adp_df['player_name'].apply(normalize_name)

    # nflverse 2025 team: most common team each player appeared for in week 1-17
    nfl_team_2025 = (
        weekly[weekly['season'] == 2025]
        .groupby('player_display_name')['team']
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )
    nfl_team_2025['name_norm'] = nfl_team_2025['player_display_name'].apply(normalize_name)
    nfl_lookup = dict(zip(nfl_team_2025['name_norm'], nfl_team_2025['team']))

    # Merge: footballguys (current) overrides nflverse (may be one year stale)
    combined_lookup = {**nfl_lookup, **fbg_lookup}
    adp_df['team'] = adp_df['name_norm'].map(combined_lookup).fillna('UNK')

    # (profile and weekly already loaded above)
    team_from_weekly = (
        weekly.groupby(['player_id', 'season'])['team']
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )
    profile = profile.merge(team_from_weekly, on=['player_id', 'season'], how='left')

    # Get most recent season cluster per player name
    recent = (
        profile.sort_values('season', ascending=False)
        .drop_duplicates(subset=['player_display_name'])
        [['player_display_name', 'season', 'cluster', 'effective_cluster', 'subcluster', 'role_tag',
          'games', 'ppr_pg', 'offense_pct', 'wopr', 'adot',
          'carry_share', 'target_share', 'rz_carry_share', 'rz_targets_pg']]
        .copy()
    )
    recent['name_norm'] = recent['player_display_name'].apply(normalize_name)
    recent = recent.rename(columns={'season': 'profile_season'})

    # Join cluster onto ADP list
    out = adp_df.merge(
        recent,
        on='name_norm',
        how='left'
    )

    # Add blank override column
    out.insert(out.columns.get_loc('cluster') + 1 if 'cluster' in out.columns else len(out.columns),
               'cluster_override', '')

    # Flag unmatched
    out['note'] = ''
    out.loc[out['cluster'].isna(), 'note'] = 'No historical profile — new player or position change; set cluster_override manually'
    out.loc[out['team'] == 'UNK', 'note'] += ' | Team not found in lookup'

    # Sort by ADP
    out = out.sort_values('adp').reset_index(drop=True)

    # Round stats
    stat_cols = ['ppr_pg', 'offense_pct', 'wopr', 'adot', 'carry_share', 'target_share', 'rz_carry_share', 'rz_targets_pg']
    for col in stat_cols:
        if col in out.columns:
            out[col] = out[col].round(3)

    keep_cols = ['Rank', 'player_name', 'position', 'position_rank', 'team', 'adp',
                 'profile_season', 'cluster', 'cluster_override', 'effective_cluster', 'subcluster', 'role_tag',
                 'games', 'ppr_pg', 'offense_pct', 'wopr', 'adot',
                 'carry_share', 'target_share', 'rz_carry_share', 'rz_targets_pg', 'note']
    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols]

    out.to_csv(OUT_PATH, index=False)

    n_matched   = out['cluster'].notna().sum()
    n_unmatched = out['cluster'].isna().sum()
    n_unk_team  = (out['team'] == 'UNK').sum()

    print(f"Current roster file: {OUT_PATH}")
    print(f"  Total players:  {len(out)}")
    print(f"  With cluster:   {n_matched}")
    print(f"  No cluster:     {n_unmatched} (new players — set cluster_override)")
    print(f"  Unknown team:   {n_unk_team}")
    print(f"\nCluster distribution:")
    for cluster, count in out['cluster'].value_counts().items():
        print(f"  {cluster:<35} {count:>4}")

    return out

if __name__ == '__main__':
    generate_current_rosters()

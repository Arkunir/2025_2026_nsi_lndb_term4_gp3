import pandas as pd
import numpy as np

# --- 1. Chargement et Préparation des Données Brutes (Identique) ---
print("Chargement des données brutes...")
results_df = pd.read_csv('True_Version/results.csv')
fifa_ranking_df = pd.read_csv('True_Version/fifa_ranking.csv')

fifa_ranking_df = fifa_ranking_df.rename(columns={'country_full': 'country', 'rank_date': 'date'})
results_df = results_df.rename(columns={'date': 'date'})
results_df['date'] = pd.to_datetime(results_df['date'])
fifa_ranking_df['date'] = pd.to_datetime(fifa_ranking_df['date'])

def get_fifa_rank(team, date):
    team_rankings = fifa_ranking_df[fifa_ranking_df['country'] == team]
    if team_rankings.empty: return np.nan
    relevant_rankings = team_rankings[team_rankings['date'] < date]
    if relevant_rankings.empty: return np.nan
    return relevant_rankings.iloc[-1]['rank']

print("Fusion des données de classement FIFA...")
results_df['home_team_rank'] = results_df.apply(lambda row: get_fifa_rank(row['home_team'], row['date']), axis=1)
results_df['away_team_rank'] = results_df.apply(lambda row: get_fifa_rank(row['away_team'], row['date']), axis=1)
results_df.dropna(inplace=True)

# --- 2. Feature Engineering (Version 7 - Retour à l'Essentiel) ---
def calculate_recent_form(team, date, matches=5):
    team_matches = results_df[((results_df['home_team'] == team) | (results_df['away_team'] == team)) & (results_df['date'] < date)]
    team_matches = team_matches.sort_values(by='date').tail(matches)
    points = 0
    for _, row in team_matches.iterrows():
        if row['home_team'] == team:
            if row['home_score'] > row['away_score']: points += 3
            elif row['home_score'] == row['away_score']: points += 1
        else:
            if row['away_score'] > row['home_score']: points += 3
            elif row['away_score'] == row['home_score']: points += 1
    return points

def calculate_goal_difference(team, date, matches=5):
    team_matches = results_df[((results_df['home_team'] == team) | (results_df['away_team'] == team)) & (results_df['date'] < date)]
    team_matches = team_matches.sort_values(by='date').tail(matches)
    if len(team_matches) == 0: return 0
    goals_scored = team_matches.apply(lambda row: row['home_score'] if row['home_team'] == team else row['away_score'], axis=1).sum()
    goals_conceded = team_matches.apply(lambda row: row['away_score'] if row['home_team'] == team else row['home_score'], axis=1).sum()
    return (goals_scored - goals_conceded) / len(team_matches)

def get_h2h_points_diff(home_team, away_team, date, matches=5):
    h2h_matches = results_df[((results_df['home_team'] == home_team) & (results_df['away_team'] == away_team)) | ((results_df['home_team'] == away_team) & (results_df['away_team'] == home_team))]
    h2h_matches = h2h_matches[h2h_matches['date'] < date].sort_values(by='date').tail(matches)
    home_points, away_points = 0, 0
    for _, row in h2h_matches.iterrows():
        if row['home_team'] == home_team:
            if row['home_score'] > row['away_score']: home_points += 3
            elif row['home_score'] == row['away_score']: home_points += 1
            else: away_points += 3
        else:
            if row['away_score'] > row['home_score']: away_points += 3
            elif row['away_score'] == row['home_score']: away_points += 1
            else: home_points += 3
    return home_points - away_points

print("Calcul des features essentielles pour la V7...")
results_df['home_team_form'] = results_df.apply(lambda row: calculate_recent_form(row['home_team'], row['date']), axis=1)
results_df['away_team_form'] = results_df.apply(lambda row: calculate_recent_form(row['away_team'], row['date']), axis=1)
results_df['home_goal_diff'] = results_df.apply(lambda row: calculate_goal_difference(row['home_team'], row['date']), axis=1)
results_df['away_goal_diff'] = results_df.apply(lambda row: calculate_goal_difference(row['away_team'], row['date']), axis=1)
results_df['h2h_points_diff'] = results_df.apply(lambda row: get_h2h_points_diff(row['home_team'], row['away_team'], row['date']), axis=1)
results_df['tournament_importance'] = results_df['tournament'].apply(lambda t: 4 if 'FIFA World Cup' in t else (3 if 'UEFA Euro' in t or 'Copa América' in t else (2 if 'qualification' in t else 1)))
results_df['is_neutral'] = results_df['neutral'].astype(int)

# Créer les variables cibles
results_df['goal_diff'] = results_df['home_score'] - results_df['away_score']
def get_match_result(row):
    if row['home_score'] > row['away_score']: return 1
    if row['home_score'] < row['away_score']: return 2
    return 0
results_df['result'] = results_df.apply(get_match_result, axis=1)

# --- 3. Sauvegarde des Données Prétraitées V7 ---
# NOUVEAU : Liste de features plus simple et ciblée
features_v7 = [
    'home_team_rank', 'away_team_rank', 'home_team_form', 'away_team_form',
    'home_goal_diff', 'away_goal_diff', 'tournament_importance', 'is_neutral',
    'h2h_points_diff'
]

final_df = results_df[features_v7 + ['goal_diff', 'result']].dropna()

output_filename = 'donnees_pretraitees_football_v7.csv'
final_df.to_csv(output_filename, index=False)

print(f"\nFICHIER CRÉÉ AVEC SUCCÈS : '{output_filename}'")
print("Vous pouvez maintenant lancer le script 'etape2_modele_prediction_v7.py'.")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from xgboost import XGBRegressor
from scipy.stats import norm

# --- 1. Chargement et Préparation des Données (Identique) ---
print("Chargement des données...")
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

# --- 2. Feature Engineering (Identique) ---
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

print("Calcul de toutes les features...")
results_df['home_team_form'] = results_df.apply(lambda row: calculate_recent_form(row['home_team'], row['date']), axis=1)
results_df['away_team_form'] = results_df.apply(lambda row: calculate_recent_form(row['away_team'], row['date']), axis=1)
results_df['home_goal_diff'] = results_df.apply(lambda row: calculate_goal_difference(row['home_team'], row['date']), axis=1)
results_df['away_goal_diff'] = results_df.apply(lambda row: calculate_goal_difference(row['away_team'], row['date']), axis=1)
results_df['h2h_points_diff'] = results_df.apply(lambda row: get_h2h_points_diff(row['home_team'], row['away_team'], row['date']), axis=1)
results_df['tournament_importance'] = results_df['tournament'].apply(lambda t: 4 if 'FIFA World Cup' in t else (3 if 'UEFA Euro' in t or 'Copa América' in t else (2 if 'qualification' in t else 1)))
results_df['is_neutral'] = results_df['neutral'].astype(int)

# --- 3. Préparation pour la Régression (NOUVELLE APPROCHE) ---

features = [
    'home_team_rank', 'away_team_rank', 'home_team_form', 'away_team_form',
    'home_goal_diff', 'away_goal_diff', 'tournament_importance', 'is_neutral',
    'h2h_points_diff'
]

# NOUVEAU : La cible est l'écart de buts (continu)
results_df['goal_diff'] = results_df['home_score'] - results_df['away_score']

# On garde aussi le résultat pour l'évaluation finale
def get_match_result(row):
    if row['home_score'] > row['away_score']: return 1
    if row['home_score'] < row['away_score']: return 2
    return 0
results_df['result'] = results_df.apply(get_match_result, axis=1)

final_df = results_df[features + ['goal_diff', 'result']].dropna()

# --- 4. Entraînement du Modèle de Régression ---

X = final_df[features]
y_reg = final_df['goal_diff'] # Cible de régression

# CORRECTION : Ajout de la variable manquante y_result_train
X_train, X_test, y_reg_train, y_reg_test, y_result_train, y_result_test = train_test_split(
    X, y_reg, final_df['result'], test_size=0.2, random_state=42, stratify=final_df['result']
)

print("\n--- Entraînement du Modèle de Régression XGBoost ---")
# On utilise XGBRegressor
regressor = XGBRegressor(
    objective='reg:squarederror', 
    eval_metric='rmse', 
    random_state=42,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5
)
regressor.fit(X_train, y_reg_train)

# --- 5. Évaluation du Modèle de Régression ---

print("\n--- Évaluation du Modèle de Régression ---")
y_reg_pred = regressor.predict(X_test)

# Métriques de régression
mae = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
print(f"Erreur Absolue Moyenne (MAE) : {mae:.2f} buts")
print(f"Erreur Quadratique Moyenne (RMSE) : {rmse:.2f} buts")
print(f"Cela signifie qu'en moyenne, le modèle se trompe de {mae:.2f} buts sur l'écart final.")

# Conversion de la prédiction de régression en prédiction de classification
# Si l'écart prédit est > 0.5 -> Victoire Domicile
# Si l'écart prédit est < -0.5 -> Victoire Extérieur
# Sinon -> Nul
y_class_pred = np.where(y_reg_pred > 0.5, 1, np.where(y_reg_pred < -0.5, 2, 0))

# Calcul de la précision de classification
accuracy = accuracy_score(y_result_test, y_class_pred)
print(f"\nPrécision après conversion en Win/Draw/Loss : {accuracy:.2%}")

# --- 6. Fonction de Prédiction Finale (Régression) ---

def predict_match_regression(home_team, away_team, tournament_name, is_neutral_ground, model):
    today = pd.to_datetime('today')
    home_rank = get_fifa_rank(home_team, today)
    away_rank = get_fifa_rank(away_team, today)
    
    if pd.isna(home_rank) or pd.isna(away_rank):
        return "Erreur : Une des équipes n'est pas dans le classement FIFA."

    match_data = pd.DataFrame({
        'home_team_rank': [home_rank], 'away_team_rank': [away_rank],
        'home_team_form': [calculate_recent_form(home_team, today)],
        'away_team_form': [calculate_recent_form(away_team, today)],
        'home_goal_diff': [calculate_goal_difference(home_team, today)],
        'away_goal_diff': [calculate_goal_difference(away_team, today)],
        'tournament_importance': [4 if 'FIFA World Cup' in tournament_name else 1],
        'is_neutral': [1 if is_neutral_ground else 0],
        'h2h_points_diff': [get_h2h_points_diff(home_team, away_team, today)]
    })

    print(f"\n--- PRÉDICTION PAR RÉGRESSION pour {home_team} vs {away_team} ---")
    
    # Prédiction de l'écart de buts
    predicted_diff = model.predict(match_data)[0]
    print(f"Écart de buts prédit : {predicted_diff:.2f}")
    
    # Calcul des probabilités en utilisant une distribution normale
    # L'écart-type (sigma) est basé sur l'erreur du modèle (RMSE)
    sigma = rmse
    
    prob_draw = norm.cdf(0.5, loc=predicted_diff, scale=sigma) - norm.cdf(-0.5, loc=predicted_diff, scale=sigma)
    prob_home_win = 1 - norm.cdf(0.5, loc=predicted_diff, scale=sigma)
    prob_away_win = norm.cdf(-0.5, loc=predicted_diff, scale=sigma)

    # Détermination du résultat le plus probable
    probabilities = {'Nul': prob_draw, f'Victoire de {home_team}': prob_home_win, f'Victoire de {away_team}': prob_away_win}
    most_probable_result = max(probabilities, key=probabilities.get)
    
    print(f"\nRésultat le plus probable : {most_probable_result}")
    print("Probabilités calculées :")
    print(f"  - Victoire de {home_team}: {prob_home_win:.2%}")
    print(f"  - Match Nul: {prob_draw:.2%}")
    print(f"  - Victoire de {away_team}: {prob_away_win:.2%}")


# --- Exemple d'utilisation ---
predict_match_regression('France', 'Brazil', 'FIFA World Cup', True, regressor)
predict_match_regression('Spain', 'Germany', 'FIFA World Cup qualification', False, regressor)
predict_match_regression('Italy', 'England', 'UEFA Euro', True, regressor)
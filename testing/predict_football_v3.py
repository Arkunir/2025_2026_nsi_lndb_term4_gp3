import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

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

# --- 2. Feature Engineering (Amélioré) ---

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

# AMÉLIORATION FINALE 1 : Feature Head-to-Head (H2H)
def get_h2h_points_diff(home_team, away_team, date, matches=5):
    h2h_matches = results_df[
        ((results_df['home_team'] == home_team) & (results_df['away_team'] == away_team)) |
        ((results_df['home_team'] == away_team) & (results_df['away_team'] == home_team))
    ]
    h2h_matches = h2h_matches[h2h_matches['date'] < date].sort_values(by='date').tail(matches)
    
    home_points = 0
    away_points = 0
    
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

print("Calcul de toutes les features (y compris H2H)...")
results_df['home_team_form'] = results_df.apply(lambda row: calculate_recent_form(row['home_team'], row['date']), axis=1)
results_df['away_team_form'] = results_df.apply(lambda row: calculate_recent_form(row['away_team'], row['date']), axis=1)
results_df['home_goal_diff'] = results_df.apply(lambda row: calculate_goal_difference(row['home_team'], row['date']), axis=1)
results_df['away_goal_diff'] = results_df.apply(lambda row: calculate_goal_difference(row['away_team'], row['date']), axis=1)
results_df['h2h_points_diff'] = results_df.apply(lambda row: get_h2h_points_diff(row['home_team'], row['away_team'], row['date']), axis=1)

results_df['tournament_importance'] = results_df['tournament'].apply(lambda t: 4 if 'FIFA World Cup' in t else (3 if 'UEFA Euro' in t or 'Copa América' in t else (2 if 'qualification' in t else 1)))
results_df['is_neutral'] = results_df['neutral'].astype(int)

def get_match_result(row):
    if row['home_score'] > row['away_score']: return 1
    if row['home_score'] < row['away_score']: return 2
    return 0

results_df['result'] = results_df.apply(get_match_result, axis=1)

# Mise à jour de la liste des features
features = [
    'home_team_rank', 'away_team_rank', 'home_team_form', 'away_team_form',
    'home_goal_diff', 'away_goal_diff', 'tournament_importance', 'is_neutral',
    'h2h_points_diff' # NOUVELLE FEATURE
]

final_df = results_df[features + ['result']].dropna()

# --- 3. Entraînement des Modèles Ultime ---

print("\nPréparation de l'entraînement du modèle Ultime...")
X = final_df[features]
y = final_df['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# AMÉLIORATION FINALE 2 : Optimisation des Hyperparamètres pour XGBoost
print("Recherche des meilleurs hyperparamètres pour XGBoost (cela peut prendre un moment)...")
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}
# Utilisation de RandomizedSearchCV pour une recherche efficace
xgb_search = RandomizedSearchCV(
    estimator=XGBClassifier(eval_metric='mlogloss', random_state=42),
    param_distributions=xgb_param_grid,
    n_iter=20, # Nombre de combinaisons à tester
    scoring='accuracy',
    cv=3, # Validation croisée à 3 plis
    verbose=1,
    random_state=42,
    n_jobs=-1 # Utilise tous les cœurs du CPU
)
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_
print(f"Meilleurs paramètres XGBoost trouvés : {xgb_search.best_params_}")

# Modèle Random Forest (gardé comme élément de vote)
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, min_samples_leaf=10, class_weight='balanced')

# AMÉLIORATION FINALE 3 : Création du Voting Classifier
print("\nCréation du Voting Classifier (modèle Ultime)...")
# On combine notre XGBoost optimisé et le Random Forest
voting_model = VotingClassifier(
    estimators=[('xgb', best_xgb), ('rf', rf_model)],
    voting='soft' # Moyenne des probabilités pour plus de nuance
)
voting_model.fit(X_train, y_train)


# --- 4. Évaluation du Modèle Ultime ---

def evaluate_model(model, model_name, X_test, y_test):
    print(f"\n--- Évaluation du Modèle : {model_name} ---")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Précision : {accuracy:.2%}")
    print(classification_report(y_test, y_pred, target_names=['Nul', 'Victoire Domicile', 'Victoire Extérieur'], zero_division=0))

# Évaluons le modèle optimisé et le modèle de vote final
evaluate_model(best_xgb, "XGBoost Optimisé", X_test, y_test)
evaluate_model(voting_model, "VOTING CLASSIFIER (Modèle Ultime)", X_test, y_test)


# --- 5. Fonction de Prédiction Finale ---

def predict_match_final(home_team, away_team, tournament_name, is_neutral_ground, model):
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

    print(f"\n--- PRÉDICTION FINALE pour {home_team} vs {away_team} ({tournament_name}) ---")
    prediction = model.predict(match_data)[0]
    probabilities = model.predict_proba(match_data)[0]
    result_map = {0: 'Match Nul', 1: f'Victoire de {home_team}', 2: f'Victoire de {away_team}'}
    
    print(f"Résultat le plus probable : {result_map[prediction]}")
    print("Probabilités :")
    print(f"  - Victoire de {home_team}: {probabilities[1]:.2%}")
    print(f"  - Match Nul: {probabilities[0]:.2%}")
    print(f"  - Victoire de {away_team}: {probabilities[2]:.2%}")

# --- Exemple d'utilisation avec le modèle Ultime ---
predict_match_final('France', 'Brazil', 'FIFA World Cup', True, voting_model)
predict_match_final('Spain', 'Germany', 'FIFA World Cup qualification', False, voting_model)
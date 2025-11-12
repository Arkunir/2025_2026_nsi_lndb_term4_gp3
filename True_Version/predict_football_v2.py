import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
# NOUVEAU : Importation de XGBoost
from xgboost import XGBClassifier

# --- 1. Chargement et Préparation des Données ---

# Charger les datasets
print("Chargement des données...")
results_df = pd.read_csv('True_Version/results.csv')
fifa_ranking_df = pd.read_csv('True_Version/fifa_ranking.csv')

# Standardiser les noms des colonnes et des pays pour la fusion
fifa_ranking_df = fifa_ranking_df.rename(columns={'country_full': 'country', 'rank_date': 'date'})
results_df = results_df.rename(columns={'date': 'date'})

# Convertir les dates en format datetime
results_df['date'] = pd.to_datetime(results_df['date'])
fifa_ranking_df['date'] = pd.to_datetime(fifa_ranking_df['date'])

# Fusionner les données de classement FIFA
def get_fifa_rank(team, date):
    team_rankings = fifa_ranking_df[fifa_ranking_df['country'] == team]
    if team_rankings.empty:
        return np.nan
    relevant_rankings = team_rankings[team_rankings['date'] < date]
    if relevant_rankings.empty:
        return np.nan
    last_ranking = relevant_rankings.iloc[-1]
    return last_ranking['rank']

print("Fusion des données de classement FIFA...")
results_df['home_team_rank'] = results_df.apply(lambda row: get_fifa_rank(row['home_team'], row['date']), axis=1)
results_df['away_team_rank'] = results_df.apply(lambda row: get_fifa_rank(row['away_team'], row['date']), axis=1)
results_df.dropna(inplace=True)


# --- 2. Feature Engineering (Création de caractéristiques) ---

# Fonction pour la forme récente (points sur les 5 derniers matchs)
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

# NOUVEAU : Fonction pour la différence de buts moyenne
def calculate_goal_difference(team, date, matches=5):
    team_matches = results_df[((results_df['home_team'] == team) | (results_df['away_team'] == team)) & (results_df['date'] < date)]
    team_matches = team_matches.sort_values(by='date').tail(matches)
    
    goals_scored = 0
    goals_conceded = 0
    
    for _, row in team_matches.iterrows():
        if row['home_team'] == team:
            goals_scored += row['home_score']
            goals_conceded += row['away_score']
        else:
            goals_scored += row['away_score']
            goals_conceded += row['home_score']
            
    if len(team_matches) == 0:
        return 0
    return (goals_scored - goals_conceded) / len(team_matches)

print("Calcul des features de forme et de différence de buts...")
results_df['home_team_form'] = results_df.apply(lambda row: calculate_recent_form(row['home_team'], row['date']), axis=1)
results_df['away_team_form'] = results_df.apply(lambda row: calculate_recent_form(row['away_team'], row['date']), axis=1)
results_df['home_goal_diff'] = results_df.apply(lambda row: calculate_goal_difference(row['home_team'], row['date']), axis=1)
results_df['away_goal_diff'] = results_df.apply(lambda row: calculate_goal_difference(row['away_team'], row['date']), axis=1)

# NOUVEAU : Feature pour l'importance du tournoi
def get_tournament_importance(tournament):
    if 'FIFA World Cup' in tournament: return 4
    if 'UEFA Euro' in tournament or 'Copa América' in tournament: return 3
    if 'Confederations Cup' in tournament or 'FIFA World Cup qualification' in tournament: return 2
    return 1 # Par défaut, les matchs amicaux ou autres

results_df['tournament_importance'] = results_df['tournament'].apply(get_tournament_importance)

# NOUVEAU : Feature pour le terrain neutre
results_df['is_neutral'] = results_df['neutral'].astype(int)

# Définir la variable cible
def get_match_result(row):
    if row['home_score'] > row['away_score']: return 1
    if row['home_score'] < row['away_score']: return 2
    return 0

results_df['result'] = results_df.apply(get_match_result, axis=1)

# NOUVEAU : Sélection des features enrichies
features = [
    'home_team_rank',
    'away_team_rank',
    'home_team_form',
    'away_team_form',
    'home_goal_diff',
    'away_goal_diff',
    'tournament_importance',
    'is_neutral'
]

final_df = results_df[features + ['result']].dropna()

# --- 3. Entraînement des Modèles ---

print("\nPréparation de l'entraînement des modèles...")
X = final_df[features]
y = final_df['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Modèle 1: Random Forest Amélioré
print("Entraînement du modèle Random Forest amélioré...")
# NOUVEAU : class_weight='balanced' pour mieux gérer les matchs nuls
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_leaf=10, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Modèle 2: XGBoost
print("Entraînement du modèle XGBoost...")
# NOUVEAU : Utilisation de use_label_encoder=False et eval_metric pour éviter les warnings
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_estimators=100, learning_rate=0.05)
xgb_model.fit(X_train, y_train)

# --- 4. Évaluation des Modèles ---

def evaluate_model(model, model_name, X_test, y_test):
    print(f"\n--- Évaluation du Modèle : {model_name} ---")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Précision : {accuracy:.2%}")
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred, target_names=['Nul', 'Victoire Domicile', 'Victoire Extérieur']))

evaluate_model(rf_model, "Random Forest Amélioré", X_test, y_test)
evaluate_model(xgb_model, "XGBoost", X_test, y_test)


# --- 5. Fonction de Prédiction Améliorée ---

def predict_match_improved(home_team, away_team, tournament_name, is_neutral_ground, models):
    """
    Prédit le résultat d'un match en utilisant les modèles améliorés.
    """
    today = pd.to_datetime('today')
    
    # Récupérer toutes les features
    home_rank = get_fifa_rank(home_team, today)
    away_rank = get_fifa_rank(away_team, today)
    home_form = calculate_recent_form(home_team, today)
    away_form = calculate_recent_form(away_team, today)
    home_gd = calculate_goal_difference(home_team, today)
    away_gd = calculate_goal_difference(away_team, today)
    tournament_imp = get_tournament_importance(tournament_name)
    is_neutral = 1 if is_neutral_ground else 0

    if pd.isna(home_rank) or pd.isna(away_rank):
        return "Erreur : Une des équipes n'est pas dans le classement FIFA."

    match_data = pd.DataFrame({
        'home_team_rank': [home_rank], 'away_team_rank': [away_rank],
        'home_team_form': [home_form], 'away_team_form': [away_form],
        'home_goal_diff': [home_gd], 'away_goal_diff': [away_gd],
        'tournament_importance': [tournament_imp], 'is_neutral': [is_neutral]
    })

    print(f"\n--- Prédiction pour {home_team} vs {away_team} ({tournament_name}) ---")
    print(f"Données : Rang [{home_rank} vs {away_rank}], Forme [{home_form} vs {away_form}], Diff. Buts [{home_gd:.2f} vs {away_gd:.2f}]")

    for name, model in models.items():
        prediction = model.predict(match_data)[0]
        probabilities = model.predict_proba(match_data)[0]
        result_map = {0: 'Match Nul', 1: f'Victoire de {home_team}', 2: f'Victoire de {away_team}'}
        
        print(f"\n--- Prédiction {name} ---")
        print(f"Résultat le plus probable : {result_map[prediction]}")
        print("Probabilités :")
        print(f"  - Victoire de {home_team}: {probabilities[1]:.2%}")
        print(f"  - Match Nul: {probabilities[0]:.2%}")
        print(f"  - Victoire de {away_team}: {probabilities[2]:.2%}")

# --- Exemple d'utilisation ---
models_to_test = {
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

# Testons avec une finale de Coupe du Monde sur terrain neutre
predict_match_improved('France', 'Brazil', 'FIFA World Cup', True, models_to_test)
# Testons avec un match de qualification à domicile
predict_match_improved('Spain', 'Germany', 'FIFA World Cup qualification', False, models_to_test)
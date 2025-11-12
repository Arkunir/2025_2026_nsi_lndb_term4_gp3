import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# --- 1. Chargement et Préparation des Données ---

# Charger les datasets
print("Chargement des données...")
results_df = pd.read_csv('True_Version/results.csv')
fifa_ranking_df = pd.read_csv('True_Version/fifa_ranking.csv')

# Standardiser les noms des colonnes et des pays pour la fusion
# On garde 'rank_date' et 'country_full' du classement FIFA
fifa_ranking_df = fifa_ranking_df.rename(columns={'country_full': 'country', 'rank_date': 'date'})
# On garde 'date', 'home_team', 'away_team' des résultats
results_df = results_df.rename(columns={'date': 'date'})

# Convertir les dates en format datetime pour pouvoir les manipuler
results_df['date'] = pd.to_datetime(results_df['date'])
fifa_ranking_df['date'] = pd.to_datetime(fifa_ranking_df['date'])

# Fusionner les données de classement FIFA pour les équipes à domicile et à l'extérieur
# On récupère le classement le plus récent AVANT la date du match
def get_fifa_rank(team, date):
    team_rankings = fifa_ranking_df[fifa_ranking_df['country'] == team]
    if team_rankings.empty:
        return np.nan # Retourne NaN si le pays n'est pas dans le classement FIFA
    # Trouver le classement le plus proche avant la date du match
    relevant_rankings = team_rankings[team_rankings['date'] < date]
    if relevant_rankings.empty:
        return np.nan
    last_ranking = relevant_rankings.iloc[-1]
    return last_ranking['rank']

print("Fusion des données de classement FIFA...")
# Appliquer la fonction pour obtenir le rang de chaque équipe
results_df['home_team_rank'] = results_df.apply(lambda row: get_fifa_rank(row['home_team'], row['date']), axis=1)
results_df['away_team_rank'] = results_df.apply(lambda row: get_fifa_rank(row['away_team'], row['date']), axis=1)

# Supprimer les lignes où le classement FIFA n'est pas disponible
results_df.dropna(inplace=True)

# --- 2. Feature Engineering (Création de caractéristiques) ---

# Calculer la forme récente (points sur les 5 derniers matchs)
def calculate_recent_form(team, date, matches=5):
    team_matches = results_df[((results_df['home_team'] == team) | (results_df['away_team'] == team)) & (results_df['date'] < date)]
    team_matches = team_matches.sort_values(by='date').tail(matches)
    
    points = 0
    for _, row in team_matches.iterrows():
        if row['home_team'] == team:
            if row['home_score'] > row['away_score']:
                points += 3
            elif row['home_score'] == row['away_score']:
                points += 1
        else: # away team
            if row['away_score'] > row['home_score']:
                points += 3
            elif row['away_score'] == row['home_score']:
                points += 1
    return points

print("Calcul de la forme récente des équipes...")
results_df['home_team_form'] = results_df.apply(lambda row: calculate_recent_form(row['home_team'], row['date']), axis=1)
results_df['away_team_form'] = results_df.apply(lambda row: calculate_recent_form(row['away_team'], row['date']), axis=1)

# Définir la variable cible (le résultat du match)
# 1 = Victoire à domicile, 0 = Nul, 2 = Victoire à l'extérieur
def get_match_result(row):
    if row['home_score'] > row['away_score']:
        return 1
    elif row['home_score'] < row['away_score']:
        return 2
    else:
        return 0

results_df['result'] = results_df.apply(get_match_result, axis=1)

# Sélectionner les features (caractéristiques) pour le modèle
features = [
    'home_team_rank',
    'away_team_rank',
    'home_team_form',
    'away_team_form'
]

# Créer le dataset final
final_df = results_df[features + ['result']].dropna()

# --- 3. Entraînement du Modèle ---

print("Préparation de l'entraînement du modèle...")
X = final_df[features]
y = final_df['result']

# Diviser les données en ensemble d'entraînement (80%) et de test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle Random Forest
print("Entraînement du modèle Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_leaf=10)
model.fit(X_train, y_train)

# --- 4. Évaluation du Modèle ---

print("\n--- Évaluation du Modèle ---")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Précision du modèle sur l'ensemble de test : {accuracy:.2%}")
print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=['Nul', 'Victoire Domicile', 'Victoire Extérieur']))

# --- 5. Fonction de Prédiction pour un Nouveau Match ---

def predict_match(home_team, away_team, model, data):
    """
    Prédit le résultat d'un match entre deux équipes.
    """
    # Obtenir la date actuelle pour le calcul du classement et de la forme
    today = pd.to_datetime('today')
    
    # Récupérer les features pour chaque équipe
    home_rank = get_fifa_rank(home_team, today)
    away_rank = get_fifa_rank(away_team, today)
    home_form = calculate_recent_form(home_team, today)
    away_form = calculate_recent_form(away_team, today)

    if pd.isna(home_rank) or pd.isna(away_rank):
        return "Erreur : Une des équipes n'est pas dans le classement FIFA ou n'a pas d'historique récent."

    # Créer le DataFrame pour la prédiction
    match_data = pd.DataFrame({
        'home_team_rank': [home_rank],
        'away_team_rank': [away_rank],
        'home_team_form': [home_form],
        'away_team_form': [away_form]
    })

    # Faire la prédiction
    prediction = model.predict(match_data)[0]
    probabilities = model.predict_proba(match_data)[0]

    # Interpréter le résultat
    result_map = {0: 'Match Nul', 1: f'Victoire de {home_team}', 2: f'Victoire de {away_team}'}
    
    print(f"\n--- Prédiction pour {home_team} vs {away_team} ---")
    print(f"Classement FIFA: {home_team} ({home_rank}) vs {away_team} ({away_rank})")
    print(f"Forme récente (sur 5 matchs): {home_team} ({home_form} pts) vs {away_team} ({away_form} pts)")
    print(f"\nRésultat le plus probable : {result_map[prediction]}")
    print("\nProbabilités :")
    print(f"  - Victoire de {home_team}: {probabilities[1]:.2%}")
    print(f"  - Match Nul: {probabilities[0]:.2%}")
    print(f"  - Victoire de {away_team}: {probabilities[2]:.2%}")

# --- Exemple d'utilisation ---
# Testons avec un match fictif entre la France et le Brésil
predict_match('France', 'Brazil', model, results_df)
predict_match('Spain', 'Germany', model, results_df)
predict_match('Argentina', 'Portugal', model, results_df)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

print("Calcul de toutes les features...")
results_df['home_team_form'] = results_df.apply(lambda row: calculate_recent_form(row['home_team'], row['date']), axis=1)
results_df['away_team_form'] = results_df.apply(lambda row: calculate_recent_form(row['away_team'], row['date']), axis=1)
results_df['home_goal_diff'] = results_df.apply(lambda row: calculate_goal_difference(row['home_team'], row['date']), axis=1)
results_df['away_goal_diff'] = results_df.apply(lambda row: calculate_goal_difference(row['away_team'], row['date']), axis=1)
results_df['h2h_points_diff'] = results_df.apply(lambda row: get_h2h_points_diff(row['home_team'], row['away_team'], row['date']), axis=1)
results_df['tournament_importance'] = results_df['tournament'].apply(lambda t: 4 if 'FIFA World Cup' in t else (3 if 'UEFA Euro' in t or 'Copa América' in t else (2 if 'qualification' in t else 1)))
results_df['is_neutral'] = results_df['neutral'].astype(int)

# --- 3. Préparation pour l'Architecture en Deux Étapes ---

# Définir les features et les cibles
features = [
    'home_team_rank', 'away_team_rank', 'home_team_form', 'away_team_form',
    'home_goal_diff', 'away_goal_diff', 'tournament_importance', 'is_neutral',
    'h2h_points_diff'
]

# Cible pour l'étape 1 : Est-ce un nul ? (1 pour Nul, 0 pour Non-Nul)
results_df['is_draw'] = (results_df['result'] == 0).astype(int)
# Cible pour l'étape 2 : Qui gagne ? (1 pour Victoire Domicile, 2 pour Victoire Extérieur)
results_df['winner'] = results_df['result'].replace({0: np.nan}).astype('Int64') # On remplace les nuls par NaN pour les exclure

final_df = results_df[features + ['is_draw', 'winner', 'result']].dropna(subset=['is_draw', 'winner'])

# --- 4. Entraînement des Modèles ---

# Séparation des données d'entraînement et de test (commune aux deux étapes)
X = final_df[features]
y_draw = final_df['is_draw']
y_winner = final_df['winner']

X_train, X_test, y_draw_train, y_draw_test, y_winner_train, y_winner_test, y_result_test = train_test_split(
    X, y_draw, y_winner, final_df['result'], test_size=0.2, random_state=42, stratify=final_df['result']
)

# Étape 1 : Entraînement du "Détecteur de Nul"
print("\n--- Entraînement de l'Étape 1 : Détecteur de Nul ---")
draw_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
draw_classifier.fit(X_train, y_draw_train)

# Étape 2 : Entraînement du "Prédicteur de Vainqueur"
# On n'entraîne que sur les matchs qui ne sont PAS des nuls
print("--- Entraînement de l'Étape 2 : Prédicteur de Vainqueur ---")
non_draw_mask_train = y_winner_train.notna()
X_train_non_draw = X_train[non_draw_mask_train]
y_winner_train_filtered = y_winner_train[non_draw_mask_train]

# On utilise XGBoost, qui est excellent pour cette tâche binaire
winner_classifier = XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)
# On ajuste les labels pour XGBoost (0 et 1 au lieu de 1 et 2)
y_winner_train_filtered_adjusted = y_winner_train_filtered.replace({1: 0, 2: 1})
winner_classifier.fit(X_train_non_draw, y_winner_train_filtered_adjusted)


# --- 5. Évaluation du Modèle Complet ---

def evaluate_two_stage_model(X_test, y_result_test, draw_clf, winner_clf):
    print("\n--- Évaluation du Modèle Complet à Deux Étapes ---")
    
    # Prédiction de l'étape 1 (Nul ou Non-Nul)
    is_draw_pred = draw_clf.predict(X_test)
    
    # Prédiction de l'étape 2 (Vainqueur)
    # On ne lance la prédiction du vainqueur que là où le modèle 1 a prédit "Non-Nul"
    non_draw_pred_mask = (is_draw_pred == 0)
    X_test_non_draw = X_test[non_draw_pred_mask]
    
    final_predictions = np.full(len(X_test), -1, dtype=int) # -1 comme placeholder

    if not X_test_non_draw.empty:
        winner_pred_adjusted = winner_clf.predict(X_test_non_draw)
        # Reconvertir les labels (0 et 1 vers 1 et 2)
        winner_pred = np.where(winner_pred_adjusted == 0, 1, 2)
        
        # Assemblage des prédictions finales
        final_predictions[non_draw_pred_mask] = winner_pred
    
    # Les matchs prédits comme "Nul" ont la prédiction finale 0
    final_predictions[is_draw_pred == 1] = 0
    
    # Calcul de la précision globale
    accuracy = accuracy_score(y_result_test, final_predictions)
    print(f"Précision globale du modèle à deux étapes : {accuracy:.2%}")
    
    # Rapport de classification détaillé
    print("\nRapport de classification final :")
    print(classification_report(y_result_test, final_predictions, target_names=['Nul', 'Victoire Domicile', 'Victoire Extérieur'], zero_division=0))

evaluate_two_stage_model(X_test, y_result_test, draw_classifier, winner_classifier)


# --- 6. Fonction de Prédiction Finale (Deux Étapes) ---

def predict_match_two_stages(home_team, away_team, tournament_name, is_neutral_ground, draw_clf, winner_clf):
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

    print(f"\n--- PRÉDICTION À DEUX ÉTAPES pour {home_team} vs {away_team} ---")
    
    # Étape 1 : Est-ce un nul ?
    is_draw_prob = draw_clf.predict_proba(match_data)[0]
    print(f"Probabilité de Nul (Étape 1) : {is_draw_prob[1]:.2%}")
    
    if is_draw_prob[1] > 0.40: # Seuil de confiance pour le nul
        print("Conclusion : Le modèle estime qu'il y a une forte chance de match nul.")
        print("Résultat le plus probable : Match Nul")
    else:
        print("Conclusion : Le modèle estime qu'il y aura un vainqueur. Lancement de l'Étape 2...")
        # Étape 2 : Qui gagne ?
        winner_pred_adjusted = winner_clf.predict(match_data)[0]
        winner_prob = winner_clf.predict_proba(match_data)[0]
        
        result_map = {0: f'Victoire de {home_team}', 1: f'Victoire de {away_team}'}
        
        print(f"Résultat le plus probable : {result_map[winner_pred_adjusted]}")
        print("Probabilités (Étape 2) :")
        print(f"  - Victoire de {home_team}: {winner_prob[0]:.2%}")
        print(f"  - Victoire de {away_team}: {winner_prob[1]:.2%}")


# --- Exemple d'utilisation ---
predict_match_two_stages('France', 'Brazil', 'FIFA World Cup', True, draw_classifier, winner_classifier)
predict_match_two_stages('Spain', 'Germany', 'FIFA World Cup qualification', False, draw_classifier, winner_classifier)
predict_match_two_stages('Italy', 'England', 'UEFA Euro', True, draw_classifier, winner_classifier)
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

# --- 1. Chargement des Données Prétraitées (Identique) ---
try:
    final_df = pd.read_csv('donnees_pretraitees_football_v7.csv')
    print("Fichier 'donnees_pretraitees_football_v7.csv' chargé avec succès.")
except FileNotFoundError:
    print("ERREUR : Le fichier 'donnees_pretraitees_football_v7.csv' n'a pas été trouvé.")
    exit()

features_v7 = [
    'home_team_rank', 'away_team_rank', 'home_team_form', 'away_team_form',
    'home_goal_diff', 'away_goal_diff', 'tournament_importance', 'is_neutral',
    'h2h_points_diff'
]

X = final_df[features_v7]
y = final_df['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# --- 2. La Compétition à Objectif (Corrigé) ---

# NOUVEAU : Définir l'objectif et la limite de sécurité
TARGET_ACCURACY = 0.58 # J'ai remis 58% comme dans votre exemple
MAX_TRIALS = 2000  # Limite de sécurité pour éviter une boucle infinie

def objective_focused(trial):
    """
    Cette fonction ne fait concourir que XGBoost avec des plages de paramètres très ciblées.
    """
    params = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 500, 900),
        'max_depth': trial.suggest_int('max_depth',3, 4),
        'learning_rate': trial.suggest_float('learning_rate', 0.008, 0.015, log=True),
        'subsample': trial.suggest_float('subsample', 0.60, 0.75),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.90, 0.99),
        'gamma': trial.suggest_float('gamma', 0.05, 0.2),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 0.2),
        'reg_lambda': trial.suggest_float('reg_lambda', 3.5, 5.5)
    }
    
    classifier = XGBClassifier(**params)
    score = cross_val_score(classifier, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1).mean()
    
    return score

# --- Lancement de la Compétition à Objectif ---
print(f"\nDÉBUT DE LA COMPÉTITION... Objectif : {TARGET_ACCURACY:.2%} de précision.")

# On crée une nouvelle étude Optuna pour maximiser la précision
study_objective = optuna.create_study(direction='maximize')

# NOUVEAU : Boucle while True avec des conditions de sortie à l'intérieur
trial_count = 0
while True:
    # On lance un seul essai à la fois
    study_objective.optimize(objective_focused, n_trials=1)
    trial_count += 1
    
    # On récupère le meilleur score actuel
    current_best_value = study_objective.best_value
    
    # On affiche la progression
    print(f"Essai n°{trial_count}: Meilleur score actuel = {current_best_value:.4f}")
    
    # NOUVEAU : On vérifie les conditions de sortie ici
    if current_best_value >= TARGET_ACCURACY:
        print(f"\nSUCCÈS ! Objectif de {TARGET_ACCURACY:.2%} atteint en {trial_count} essais !")
        break
        
    if trial_count >= MAX_TRIALS:
        print(f"\nÉCHEC. L'objectif de {TARGET_ACCURACY:.2%} n'a pas été atteint après {MAX_TRIALS} essais.")
        break

# --- Résultats du Champion ---
print("\n--- LE MODÈLE CHAMPION ---")
best_params_objective = study_objective.best_params
print("Meilleure configuration trouvée :")
for key, value in best_params_objective.items():
    print(f"  - {key}: {value}")

# On entraîne le modèle champion sur toutes les données d'entraînement
champion_model_objective = XGBClassifier(**best_params_objective)

print("\nEntraînement du modèle champion sur le jeu de données complet...")
champion_model_objective.fit(X_train, y_train)

# --- Évaluation Finale sur le Jeu de Test ---
print("\n--- ÉVALUATION FINALE DU CHAMPION ---")
y_pred = champion_model_objective.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)
print(f"Précision finale du modèle champion sur le jeu de test : {final_accuracy:.2%}")
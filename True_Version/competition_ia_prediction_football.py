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


# --- 2. La Compétition Affinée sur le Champion (XGBoost) ---

def objective_focused(trial):
    """
    Cette fonction ne fait concourir que XGBoost avec des plages de paramètres très ciblées.
    """
    
    # NOUVEAU : On ne suggère plus le type de classifieur, on est concentré sur XGBoost
    
    # NOUVEAU : Plages de paramètres plus serrées autour du meilleur résultat précédent
    # Meilleurs params précédents : n_estimators=668, max_depth=3, learning_rate=0.010...
    params = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'random_state': 42,
        
        # On explore autour de 668
        'n_estimators': trial.suggest_int('n_estimators', 500, 800),
        
        # On teste 3 et 4, car 3 était le meilleur
        'max_depth': trial.suggest_int('max_depth', 3, 4),
        
        # On explore autour de 0.01
        'learning_rate': trial.suggest_float('learning_rate', 0.008, 0.015, log=True),
        
        # On explore autour de 0.66
        'subsample': trial.suggest_float('subsample', 0.60, 0.75),
        
        # On explore autour de 0.94
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.90, 0.99),
        
        # On explore autour de 0.12
        'gamma': trial.suggest_float('gamma', 0.05, 0.2),
        
        # On explore autour de 0.08
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 0.2),
        
        # On explore autour de 4.6
        'reg_lambda': trial.suggest_float('reg_lambda', 3.5, 5.5)
    }
    
    classifier = XGBClassifier(**params)

    # On utilise une validation croisée pour un score robuste
    score = cross_val_score(classifier, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1).mean()
    
    return score

# --- Lancement de la Compétition Affinée ---
print("\nDÉBUT DE LA COMPÉTITION AFFINÉE... (Zoom sur le champion XGBoost)")

# On crée une nouvelle étude Optuna pour maximiser la précision
study_focused = optuna.create_study(direction='maximize')
# On lance plus de "manches" pour explorer cet espace plus petit en détail
study_focused.optimize(objective_focused, n_trials=200)

print("\n--- COMPÉTITION AFFINÉE TERMINÉE ---")
print(f"Nombre de manches jouées : {len(study_focused.trials)}")
print(f"Meilleur score (précision) obtenu : {study_focused.best_value:.4f}")

# --- Résultats du Nouveau Champion ---
print("\n--- LE NOUVEAU MODÈLE CHAMPION ---")
best_params_focused = study_focused.best_params
print("Meilleure configuration affinée trouvée :")
for key, value in best_params_focused.items():
    print(f"  - {key}: {value}")

# On entraîne le nouveau modèle champion sur toutes les données d'entraînement
champion_model_v2 = XGBClassifier(**best_params_focused)

print("\nEntraînement du nouveau modèle champion sur le jeu de données complet...")
champion_model_v2.fit(X_train, y_train)

# --- Évaluation Finale sur le Jeu de Test ---
print("\n--- ÉVALUATION FINALE DU NOUVEAU CHAMPION ---")
y_pred = champion_model_v2.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)
print(f"Précision finale du nouveau modèle champion sur le jeu de test : {final_accuracy:.2%}")

# Comparaison avec l'ancien champion
ancienne_precision = 0.5778 # 57.78%
print(f"\nAncienne précision du champion : {ancienne_precision:.2%}")
if final_accuracy > ancienne_precision:
    print("SUCCÈS ! Nous avons amélioré le modèle.")
else:
    print("L'amélioration n'a pas été significative, mais nous avons confirmé la robustesse du modèle.")

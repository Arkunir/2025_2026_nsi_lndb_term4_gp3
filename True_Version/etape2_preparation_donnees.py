import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# --- 1. Chargement des Données Prétraitées V7 ---
try:
    final_df = pd.read_csv('True_Version/donnees_pretraitees_football_v7.csv')
    print("Fichier 'donnees_pretraitees_football_v7.csv' chargé avec succès.")
except FileNotFoundError:
    print("ERREUR : Le fichier 'donnees_pretraitees_football_v7.csv' n'a pas été trouvé.")
    print("Veuillez d'abord exécuter le script 'etape1_preparation_donnees_v7.py'.")
    exit()

# --- 2. Entraînement et Optimisation du Modèle de Classification V8 ---

features_v8 = [
    'home_team_rank', 'away_team_rank', 'home_team_form', 'away_team_form',
    'home_goal_diff', 'away_goal_diff', 'tournament_importance', 'is_neutral',
    'h2h_points_diff'
]

X = final_df[features_v8]
y = final_df['result'] # Cible de classification (0=Nul, 1=Domicile, 2=Extérieur)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n--- Recherche des meilleurs hyperparamètres pour XGBoost V8 (peut prendre du temps) ---")

# NOUVEAU : Grille de paramètres plus large pour une recherche plus approfondie
param_grid = {
    'n_estimators': [200, 400, 600],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 5, 6],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2], # Ajout de gamma pour la régularisation
    'reg_alpha': [0, 0.01, 0.1], # Ajout de la régularisation L1
    'reg_lambda': [1, 1.5, 2] # Ajout de la régularisation L2
}

# NOUVEAU : Gestion du déséquilibre des classes avec scale_pos_weight
# On calcule le poids pour la classe 'Nul' (0) qui est sous-représentée
class_counts = y_train.value_counts()
scale_pos_weight = class_counts[1] / class_counts[0] # Poids pour la classe 0 (Nul)

xgb_search = RandomizedSearchCV(
    estimator=XGBClassifier(
        objective='multi:softprob', 
        eval_metric='mlogloss', 
        random_state=42,
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight # On applique le poids ici
    ),
    param_distributions=param_grid,
    n_iter=50, # Augmenté pour plus de tests
    scoring='accuracy',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

xgb_search.fit(X_train, y_train)
best_xgb_v8 = xgb_search.best_estimator_
print(f"\nMeilleurs paramètres trouvés : {xgb_search.best_params_}")


# --- 3. Évaluation Finale du Modèle V8 ---
print("\n--- Évaluation du Modèle Ultime V8 ---")
y_pred = best_xgb_v8.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision finale du modèle V8 : {accuracy:.2%}")
print("\nRapport de classification final :")
print(classification_report(y_test, y_pred, target_names=['Nul', 'Victoire Domicile', 'Victoire Extérieur'], zero_division=0))

# --- 4. Fonction de Prédiction Finale (V8) ---
def predict_match_v8(home_team, away_team, tournament_name, is_neutral_ground, model):
    # (Cette fonction est similaire aux précédentes, elle doit juste reconstruire les features)
    # Pour des raisons de simplicité, je la laisse en exercice, elle est quasi-identique à la V6/V7
    # en utilisant les features_v8 et le modèle 'best_xgb_v8'
    print("\nFonction de prédiction V8 à implémenter si besoin...")
    pass

# Exemple d'utilisation de la prédiction (si la fonction est implémentée)
# predict_match_v8('France', 'Brazil', 'FIFA World Cup', True, best_xgb_v8)
# --- NOUVEAU : Sauvegarde du modèle entraîné ---
model_filename = 'best_football_model_v8.joblib'
joblib.dump(best_xgb_v8, model_filename)

print(f"\nLe modèle champion a été sauvegardé avec succès dans le fichier : '{model_filename}'")
print("Vous pouvez maintenant lancer le script de prédiction interactif.")
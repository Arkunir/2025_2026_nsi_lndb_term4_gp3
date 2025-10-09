import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report

# ============================================
# 1. Charger le modèle et le scaler
# ============================================
model = joblib.load("football_nn_model.pkl")
scaler = joblib.load("football_scaler.pkl")

# ============================================
# 2. Charger le mapping des équipes
# ============================================
train_df = pd.read_csv("matches_RL_ready.csv")
teams = list(set(train_df["home_team"]).union(set(train_df["away_team"])))
team_to_id = {team: i for i, team in enumerate(teams)}

# ============================================
# 3. Lire le fichier tabulé des matchs récents
# ============================================
file_path = "100_Recents_Matches.csv"
df = pd.read_csv(file_path, sep="\t", header=None)
df.columns = ["date", "match", "score"]

# Séparer home et away
df[["home_team", "away_team"]] = df["match"].str.split(" vs. ", expand=True)

# Séparer les scores
df[["home_score", "away_score"]] = df["score"].str.replace("–","-").str.split("-", expand=True)
df["home_score"] = df["home_score"].astype(int)
df["away_score"] = df["away_score"].astype(int)

# Garder les colonnes nécessaires
df = df[["date", "home_team", "away_team", "home_score", "away_score"]]

# ============================================
# 4. Filtrer les équipes connues du modèle
# ============================================
df = df[df["home_team"].isin(team_to_id) & df["away_team"].isin(team_to_id)].copy()
df["home_id"] = df["home_team"].map(team_to_id)
df["away_id"] = df["away_team"].map(team_to_id)

# ============================================
# 5. Calculer le résultat réel
# ============================================
def get_result(row):
    if row["home_score"] > row["away_score"]:
        return 0  # HomeWin
    elif row["home_score"] < row["away_score"]:
        return 2  # AwayWin
    else:
        return 1  # Draw

df["true_result"] = df.apply(get_result, axis=1)

# ============================================
# 6. Préparer les features pour le modèle
# ============================================
df["goal_diff"] = df["home_score"] - df["away_score"]
df["home_avg_goals"] = 1.5  # valeur moyenne fictive
df["away_avg_goals"] = 1.2  # valeur moyenne fictive

features = ["home_id", "away_id", "goal_diff", "home_avg_goals", "away_avg_goals"]
X_test = df[features]
X_test_scaled = scaler.transform(X_test)

# ============================================
# 7. Prédictions
# ============================================
y_pred = model.predict(X_test_scaled)
y_true = df["true_result"]

# ============================================
# 8. Évaluation
# ============================================
accuracy = accuracy_score(y_true, y_pred)
print(f"📊 Nombre de matchs testés : {len(y_true)}")
print(f"📊 Taux de précision : {accuracy*100:.2f}%")
print("\nDétails par classe (0=HomeWin, 1=Draw, 2=AwayWin):")
print(classification_report(y_true, y_pred, digits=3))

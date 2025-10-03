import pandas as pd
import numpy as np
from stable_baselines3 import DQN

# Charger le modèle entraîné
model = DQN.load("football_rl_agent")

# =========================
# 1. Charger et préparer les données de test
# =========================
file_path = "100_Recents_Matches.csv"   # <--- fichier dans la racine

# Lecture brute du fichier (il n’est pas en CSV standard, donc parsing manuel)
matches = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            date, match, score = line.split("\t")
            home, away = match.replace(" vs. ", "\t").split("\t")
            home_score, away_score = score.replace("–", "-").split("-")
            matches.append([date, home.strip(), away.strip(),
                            int(home_score.strip()), int(away_score.strip())])
        except Exception as e:
            print("Erreur parsing ligne :", line, e)

df_test = pd.DataFrame(matches, columns=["date", "home_team", "away_team", "home_score", "away_score"])

# Créer la colonne "resultat" (comme dans le training set)
def get_result(row):
    if row["home_score"] > row["away_score"]:
        return 0  # HomeWin
    elif row["home_score"] < row["away_score"]:
        return 2  # AwayWin
    else:
        return 1  # Draw

df_test["true_result"] = df_test.apply(get_result, axis=1)

# =========================
# 2. Mapping des équipes
# =========================
# Charger le mapping original des équipes utilisé pour l'entraînement
train_df = pd.read_csv("matches_RL_ready.csv")
teams = list(set(train_df["home_team"]).union(set(train_df["away_team"])))
team_to_id = {team: i for i, team in enumerate(teams)}

# Ajouter les colonnes encodées (attention aux équipes inconnues)
df_test["home_id"] = df_test["home_team"].map(lambda t: team_to_id.get(t, -1))
df_test["away_id"] = df_test["away_team"].map(lambda t: team_to_id.get(t, -1))

# On ne garde que les matchs avec des équipes connues du modèle
df_test = df_test[(df_test["home_id"] >= 0) & (df_test["away_id"] >= 0)].reset_index(drop=True)

# =========================
# 3. Prédictions du modèle
# =========================
y_true = []
y_pred = []

for _, row in df_test.iterrows():
    obs = np.array([row["home_id"], row["away_id"]], dtype=np.int32)
    action, _ = model.predict(obs, deterministic=True)
    y_true.append(row["true_result"])
    y_pred.append(action)

# =========================
# 4. Évaluation
# =========================
accuracy = np.mean(np.array(y_true) == np.array(y_pred))

print("📊 Résultats du test :")
print(f"Nombre de matchs testés : {len(y_true)}")
print(f"Taux de précision       : {accuracy*100:.2f}%")

# Facultatif : affichage par catégories
from sklearn.metrics import classification_report
print("\nDétails par classe (0=HomeWin, 1=Draw, 2=AwayWin):")
print(classification_report(y_true, y_pred, digits=3))

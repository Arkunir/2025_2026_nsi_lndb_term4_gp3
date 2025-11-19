import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# ============================================
# 1. Charger et préparer les données
# ============================================
df = pd.read_csv("matches_RL_ready.csv")

# Encodage des équipes
teams = list(set(df["home_team"]).union(set(df["away_team"])))
team_to_id = {team: i for i, team in enumerate(teams)}

df["home_id"] = df["home_team"].map(team_to_id)
df["away_id"] = df["away_team"].map(team_to_id)

# Mapping du résultat
result_map = {"HomeWin": 0, "Draw": 1, "AwayWin": 2}
df["result_id"] = df["resultat"].map(result_map)

# ============================================
# 2. Création de features supplémentaires
# ============================================

# Buts marqués à domicile et à l'extérieur
df["home_goals"] = df["home_score"]
df["away_goals"] = df["away_score"]

# Différence de buts
df["goal_diff"] = df["home_goals"] - df["away_goals"]

# Moyennes mobiles des 5 derniers matchs pour chaque équipe
def rolling_stats(df, team_col, goals_col, prefix):
    team_stats = (
        df.groupby(team_col)[goals_col]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df[f"{prefix}_avg_goals"] = team_stats
    return df

df = rolling_stats(df, "home_team", "home_goals", "home")
df = rolling_stats(df, "away_team", "away_goals", "away")

# ============================================
# 3. Préparation des données d'entraînement
# ============================================
features = [
    "home_id",
    "away_id",
    "goal_diff",
    "home_avg_goals",
    "away_avg_goals",
]

X = df[features].fillna(0)
y = df["result_id"]

# Séparation train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 4. Entraînement du réseau de neurones
# ============================================
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42,
)

model.fit(X_train_scaled, y_train)

# ============================================
# 5. Évaluation
# ============================================
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print("📊 Évaluation du modèle :")
print(f"Précision globale : {acc * 100:.2f}%")
print("\nDétails par classe (0=HomeWin, 1=Draw, 2=AwayWin):")
print(classification_report(y_test, y_pred, digits=3))

# ============================================
# 6. Sauvegarde du modèle et du scaler
# ============================================
import joblib

joblib.dump(model, "football_nn_model.pkl")
joblib.dump(scaler, "football_scaler.pkl")
print("\n✅ Modèle et scaler sauvegardés !")

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import torch
import tkinter as tk
from tkinter import messagebox

# ---------------- Charger le modèle ----------------
MODEL_PATH = "ppo_football.zip"
model = PPO.load(MODEL_PATH)

# ---------------- Reimporter tes fonctions ----------------
from main import compute_elo, add_features  # Ton script d'entraînement doit être dans le même dossier

# ---------------- Fonction pour préparer les features d’un nouveau match ----------------
def build_features_for_match(df, home_team, away_team, feature_cols, date=None):
    if date is not None:
        df = df[df["date"] < pd.to_datetime(date)]

    df = compute_elo(df)
    df = add_features(df)

    # teams = set(df["home_team"]).union(set(df["away_team"]))
    # if home_team not in teams or away_team not in teams:
    #     raise ValueError(f"Les équipes {home_team} et/ou {away_team} sont invalides ou absentes des données.")


    # Filtrer uniquement les matchs entre ces 2 équipes ou impliquant au moins une des deux
    sub_df = df[(df["home_team"] == home_team) | (df["away_team"] == home_team) |
                (df["home_team"] == away_team) | (df["away_team"] == away_team)]

    if sub_df.empty:
        raise ValueError(f"Aucun historique trouvé pour {home_team} ou {away_team}")

    # On prend le dernier match historique impliquant ces équipes
    last_row = sub_df.iloc[-1]

    # Construire un vecteur de features
    match_features = last_row[feature_cols].values.astype(np.float32)

    # Nettoyer les NaN et inf
    match_features = np.nan_to_num(match_features, nan=0.0, posinf=0.0, neginf=0.0)

    return match_features.reshape(1, -1)

# ---------------- Fonction pour prédire ----------------
def predict_match_probability(model, features):
    obs = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs)
        logits = dist.distribution.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return {
        "home_win": float(probs[0]),
        "draw": float(probs[1]),
        "away_win": float(probs[2])
    }

# ---------------- Interface Tkinter ----------------
def run_gui():
    df = pd.read_csv("international_football_results_1872_2017_combined.csv", parse_dates=["date"])
    df = df.rename(columns={"home_score": "home_goals", "away_score": "away_goals"})
    df = df.sort_values("date").reset_index(drop=True)

    feature_cols = [
        'elo_home_pre','elo_away_pre','home_form_pts','away_form_pts',
        'h2h_home_wins','h2h_away_wins',
        'home_goal_diff_avg','away_goal_diff_avg',
        'home_win_rate','away_win_rate'
    ]

    def on_predict():
        home_team = entry_home.get().strip()
        away_team = entry_away.get().strip()
        if not home_team or not away_team:
            messagebox.showwarning("Entrée manquante", "Merci de remplir les deux équipes")
            return
        try:
            features = build_features_for_match(df, home_team, away_team, feature_cols, date="2025-09-10")
            probs = predict_match_probability(model, features)
            result_text = (
                f"Victoire {home_team} : {probs['home_win']:.2%}\n"
                f"Nul : {probs['draw']:.2%}\n"
                f"Victoire {away_team} : {probs['away_win']:.2%}"
            )
            messagebox.showinfo("Résultat de la prédiction", result_text)
        except Exception as e:
            messagebox.showerror("Erreur", str(e))

    # Fenêtre principale
    root = tk.Tk()
    root.title("Prédiction de match de football")

    tk.Label(root, text="Équipe à domicile :").pack(pady=5)
    entry_home = tk.Entry(root, width=30)
    entry_home.pack(pady=5)

    tk.Label(root, text="Équipe à l'extérieur :").pack(pady=5)
    entry_away = tk.Entry(root, width=30)
    entry_away.pack(pady=5)

    predict_button = tk.Button(root, text="Prédiction", command=on_predict, bg="lightblue", font=("Arial", 12, "bold"))
    predict_button.pack(pady=20)

    root.mainloop()

# ---------------- Lancement ----------------
if __name__ == "__main__":
    run_gui()

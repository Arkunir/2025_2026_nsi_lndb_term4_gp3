import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import torch
import tkinter as tk
from tkinter import messagebox
from scipy.stats import poisson

# ---------------- Charger le modèle ----------------
MODEL_PATH = "ppo_football.zip"
model = PPO.load(MODEL_PATH)

# ---------------- Reimporter les fonctions de features ----------------
from main import compute_elo, add_features  # Ton script d'entraînement doit être dans le même dossier

# ---------------- Fonction pour préparer les features d’un nouveau match ----------------
def build_features_for_match(df, home_team, away_team, feature_cols, date=None):
    if date is not None:
        df = df[df["date"] < pd.to_datetime(date)]

    df = compute_elo(df)
    df = add_features(df)

    sub_df = df[(df["home_team"] == home_team) | (df["away_team"] == home_team) |
                (df["home_team"] == away_team) | (df["away_team"] == away_team)]

    if sub_df.empty:
        raise ValueError(f"Aucun historique trouvé pour {home_team} ou {away_team}")

    last_row = sub_df.iloc[-1]
    match_features = last_row[feature_cols].values.astype(np.float32)
    match_features = np.nan_to_num(match_features, nan=0.0, posinf=0.0, neginf=0.0)
    return match_features.reshape(1, -1)

# ---------------- Fonction pour prédire victoire/nul/défaite ----------------
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

# ---------------- Fonction pour prédire les scores exacts ----------------
def predict_score_distribution(df, home_team, away_team, feature_cols, model, max_goals=5, date=None):
    features = build_features_for_match(df, home_team, away_team, feature_cols, date=date)
    probs = predict_match_probability(model, features)

    h_avg = features[0][feature_cols.index('home_goal_diff_avg')]
    a_avg = features[0][feature_cols.index('away_goal_diff_avg')]

    lambda_home = max(0.1, (h_avg + 1) * probs['home_win'] + 0.5 * probs['draw'])
    lambda_away = max(0.1, (a_avg + 1) * probs['away_win'] + 0.5 * probs['draw'])

    score_probs = {}
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            p = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
            score_probs[f"{i}-{j}"] = round(p*100, 2)

    score_probs = dict(sorted(score_probs.items(), key=lambda x: x[1], reverse=True))
    return score_probs

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
            score_probs = predict_score_distribution(df, home_team, away_team, feature_cols, model, max_goals=5, date="2025-09-10")

            # Affichage global
            result_text = (
                f"Probabilités globales :\n"
                f"Victoire {home_team} : {probs['home_win']:.2%}\n"
                f"Nul : {probs['draw']:.2%}\n"
                f"Victoire {away_team} : {probs['away_win']:.2%}\n\n"
                f"Scores les plus probables :\n"
            )
            top_scores = list(score_probs.items())[:6]
            result_text += "\n".join([f"{s}: {p:.2f}%" for s, p in top_scores])

            messagebox.showinfo("Résultat de la prédiction", result_text)
        except Exception as e:
            messagebox.showerror("Erreur", str(e))

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

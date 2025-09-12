import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import torch
import tkinter as tk
from tkinter import messagebox, ttk
from datetime import datetime

# ---------------- Charger le modèle ----------------
MODEL_PATH = "ppo_football.zip"
model = PPO.load(MODEL_PATH)

# ---------------- Reimporter tes fonctions ----------------
from main import compute_elo, add_features  # Ton script d'entraînement doit être dans le même dossier

# ---------------- Fonction pour préparer les features d'un nouveau match (du premier fichier) ----------------
def build_features_for_match(df, home_team, away_team, feature_cols, date=None):
    if date is not None:
        df = df[df["date"] < pd.to_datetime(date)]

    df = compute_elo(df)
    df = add_features(df)

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

# ---------------- Fonction pour prédire (du premier fichier) ----------------
def predict_match_probability(model, features):
    obs = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs)
        logits = dist.distribution.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    max_prob = float(np.max(probs))
    
    return {
        "home_win": float(probs[0]),
        "draw": float(probs[1]),
        "away_win": float(probs[2]),
        "confidence": max_prob
    }

def get_available_teams(df):
    """Retourne la liste des équipes disponibles"""
    home_teams = set(df['home_team'].unique())
    away_teams = set(df['away_team'].unique())
    return sorted(list(home_teams.union(away_teams)))

# ---------------- Interface Tkinter améliorée (du second fichier) ----------------
def run_gui():
    # Charger les données (comme dans le premier fichier)
    df = pd.read_csv("international_football_results_1872_2017_combined.csv", parse_dates=["date"])
    df = df.rename(columns={"home_score": "home_goals", "away_score": "away_goals"})
    df = df.sort_values("date").reset_index(drop=True)

    feature_cols = [
        'elo_home_pre','elo_away_pre','home_form_pts','away_form_pts',
        'h2h_home_wins','h2h_away_wins',
        'home_goal_diff_avg','away_goal_diff_avg',
        'home_win_rate','away_win_rate'
    ]
    
    available_teams = get_available_teams(df)

    def on_predict():
        home_team = combo_home.get().strip()
        away_team = combo_away.get().strip()
        
        if not home_team or not away_team:
            messagebox.showwarning("Entrée manquante", "Merci de sélectionner les deux équipes")
            return
        
        if home_team == away_team:
            messagebox.showwarning("Équipes identiques", "Les équipes ne peuvent pas être identiques")
            return
        
        if home_team not in available_teams or away_team not in available_teams:
            messagebox.showwarning("Équipe inconnue", "Une ou plusieurs équipes ne sont pas dans la base de données")
            return
        
        try:
            # Afficher la barre de progression
            progress_bar.pack(pady=5)
            root.update()
            
            # Utiliser la fonction du premier fichier (plus robuste)
            features = build_features_for_match(df, home_team, away_team, feature_cols, date="2025-09-10")
            probs = predict_match_probability(model, features)
            
            # Cacher la barre de progression
            progress_bar.pack_forget()
            
            # Déterminer le favori
            max_prob = max(probs['home_win'], probs['draw'], probs['away_win'])
            if probs['home_win'] == max_prob:
                favorite = f"{home_team} (domicile)"
            elif probs['away_win'] == max_prob:
                favorite = f"{away_team} (extérieur)"
            else:
                favorite = "Match nul"
            
            confidence = probs.get('confidence', max_prob)
            
            # Affichage amélioré (du second fichier)
            result_text = (
                f"🏠 Victoire {home_team}: {probs['home_win']:.1%}\n"
                f"🤝 Match nul: {probs['draw']:.1%}\n"
                f"✈️ Victoire {away_team}: {probs['away_win']:.1%}\n\n"
                f"📊 Favori: {favorite}\n"
                f"🎯 Confiance: {confidence:.1%}"
            )
            
            messagebox.showinfo("🔮 Prédiction de match", result_text)
            
        except Exception as e:
            progress_bar.pack_forget()
            messagebox.showerror("Erreur", f"Erreur lors de la prédiction:\n{str(e)}")

    def on_team_selected(event):
        """Met à jour l'autre combo quand une équipe est sélectionnée"""
        pass  # Peut être étendu pour des suggestions intelligentes

    # Interface graphique améliorée (du second fichier)
    root = tk.Tk()
    root.title("⚽ Prédicteur de Matchs de Football - IA")
    root.geometry("500x400")
    root.configure(bg='#f0f8ff')
    
    # Style
    style = ttk.Style()
    style.theme_use('clam')
    
    # Titre
    title_label = tk.Label(
        root, 
        text="⚽ PRÉDICTEUR DE MATCHS ⚽", 
        font=("Arial", 18, "bold"), 
        bg='#f0f8ff', 
        fg='#2c3e50'
    )
    title_label.pack(pady=20)
    
    # Frame principal
    main_frame = tk.Frame(root, bg='#f0f8ff')
    main_frame.pack(expand=True, fill='both', padx=20)
    
    # Équipe à domicile
    home_frame = tk.Frame(main_frame, bg='#f0f8ff')
    home_frame.pack(pady=10, fill='x')
    
    tk.Label(
        home_frame, 
        text="🏠 Équipe à domicile:", 
        font=("Arial", 12, "bold"), 
        bg='#f0f8ff'
    ).pack(anchor='w')
    
    combo_home = ttk.Combobox(
        home_frame, 
        values=available_teams, 
        width=40, 
        font=("Arial", 11)
    )
    combo_home.pack(pady=5, fill='x')
    combo_home.bind('<<ComboboxSelected>>', on_team_selected)
    
    # Équipe à l'extérieur
    away_frame = tk.Frame(main_frame, bg='#f0f8ff')
    away_frame.pack(pady=10, fill='x')
    
    tk.Label(
        away_frame, 
        text="✈️ Équipe à l'extérieur:", 
        font=("Arial", 12, "bold"), 
        bg='#f0f8ff'
    ).pack(anchor='w')
    
    combo_away = ttk.Combobox(
        away_frame, 
        values=available_teams, 
        width=40, 
        font=("Arial", 11)
    )
    combo_away.pack(pady=5, fill='x')
    combo_away.bind('<<ComboboxSelected>>', on_team_selected)
    
    # Bouton de prédiction
    predict_button = tk.Button(
        main_frame, 
        text="🔮 PRÉDIRE LE MATCH", 
        command=on_predict, 
        bg="#3498db", 
        fg="white",
        font=("Arial", 14, "bold"),
        padx=20,
        pady=10,
        relief='raised',
        borderwidth=3
    )
    predict_button.pack(pady=30)
    
    # Barre de progression
    progress_bar = ttk.Progressbar(main_frame, mode='indeterminate')
    
    # Informations
    info_label = tk.Label(
        main_frame,
        text=f"📈 Modèle entraîné sur {len(df)} matchs • {len(available_teams)} équipes disponibles",
        font=("Arial", 9),
        bg='#f0f8ff',
        fg='#7f8c8d'
    )
    info_label.pack(side='bottom', pady=10)
    
    root.mainloop()

# ---------------- Test en ligne de commande ----------------
def predict_command_line(home_team, away_team):
    """Interface en ligne de commande pour les prédictions"""
    df = pd.read_csv("international_football_results_1872_2017_combined.csv", parse_dates=["date"])
    df = df.rename(columns={"home_score": "home_goals", "away_score": "away_goals"})
    
    # Nettoyer les données
    df = df.dropna(subset=['home_team', 'away_team'])
    df = df.sort_values("date").reset_index(drop=True)

    feature_cols = [
        'elo_home_pre','elo_away_pre','home_form_pts','away_form_pts',
        'h2h_home_wins','h2h_away_wins',
        'home_goal_diff_avg','away_goal_diff_avg',
        'home_win_rate','away_win_rate'
    ]
    
    try:
        features = build_features_for_match(df, home_team, away_team, feature_cols, date="2025-09-10")
        probs = predict_match_probability(model, features)
        
        print(f"\n⚽ PRÉDICTION: {home_team} vs {away_team}")
        print("="*50)
        print(f"🏠 Victoire {home_team}: {probs['home_win']:.1%}")
        print(f"🤝 Match nul: {probs['draw']:.1%}")
        print(f"✈️ Victoire {away_team}: {probs['away_win']:.1%}")
        
        max_prob = max(probs['home_win'], probs['draw'], probs['away_win'])
        if probs['home_win'] == max_prob:
            print(f"📊 Favori: {home_team} (domicile)")
        elif probs['away_win'] == max_prob:
            print(f"📊 Favori: {away_team} (extérieur)")
        else:
            print(f"📊 Favori: Match nul")
        
        confidence = probs.get('confidence', max_prob)
        print(f"🎯 Confiance: {confidence:.1%}")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")

# ---------------- Lancement ----------------
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 3:
        # Mode ligne de commande
        predict_command_line(sys.argv[1], sys.argv[2])
    else:
        # Mode interface graphique
        run_gui()
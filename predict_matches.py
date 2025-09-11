import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import torch
import tkinter as tk
from tkinter import messagebox, ttk
from sklearn.preprocessing import StandardScaler
import pickle
from datetime import datetime

# ---------------- Configuration ----------------
MODEL_PATH = "ppo_football.zip"
FEATURES_PATH = "ppo_football_features.pkl"
DATA_PATH = "international_football_results_1872_2017_combined.csv"

# ---------------- Charger le modèle et features ----------------
def load_model_and_features():
    """Charge le modèle pré-entraîné et les features"""
    try:
        print("Chargement du modèle...")
        model = PPO.load(MODEL_PATH)
        
        print("Chargement des features...")
        with open(FEATURES_PATH, "rb") as f:
            feature_cols = pickle.load(f)
            
        print(f"✅ Modèle chargé avec {len(feature_cols)} features")
        print(f"🔧 Features: {feature_cols}")
        return model, feature_cols
        
    except FileNotFoundError as e:
        print(f"❌ ERREUR: Fichier non trouvé - {e}")
        print("🔧 Veuillez d'abord exécuter main.py pour entraîner le modèle")
        return None, None
    except Exception as e:
        print(f"❌ ERREUR lors du chargement: {e}")
        return None, None

# ---------------- Données pré-calculées ----------------
def load_preprocessed_data():
    """Charge les données brutes et les statistiques d'équipes pré-calculées"""
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=['date'], low_memory=False)
        df = df.rename(columns={'home_score': 'home_goals', 'away_score': 'away_goals'})
        df['home_goals'] = pd.to_numeric(df['home_goals'], errors='coerce')
        df['away_goals'] = pd.to_numeric(df['away_goals'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).sort_values('date')
        return df
    except Exception as e:
        print(f"❌ Erreur chargement données: {e}")
        return None

# ---------------- Calculs ELO simplifiés ----------------
def get_current_elo(df, team_name, reference_date=None):
    """Calcule l'ELO actuel d'une équipe de manière simple"""
    if reference_date is None:
        reference_date = datetime.now()
    else:
        reference_date = pd.to_datetime(reference_date)
    
    # ELO de base
    current_elo = 1500
    k = 32
    home_advantage = 65
    
    # Parcourir les matchs de l'équipe chronologiquement
    team_matches = df[
        ((df['home_team'] == team_name) | (df['away_team'] == team_name)) &
        (df['date'] < reference_date)
    ].sort_values('date')
    
    for _, match in team_matches.iterrows():
        if match['home_team'] == team_name:
            # Équipe à domicile
            opponent_elo = 1500  # Simplification
            expected = 1 / (1 + 10 ** (-((current_elo + home_advantage) - opponent_elo) / 400))
            
            if match['home_goals'] > match['away_goals']:
                result = 1.0
            elif match['home_goals'] == match['away_goals']:
                result = 0.5
            else:
                result = 0.0
                
        else:
            # Équipe à l'extérieur
            opponent_elo = 1500  # Simplification
            expected = 1 / (1 + 10 ** (-(current_elo - (opponent_elo + home_advantage)) / 400))
            
            if match['away_goals'] > match['home_goals']:
                result = 1.0
            elif match['away_goals'] == match['home_goals']:
                result = 0.5
            else:
                result = 0.0
        
        # Pondération temporelle simple
        year = match['date'].year
        if year < 2010:
            weight = 0.1
        elif year <= 2022:
            weight = 0.5 + ((year - 2010) / (2022 - 2010)) * 1.0
        else:
            weight = 3.0
            
        current_elo += k * weight * (result - expected)
    
    return current_elo

<<<<<<< HEAD
def get_team_recent_form(df, team_name, reference_date=None, window=8):
    """Calcule la forme récente d'une équipe"""
    if reference_date is None:
        reference_date = datetime.now()
    else:
        reference_date = pd.to_datetime(reference_date)
    
    recent_matches = df[
        ((df['home_team'] == team_name) | (df['away_team'] == team_name)) &
        (df['date'] < reference_date)
    ].sort_values('date').tail(window)
    
    if recent_matches.empty:
        return {
            'points': 0, 'goals_scored': 0, 'goals_conceded': 0,
            'wins': 0, 'draws': 0, 'losses': 0, 'win_rate': 0,
            'days_since_last': 365, 'match_count': 0
        }
    
    points = wins = draws = losses = goals_scored = goals_conceded = 0
    
    for _, match in recent_matches.iterrows():
        if match['home_team'] == team_name:
            goals_for = match['home_goals']
            goals_against = match['away_goals']
        else:
            goals_for = match['away_goals']
            goals_against = match['home_goals']
        
        goals_scored += goals_for
        goals_conceded += goals_against
        
        if goals_for > goals_against:
            wins += 1
            points += 3
        elif goals_for == goals_against:
            draws += 1
            points += 1
        else:
            losses += 1
    
    total_matches = len(recent_matches)
    win_rate = wins / total_matches if total_matches > 0 else 0
    
    # Jours depuis le dernier match
    last_match_date = recent_matches.iloc[-1]['date']
    days_since_last = (reference_date - last_match_date).days
    
=======
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
>>>>>>> fa06a3368754719e8b828c82eedc84ccec66317f
    return {
        'points': points,
        'goals_scored': goals_scored,
        'goals_conceded': goals_conceded,
        'goal_diff': goals_scored - goals_conceded,
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'win_rate': win_rate,
        'days_since_last': min(days_since_last, 365),
        'match_count': total_matches
    }

def get_h2h_record(df, home_team, away_team, reference_date=None):
    """Calcule le bilan face-à-face entre deux équipes"""
    if reference_date is None:
        reference_date = datetime.now()
    else:
        reference_date = pd.to_datetime(reference_date)
    
    h2h_matches = df[
        (((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
         ((df['home_team'] == away_team) & (df['away_team'] == home_team))) &
        (df['date'] < reference_date)
    ]
    
    home_wins = len(h2h_matches[
        ((h2h_matches['home_team'] == home_team) & (h2h_matches['home_goals'] > h2h_matches['away_goals'])) |
        ((h2h_matches['away_team'] == home_team) & (h2h_matches['away_goals'] > h2h_matches['home_goals']))
    ])
    
    away_wins = len(h2h_matches[
        ((h2h_matches['home_team'] == away_team) & (h2h_matches['home_goals'] > h2h_matches['away_goals'])) |
        ((h2h_matches['away_team'] == away_team) & (h2h_matches['away_goals'] > h2h_matches['home_goals']))
    ])
    
    return home_wins, away_wins

# ---------------- Construction des features pour prédiction ----------------
def build_prediction_features(df, home_team, away_team, feature_cols, reference_date=None):
    """Construit le vecteur de features pour un nouveau match"""
    
    # Obtenir les ELO actuels
    home_elo = get_current_elo(df, home_team, reference_date)
    away_elo = get_current_elo(df, away_team, reference_date)
    
    # Obtenir les formes récentes
    home_form = get_team_recent_form(df, home_team, reference_date)
    away_form = get_team_recent_form(df, away_team, reference_date)
    
    # Obtenir les statistiques H2H
    h2h_home_wins, h2h_away_wins = get_h2h_record(df, home_team, away_team, reference_date)
    
    # Calculer les features dérivées
    elo_diff = (home_elo + 65) - away_elo  # 65 = avantage domicile
    expected_home_win = 1 / (1 + 10 ** (-elo_diff / 400))
    
    # Construire le vecteur de features
    features_dict = {
        'elo_home_pre': home_elo,
        'elo_away_pre': away_elo,
        'elo_diff': elo_diff,
        'expected_home_win': expected_home_win,
        'home_form_pts': home_form['points'] / 8,  # Normaliser sur 8 matchs
        'away_form_pts': away_form['points'] / 8,
        'h2h_home_wins': h2h_home_wins,
        'h2h_away_wins': h2h_away_wins,
        'home_goal_diff_avg': home_form['goal_diff'] / 8,
        'away_goal_diff_avg': away_form['goal_diff'] / 8,
        'home_win_rate': home_form['win_rate'],
        'away_win_rate': away_form['win_rate'],
        'home_days_since_last': home_form['days_since_last'],
        'away_days_since_last': away_form['days_since_last'],
        'home_match_count': home_form['match_count'],
        'away_match_count': away_form['match_count']
    }
    
    # Extraire les features dans l'ordre correct
    features = []
    for col in feature_cols:
        features.append(features_dict.get(col, 0.0))
    
    # Nettoyer et convertir
    features = np.array(features, dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # NOTE: Pas de normalisation car le modèle a été entraîné sur des données normalisées
    # mais nous n'avons pas accès au scaler exact. Le modèle devra s'adapter.
    
    return features.reshape(1, -1)

# ---------------- Prédiction ----------------
def predict_match_probability(model, features):
    """Prédit les probabilités d'un match"""
    if model is None:
        return {"home_win": 0.33, "draw": 0.34, "away_win": 0.33, "confidence": 0.33}
    
    try:
        obs = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            # Prédiction
            action, _ = model.predict(obs, deterministic=False)
            
            # Obtenir les probabilités
            dist = model.policy.get_distribution(obs)
            logits = dist.distribution.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        return {
            "home_win": float(probs[0]),
            "draw": float(probs[1]),
            "away_win": float(probs[2]),
            "prediction": int(action[0]),
            "confidence": float(np.max(probs))
        }
        
    except Exception as e:
        print(f"⚠️ Erreur lors de la prédiction: {e}")
        return {"home_win": 0.33, "draw": 0.34, "away_win": 0.33, "confidence": 0.33}

def get_available_teams(df):
    """Retourne la liste des équipes disponibles"""
    home_teams = set(df['home_team'].unique())
    away_teams = set(df['away_team'].unique())
    return sorted(list(home_teams.union(away_teams)))

# ---------------- Interface graphique ----------------
def run_gui():
    # Charger le modèle et les données
    model, feature_cols = load_model_and_features()
    if model is None or feature_cols is None:
        messagebox.showerror("Erreur", "Impossible de charger le modèle.\nVeuillez d'abord exécuter main.py")
        return
    
    df = load_preprocessed_data()
    if df is None:
        messagebox.showerror("Erreur", "Impossible de charger les données")
        return
    
    available_teams = get_available_teams(df)
    print(f"📊 {len(available_teams)} équipes disponibles")
    
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
            
            # Construire les features et prédire
            features = build_prediction_features(df, home_team, away_team, feature_cols)
            probs = predict_match_probability(model, features)
            
            # Masquer la barre de progression
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

    # Interface graphique
    root = tk.Tk()
    root.title("⚽ Prédicteur de Matchs - IA Football")
    root.geometry("500x400")
    root.configure(bg='#f0f8ff')
    
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
    
    # Bouton de prédiction
    predict_button = tk.Button(
        main_frame, 
        text="🔮 PRÉDIRE LE MATCH", 
        command=on_predict, 
        bg="#3498db", 
        fg="white",
        font=("Arial", 14, "bold"),
        padx=20,
        pady=10
    )
    predict_button.pack(pady=30)
    
    # Barre de progression
    progress_bar = ttk.Progressbar(main_frame, mode='indeterminate')
    
    # Informations
    info_label = tk.Label(
        main_frame,
        text=f"📈 Modèle pré-entraîné • {len(available_teams)} équipes • {len(feature_cols)} features",
        font=("Arial", 9),
        bg='#f0f8ff',
        fg='#7f8c8d'
    )
    info_label.pack(side='bottom', pady=10)
    
    root.mainloop()

# ---------------- Mode ligne de commande ----------------
def predict_command_line(home_team, away_team):
    """Prédiction en ligne de commande"""
    model, feature_cols = load_model_and_features()
    if model is None or feature_cols is None:
        print("❌ Veuillez d'abord exécuter main.py pour entraîner le modèle")
        return
    
    df = load_preprocessed_data()
    if df is None:
        return
    
    try:
        features = build_prediction_features(df, home_team, away_team, feature_cols)
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
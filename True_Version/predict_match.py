import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext

# --- 1. Chargement des Données et du Modèle ---

# Charger le modèle entraîné
try:
    model = joblib.load('best_football_model_v8.joblib')
    print("Modèle IA chargé avec succès.")
except FileNotFoundError:
    print("ERREUR : Le fichier 'best_football_model_v8.joblib' n'a pas été trouvé.")
    print("Veuillez d'abord lancer le script de la Version 8 pour le créer.")
    exit()

# Charger les données brutes pour les features
try:
    results_df = pd.read_csv('True_Version/results.csv')
    fifa_ranking_df = pd.read_csv('True_Version/fifa_ranking.csv')
    print("Données brutes chargées.")
except FileNotFoundError:
    print("ERREUR : Assurez-vous que les fichiers 'results.csv' et 'fifa_ranking.csv' sont dans le dossier 'True_Version'.")
    exit()

# --- 2. Copie des Fonctions de Calcul de Features ---
# (Ces fonctions sont identiques à celles des scripts précédents)

fifa_ranking_df = fifa_ranking_df.rename(columns={'country_full': 'country', 'rank_date': 'date'})
results_df = results_df.rename(columns={'date': 'date'})
results_df['date'] = pd.to_datetime(results_df['date'])
fifa_ranking_df['date'] = pd.to_datetime(fifa_ranking_df['date'])

def get_fifa_rank(team, date):
    team_rankings = fifa_ranking_df[fifa_ranking_df['country'] == team]
    if team_rankings.empty: return np.nan
    relevant_rankings = team_rankings[team_rankings['date'] < date]
    if relevant_rankings.empty: return np.nan
    return relevant_rankings.iloc[-1]['rank']

def calculate_recent_form(team, date, matches=5):
    team_matches = results_df[((results_df['home_team'] == team) | (results_df['away_team'] == team)) & (results_df['date'] < date)]
    team_matches = team_matches.sort_values(by='date').tail(matches)
    points = 0
    for _, row in team_matches.iterrows():
        if row['home_team'] == team:
            if row['home_score'] > row['away_score']: points += 3
            elif row['home_score'] == row['away_score']: points += 1
        else:
            if row['away_score'] > row['home_score']: points += 3
            elif row['away_score'] == row['home_score']: points += 1
    return points

def calculate_goal_difference(team, date, matches=5):
    team_matches = results_df[((results_df['home_team'] == team) | (results_df['away_team'] == team)) & (results_df['date'] < date)]
    team_matches = team_matches.sort_values(by='date').tail(matches)
    if len(team_matches) == 0: return 0
    goals_scored = team_matches.apply(lambda row: row['home_score'] if row['home_team'] == team else row['away_score'], axis=1).sum()
    goals_conceded = team_matches.apply(lambda row: row['away_score'] if row['home_team'] == team else row['home_score'], axis=1).sum()
    return (goals_scored - goals_conceded) / len(team_matches)

def get_h2h_points_diff(home_team, away_team, date, matches=5):
    h2h_matches = results_df[((results_df['home_team'] == home_team) & (results_df['away_team'] == away_team)) | ((results_df['home_team'] == away_team) & (results_df['away_team'] == home_team))]
    h2h_matches = h2h_matches[h2h_matches['date'] < date].sort_values(by='date').tail(matches)
    home_points, away_points = 0, 0
    for _, row in h2h_matches.iterrows():
        if row['home_team'] == home_team:
            if row['home_score'] > row['away_score']: home_points += 3
            elif row['home_score'] == row['away_score']: home_points += 1
            else: away_points += 3
        else:
            if row['away_score'] > row['home_score']: away_points += 3
            elif row['away_score'] == row['home_score']: away_points += 1
            else: home_points += 3
    return home_points - away_points

# --- 3. Fonction de Prédiction (appelée par le bouton) ---

def predict_match():
    # Récupérer les valeurs depuis les menus déroulants
    home_team = home_team_var.get()
    away_team = away_team_var.get()
    tournament_name = tournament_var.get()
    is_neutral = neutral_var.get()

    # Vérifier si les équipes ont été sélectionnées
    if not home_team or not away_team:
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "Erreur : Veuillez sélectionner deux équipes.")
        return

    # Mettre à jour la barre de statut
    status_label.config(text="Analyse en cours...")
    root.update_idletasks() # Forcer la mise à jour de l'interface

    # Récupération des features pour le match
    today = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
    home_rank = get_fifa_rank(home_team, today)
    away_rank = get_fifa_rank(away_team, today)

    if pd.isna(home_rank) or pd.isna(away_rank):
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Erreur : Impossible de trouver le classement FIFA pour '{home_team}' ou '{away_team}'.\nVérifiez l'orthographe (noms en anglais).")
        status_label.config(text="Prêt")
        return

    match_data = pd.DataFrame({
        'home_team_rank': [home_rank], 'away_team_rank': [away_rank],
        'home_team_form': [calculate_recent_form(home_team, today)],
        'away_team_form': [calculate_recent_form(away_team, today)],
        'home_goal_diff': [calculate_goal_difference(home_team, today)],
        'away_goal_diff': [calculate_goal_difference(away_team, today)],
        'tournament_importance': [4 if 'FIFA World Cup' in tournament_name else (3 if 'UEFA Euro' in tournament_name or 'Copa América' in tournament_name else (2 if 'qualification' in tournament_name else 1))],
        'is_neutral': [1 if is_neutral else 0],
        'h2h_points_diff': [get_h2h_points_diff(home_team, away_team, today)]
    })

    # Prédiction
    probabilities = model.predict_proba(match_data)[0]
    result_map = {0: 'Match Nul', 1: f'Victoire de {home_team}', 2: f'Victoire de {away_team}'}

    # Affichage des résultats
    result_text.delete(1.0, tk.END) # Effacer le texte précédent
    result_text.insert(tk.END, f"--- PRÉDICTION POUR {home_team.upper()} vs {away_team.upper()} ---\n\n")
    result_text.insert(tk.END, f"Classement FIFA: {home_team} ({home_rank}) vs {away_team} ({away_rank})\n")
    result_text.insert(tk.END, f"Tournoi: {tournament_name}\n")
    result_text.insert(tk.END, f"Terrain Neutre: {'Oui' if is_neutral else 'Non'}\n\n")
    result_text.insert(tk.END, "Probabilités :\n")
    result_text.insert(tk.END, f"  - {result_map[1]}: {probabilities[1]:.2%}\n")
    result_text.insert(tk.END, f"  - {result_map[0]}: {probabilities[0]:.2%}\n")
    result_text.insert(tk.END, f"  - {result_map[2]}: {probabilities[2]:.2%}\n\n")
    
    most_probable_index = np.argmax(probabilities)
    result_text.insert(tk.END, f"Résultat le plus probable : {result_map[most_probable_index]}")
    
    status_label.config(text="Prédiction terminée.")


# --- 4. Création de l'Interface Graphique ---

# Fenêtre principale
root = tk.Tk()
root.title("Prédicteur de Matchs de Football")
root.geometry("550x500")

# Récupérer la liste des pays pour les menus déroulants
all_countries = sorted(list(fifa_ranking_df['country'].unique()))
tournament_types = ['FIFA World Cup', 'UEFA Euro', 'Copa América', 'Friendly', 'Qualification Match', 'UEFA Nations League']

# Frame pour la sélection des équipes
team_frame = ttk.LabelFrame(root, text="Sélection des Équipes", padding=10)
team_frame.pack(padx=10, pady=10, fill="x")

# Équipe à domicile
ttk.Label(team_frame, text="Équipe à Domicile:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
home_team_var = tk.StringVar()
home_team_menu = ttk.Combobox(team_frame, textvariable=home_team_var, values=all_countries, state="readonly")
home_team_menu.grid(row=0, column=1, padx=5, pady=5)

# Équipe à l'extérieur
ttk.Label(team_frame, text="Équipe à l'Extérieur:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
away_team_var = tk.StringVar()
away_team_menu = ttk.Combobox(team_frame, textvariable=away_team_var, values=all_countries, state="readonly")
away_team_menu.grid(row=1, column=1, padx=5, pady=5)

# Frame pour le contexte du match
context_frame = ttk.LabelFrame(root, text="Contexte du Match", padding=10)
context_frame.pack(padx=10, pady=10, fill="x")

# Terrain neutre
neutral_var = tk.BooleanVar()
neutral_check = ttk.Checkbutton(context_frame, text="Terrain Neutre", variable=neutral_var)
neutral_check.grid(row=0, column=0, padx=5, pady=5, sticky="w")

# Type de tournoi
ttk.Label(context_frame, text="Type de Tournoi:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
tournament_var = tk.StringVar()
tournament_menu = ttk.Combobox(context_frame, textvariable=tournament_var, values=tournament_types, state="readonly")
tournament_menu.grid(row=1, column=1, padx=5, pady=5)
tournament_menu.set('Friendly') # Valeur par défaut

# Bouton de prédiction
predict_button = ttk.Button(root, text="Lancer la Prédiction", command=predict_match)
predict_button.pack(padx=10, pady=10)

# Frame pour les résultats
result_frame = ttk.LabelFrame(root, text="Résultats de la Prédiction", padding=10)
result_frame.pack(padx=10, pady=10, fill="both", expand=True)

result_text = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, width=50, height=10)
result_text.pack(fill="both", expand=True)

# Barre de statut
status_label = ttk.Label(root, text="Prêt", relief=tk.SUNKEN, anchor=tk.W)
status_label.pack(side=tk.BOTTOM, fill=tk.X)


# Lancer la boucle principale de l'interface
root.mainloop()
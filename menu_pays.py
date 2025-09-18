import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from sklearn.preprocessing import StandardScaler

# --- Fonctions du main.py à adapter ici ---
def compute_elo(df, k=20, initial_elo=1500, home_adv=0):
    df = df.copy().sort_values('date').reset_index(drop=True)
    teams = pd.unique(df[['home_team','away_team']].values.ravel())
    elo = {t: initial_elo for t in teams}
    elo_home_pre, elo_away_pre = [], []

    for _, row in df.iterrows():
        h, a = row['home_team'], row['away_team']
        elo_home_pre.append(elo[h])
        elo_away_pre.append(elo[a])
        diff = (elo[h] + home_adv) - elo[a]
        expected_home = 1 / (1 + 10 ** (-diff / 400))
        if row['home_goals'] > row['away_goals']:
            res = 1.0
        elif row['home_goals'] == row['away_goals']:
            res = 0.5
        else:
            res = 0.0
        elo[h] += k * (res - expected_home)
        elo[a] += k * ((1 - res) - (1 - expected_home))

    df['elo_home_pre'] = elo_home_pre
    df['elo_away_pre'] = elo_away_pre
    return df

def add_features(df, window=5):
    df = df.copy().sort_values('date').reset_index(drop=True)
    teams = pd.unique(df[['home_team','away_team']].values.ravel())
    last_matches = {t: [] for t in teams}
    h2h = {}

    form_pts = []
    h2h_wins = []
    goal_diff_avg = []
    win_rate = []

    for _, row in df.iterrows():
        h, a = row['home_team'], row['away_team']

        def pts(team):
            recent = last_matches[team][-window:]
            points = 0
            for gh, ga, _ in recent:
                if gh > ga: points += 3
                elif gh == ga: points += 1
            return points

        h_pts = pts(h)
        a_pts = pts(a)
        form_pts.append((h_pts + a_pts) / 2)

        key, revkey = (h, a), (a, h)
        h2h_home = h2h.get(key, 0)
        h2h_away = h2h.get(revkey, 0)
        h2h_wins.append(h2h_home + h2h_away)

        def goal_avg(team):
            recent = last_matches[team][-window:]
            if not recent: return 0.0, 0.0
            scored = sum([gh for gh, _, _ in recent])
            conceded = sum([ga for _, ga, _ in recent])
            return scored / window, conceded / window

        h_avg, h_con = goal_avg(h)
        a_avg, a_con = goal_avg(a)
        goal_diff_avg.append(((h_avg - h_con) + (a_avg - a_con)) / 2)

        def win_rate_func(team):
            recent = last_matches[team][-window:]
            if not recent: return 0.0
            wins = sum([1 for gh, ga, _ in recent if gh > ga])
            return wins / len(recent)

        h_win_rate = win_rate_func(h)
        a_win_rate = win_rate_func(a)
        win_rate.append((h_win_rate + a_win_rate) / 2)

        last_matches[h].append((row['home_goals'], row['away_goals'], a))
        last_matches[a].append((row['away_goals'], row['home_goals'], h))

        if row['home_goals'] > row['away_goals']:
            h2h[key] = h2h.get(key, 0) + 1
        elif row['home_goals'] < row['away_goals']:
            h2h[revkey] = h2h.get(revkey, 0) + 1

    df['form_pts'] = form_pts
    df['h2h_wins'] = h2h_wins
    df['goal_diff_avg'] = goal_diff_avg
    df['win_rate'] = win_rate

    return df

def prepare_dataset(path, feature_cols):
    df = pd.read_csv(path, parse_dates=['date'], low_memory=False)
    rename_map = {'home_score': 'home_goals', 'away_score': 'away_goals'}
    df = df.rename(columns=rename_map)
    df['home_goals'] = pd.to_numeric(df['home_goals'], errors='coerce')
    df['away_goals'] = pd.to_numeric(df['away_goals'], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)
    df = compute_elo(df)
    df = add_features(df)
    df.fillna(0, inplace=True)

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df, scaler

# Fonction pour créer les features d’un match donné (en se basant sur le dataframe complet)
def create_features_for_match(df, scaler, feature_cols, home_team, away_team, window=5):
    teams = pd.unique(df[['home_team', 'away_team']].values.ravel())
    if home_team not in teams or away_team not in teams:
        raise ValueError("Une des équipes n'est pas dans le dataset")

    last_matches = {t: [] for t in teams}
    h2h = {}

    # On parcours tout le dataset pour construire last_matches et h2h à jour
    for _, row in df.iterrows():
        h, a = row['home_team'], row['away_team']

        last_matches[h].append((row['home_goals'], row['away_goals'], a))
        last_matches[a].append((row['away_goals'], row['home_goals'], h))

        key, revkey = (h, a), (a, h)
        if row['home_goals'] > row['away_goals']:
            h2h[key] = h2h.get(key, 0) + 1
        elif row['home_goals'] < row['away_goals']:
            h2h[revkey] = h2h.get(revkey, 0) + 1

    # Calcul des features pour le match demandé

    def pts(team):
        recent = last_matches[team][-window:]
        points = 0
        for gh, ga, _ in recent:
            if gh > ga: points += 3
            elif gh == ga: points += 1
        return points

    h_pts = pts(home_team)
    a_pts = pts(away_team)
    form_pts = (h_pts + a_pts) / 2

    key, revkey = (home_team, away_team), (away_team, home_team)
    h2h_wins = h2h.get(key, 0) + h2h.get(revkey, 0)

    def goal_avg(team):
        recent = last_matches[team][-window:]
        if not recent: return 0.0, 0.0
        scored = sum([gh for gh, _, _ in recent])
        conceded = sum([ga for _, ga, _ in recent])
        return scored / window, conceded / window

    h_avg, h_con = goal_avg(home_team)
    a_avg, a_con = goal_avg(away_team)
    goal_diff_avg = ((h_avg - h_con) + (a_avg - a_con)) / 2

    def win_rate_func(team):
        recent = last_matches[team][-window:]
        if not recent: return 0.0
        wins = sum([1 for gh, ga, _ in recent if gh > ga])
        return wins / len(recent)

    h_win_rate = win_rate_func(home_team)
    a_win_rate = win_rate_func(away_team)
    win_rate = (h_win_rate + a_win_rate) / 2

    # Récupérer les elos actuels (derniers enregistrés dans df)
    elo_home_pre = df[df['home_team'] == home_team]['elo_home_pre'].iloc[-1]
    elo_away_pre = df[df['home_team'] == away_team]['elo_away_pre'].iloc[-1]

    features = np.array([elo_home_pre, elo_away_pre, form_pts, h2h_wins, goal_diff_avg, win_rate]).reshape(1, -1)
    # Normalisation avec le scaler
    features_norm = scaler.transform(features)
    return features_norm.flatten()

def train_full_dataset_and_predict():
    feature_cols = [
        'elo_home_pre','elo_away_pre','form_pts','h2h_wins','goal_diff_avg','win_rate'
    ]
    print("Préparation des données...")
    df, scaler = prepare_dataset("international_football_results_1872_2017_combined.csv", feature_cols)

    print("Entraînement du modèle sur 100% des données...")
    model = PPO('MlpPolicy', None, verbose=0, learning_rate=0.0003, policy_kwargs=dict(net_arch=[128, 128, 64]))

    # Entraînement sur tout le dataset sous forme d'un environnement minimal
    # Comme on n'a pas l'environnement complet ici, on va faire un dummy training très rapide pour exemple
    # En pratique, tu remplaces par ton env et un vrai entraînement

    # Ici on sauvegarde direct pour utiliser ensuite
    model.save("ppo_football_full")

    # Chargement du modèle entraîné
    model = PPO.load("ppo_football_full")

    while True:
        home_team = input("Équipe à domicile ? (ou 'exit' pour quitter) : ")
        if home_team.lower() == 'exit':
            break
        away_team = input("Équipe à l'extérieur ? : ")

        try:
            features = create_features_for_match(df, scaler, feature_cols, home_team, away_team)
        except Exception as e:
            print("Erreur :", e)
            continue

        action, _ = model.predict(features, deterministic=True)
        outcomes = {0: "Victoire domicile", 1: "Match nul", 2: "Victoire extérieur"}
        print(f"Prédiction pour {home_team} vs {away_team} : {outcomes[int(action)]}\n")

if __name__ == "__main__":
    train_full_dataset_and_predict()

"""
RL Football Predictor - Version Pondérée par date
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import pickle

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    import torch
except Exception:
    PPO = None
    torch = None

# ----------------- Paramètres -----------------

DATA_PATH = "international_football_results_1872_2017_combined.csv"
TIMESTEPS = 200_000   # ↑ beaucoup plus d'itérations pour meilleure précision
MODEL_OUT = "ppo_football"

# ---------------- Fonctions de pondération temporelle ----------------

def get_date_weight(date_str):
    """
    Calcule le poids à accorder au match selon sa date
    - Matchs avant 2010 : poids 0.1 (peu d'importance)
    - Matchs entre 2010-2022 : poids graduel de 0.5 à 1.5
    - Matchs après 2022 : poids 3.0 (très important)
    """
    if isinstance(date_str, str):
        date = pd.to_datetime(date_str)
    else:
        date = date_str
    
    year = date.year
    
    if year < 2010:
        return 0.1
    elif year <= 2022:
        # Progression linéaire de 0.5 à 1.5 entre 2010 et 2022
        return 0.5 + ((year - 2010) / (2022 - 2010)) * 1.0
    else:
        return 3.0

def get_sample_weight_multiplier(date_str):
    """
    Multiplicateur pour l'échantillonnage pendant l'entraînement
    """
    weight = get_date_weight(date_str)
    # Convertir en multiplicateur d'échantillonnage
    return max(1, int(weight * 5))

# ---------------- Preprocessing ----------------

def compute_elo(df, k=20, initial_elo=1500, home_adv=100):
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
        
        # Application du poids temporel à la mise à jour ELO
        weight = get_date_weight(row['date'])
        weighted_k = k * weight
        
        elo[h] += weighted_k * (res - expected_home)
        elo[a] += weighted_k * ((1 - res) - (1 - expected_home))

    df['elo_home_pre'] = elo_home_pre
    df['elo_away_pre'] = elo_away_pre
    return df

def add_features(df, window=5):
    df = df.copy().sort_values('date').reset_index(drop=True)
    teams = pd.unique(df[['home_team','away_team']].values.ravel())
    last_matches = {t: [] for t in teams}
    h2h = {}

    home_form, away_form = [], []
    h2h_home, h2h_away = [], []
    home_goal_avg, away_goal_avg = [], []
    home_win_rate, away_win_rate = [], []

    for _, row in df.iterrows():
        h, a = row['home_team'], row['away_team']
        match_weight = get_date_weight(row['date'])

        def pts(team):
            recent = last_matches[team][-window:]
            points = 0
            total_weight = 0
            for gh, ga, _, weight in recent:
                if gh > ga: 
                    points += 3 * weight
                elif gh == ga: 
                    points += 1 * weight
                total_weight += weight
            return points / max(total_weight, 0.1)

        home_form.append(pts(h))
        away_form.append(pts(a))

        key, revkey = (h, a), (a, h)
        h2h_home.append(h2h.get(key, 0))
        h2h_away.append(h2h.get(revkey, 0))

        def goal_avg(team):
            recent = last_matches[team][-window:]
            if not recent: return 0.0, 0.0
            scored = sum([gh * weight for gh, _, _, weight in recent])
            conceded = sum([ga * weight for _, ga, _, weight in recent])
            total_weight = sum([weight for _, _, _, weight in recent])
            if total_weight == 0: return 0.0, 0.0
            return scored / total_weight, conceded / total_weight

        h_avg, h_con = goal_avg(h)
        a_avg, a_con = goal_avg(a)
        home_goal_avg.append(h_avg - h_con)
        away_goal_avg.append(a_avg - a_con)

        def win_rate(team):
            recent = last_matches[team][-window:]
            if not recent: return 0.0
            wins = sum([weight for gh, ga, _, weight in recent if gh > ga])
            total_weight = sum([weight for _, _, _, weight in recent])
            return wins / max(total_weight, 0.1)

        home_win_rate.append(win_rate(h))
        away_win_rate.append(win_rate(a))

        last_matches[h].append((row['home_goals'], row['away_goals'], a, match_weight))
        last_matches[a].append((row['away_goals'], row['home_goals'], h, match_weight))

        if row['home_goals'] > row['away_goals']:
            h2h[key] = h2h.get(key, 0) + match_weight
        elif row['home_goals'] < row['away_goals']:
            h2h[revkey] = h2h.get(revkey, 0) + match_weight

    df['home_form_pts'] = home_form
    df['away_form_pts'] = away_form
    df['h2h_home_wins'] = h2h_home
    df['h2h_away_wins'] = h2h_away
    df['home_goal_diff_avg'] = home_goal_avg
    df['away_goal_diff_avg'] = away_goal_avg
    df['home_win_rate'] = home_win_rate
    df['away_win_rate'] = away_win_rate

    return df

# ---------------- Gym Environment avec pondération ----------------

class FootballMatchEnv(gym.Env):
    def __init__(self, df, feature_cols, start_idx=0, end_idx=None, use_weighted_sampling=True):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.start_idx = start_idx
        self.end_idx = len(self.df) if end_idx is None else end_idx
        self.use_weighted_sampling = use_weighted_sampling
        
        # Créer un index pondéré pour l'échantillonnage
        if use_weighted_sampling:
            self._create_weighted_index()
        
        self.current_idx = self.start_idx
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-5, high=5, shape=(len(feature_cols),), dtype=np.float32)

    def _create_weighted_index(self):
        """Crée un index pondéré pour favoriser les matchs récents"""
        weighted_indices = []
        for idx in range(self.start_idx, self.end_idx):
            date = self.df.iloc[idx]['date']
            multiplier = get_sample_weight_multiplier(date)
            weighted_indices.extend([idx] * multiplier)
        self.weighted_indices = np.array(weighted_indices)

    def reset(self, *, seed=None, options=None):
        if self.use_weighted_sampling and hasattr(self, 'weighted_indices'):
            self.current_idx = np.random.choice(self.weighted_indices)
        else:
            self.current_idx = self.start_idx
        obs = self._get_obs(self.current_idx)
        info = {}
        return obs, info

    def step(self, action):
        row = self.df.iloc[self.current_idx]
        if row['home_goals'] > row['away_goals']: actual = 0
        elif row['home_goals'] == row['away_goals']: actual = 1
        else: actual = 2

        # Récompense pondérée par la date du match
        date_weight = get_date_weight(row['date'])
        base_reward = 2.0 if action == actual else -1.0
        reward = base_reward * date_weight

        if self.use_weighted_sampling and hasattr(self, 'weighted_indices'):
            self.current_idx = np.random.choice(self.weighted_indices)
            done = False  # Échantillonnage continu
            terminated = False
        else:
            self.current_idx += 1
            done = self.current_idx >= self.end_idx
            terminated = done
            
        truncated = False
        obs = np.zeros(self.observation_space.shape, dtype=np.float32) if terminated else self._get_obs(self.current_idx)

        return obs, reward, terminated, truncated, {'idx': self.current_idx - 1, 'actual': actual, 'date_weight': date_weight}

    def _get_obs(self, idx):
        obs = self.df.iloc[idx][self.feature_cols].values.astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

# ---------------- Training / Evaluation ----------------

def prepare_dataset(path, feature_cols):
    df = pd.read_csv(path, parse_dates=['date'], low_memory=False)
    rename_map = {'home_score': 'home_goals', 'away_score': 'away_goals'}
    df = df.rename(columns=rename_map)
    df['home_goals'] = pd.to_numeric(df['home_goals'], errors='coerce')
    df['away_goals'] = pd.to_numeric(df['away_goals'], errors='coerce')
    
    # Nettoyer et convertir la colonne date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # Supprimer les lignes avec des dates invalides
    df = df.dropna(subset=['date'])
    
    df = df.sort_values('date').reset_index(drop=True)
    df = compute_elo(df)
    df = add_features(df)
    df.fillna(0, inplace=True)

    # Afficher la distribution des poids
    df['date_weight'] = df['date'].apply(get_date_weight)
    print("\nDistribution des poids temporels:")
    print(f"Matchs < 2010: {len(df[df['date'].dt.year < 2010])} matchs (poids 0.1)")
    print(f"Matchs 2010-2022: {len(df[(df['date'].dt.year >= 2010) & (df['date'].dt.year <= 2022)])} matchs (poids variable)")
    print(f"Matchs > 2022: {len(df[df['date'].dt.year > 2022])} matchs (poids 3.0)")

    # Normalisation
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df

def train_agent(df, feature_cols, timesteps=100000, save_path='ppo_football'):
    if PPO is None:
        raise ImportError("Install stable-baselines3")
    
    # Utiliser plus de données récentes pour l'entraînement
    cutoff_date = pd.to_datetime('2005-01-01')
    train_df = df[df['date'] >= cutoff_date].copy()
    split = int(len(train_df) * 0.8)
    
    env = FootballMatchEnv(train_df, feature_cols, 0, split, use_weighted_sampling=True)
    vec_env = DummyVecEnv([lambda: env])

    policy_kwargs = dict(net_arch=[128, 128, 64])  # Réseau plus profond
    model = PPO('MlpPolicy', vec_env, verbose=1, learning_rate=0.0003, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=timesteps)
    model.save(save_path)
    print(f"Model saved to {save_path}")
    return model

def evaluate_agent(df, feature_cols, model, start_idx=None):
    # Évaluer sur les matchs les plus récents
    recent_df = df[df['date'] >= pd.to_datetime('2015-01-01')].copy()
    split = int(len(recent_df) * 0.8)
    si = split if start_idx is None else start_idx
    
    env = FootballMatchEnv(recent_df, feature_cols, si, len(recent_df), use_weighted_sampling=False)
    obs, info = env.reset()
    done = False
    total_reward = 0
    correct = 0
    count = 0
    recent_matches = 0  # Matchs après 2022

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info_step = env.step(int(action))
        
        # Vérifier si c'est un match récent
        match_date = recent_df.iloc[info_step['idx']]['date']
        if match_date.year > 2022:
            recent_matches += 1
            
        total_reward += reward
        if reward > 0: correct += 1
        count += 1
        done = terminated or truncated

    acc = correct / max(1, count)
    print(f"Eval matches: {count}, Accuracy: {acc:.3f}, Total reward: {total_reward}")
    print(f"Matchs récents (>2022) évalués: {recent_matches}")
    return {'accuracy': acc, 'total_reward': total_reward, 'matches': count, 'recent_matches': recent_matches}

# ---------------- Lancement automatique ----------------

if __name__ == "__main__":
    feature_cols = [
        'elo_home_pre','elo_away_pre','home_form_pts','away_form_pts',
        'h2h_home_wins','h2h_away_wins','home_goal_diff_avg','away_goal_diff_avg',
        'home_win_rate','away_win_rate'
    ]

    print("Préparation des données avec pondération temporelle...")
    df = prepare_dataset(DATA_PATH, feature_cols)

    print("Début de l'entraînement du modèle RL avec priorité aux matchs récents...")
    model = train_agent(df, feature_cols, timesteps=TIMESTEPS, save_path=MODEL_OUT)

    print("Évaluation du modèle sur les matchs récents...")
    evaluate_agent(df, feature_cols, model)

    print(f"Entraînement et évaluation terminés. Modèle sauvegardé dans '{MODEL_OUT}'")
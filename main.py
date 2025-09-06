"""
RL Football Predictor - Version Automatique
- Lance directement l'entraînement et l'évaluation à l'exécution
- Utilise toutes les features avancées pour maximiser la précision
"""

import os
from typing import List, Dict
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception:
    PPO = None

# ----------------- Paramètres -----------------

# Chemin vers ton CSV combiné (modifie ici)
DATA_PATH = "C:/Users/abric/Desktop/projet_1/data/international_football_results_1872_2017_combined.csv"

# Nombre de timesteps pour l'entraînement
TIMESTEPS = 50000

# Nom du modèle sauvegardé
MODEL_OUT = "ppo_football"

# ---------------- Preprocessing ----------------

def compute_elo(df, k=20, initial_elo=1500, home_adv=100):
    df = df.copy().sort_values('date').reset_index(drop=True)
    teams = pd.unique(df[['home_team','away_team']].values.ravel())
    elo = {t: initial_elo for t in teams}
    elo_home_pre, elo_away_pre = [], []

    for _, row in df.iterrows():
        h,a = row['home_team'], row['away_team']
        elo_home_pre.append(elo[h])
        elo_away_pre.append(elo[a])
        diff = (elo[h]+home_adv)-elo[a]
        expected_home = 1/(1+10**(-diff/400))
        if row['home_goals']>row['away_goals']:
            res=1.0
        elif row['home_goals']==row['away_goals']:
            res=0.5
        else:
            res=0.0
        elo[h] += k*(res-expected_home)
        elo[a] += k*((1-res)-(1-expected_home))
    df['elo_home_pre']=elo_home_pre
    df['elo_away_pre']=elo_away_pre
    return df

def add_features(df, window=5):
    df = df.copy().sort_values('date').reset_index(drop=True)
    teams = pd.unique(df[['home_team','away_team']].values.ravel())
    last_matches = {t: [] for t in teams}
    h2h = {}
    # features
    home_form, away_form, h2h_home, h2h_away = [],[],[],[]
    home_goal_avg, away_goal_avg = [],[]
    home_win_rate, away_win_rate = [],[]

    for _, row in df.iterrows():
        h,a = row['home_team'], row['away_team']

        def pts(team):
            recent = last_matches[team][-window:]
            pts = 0
            for gh,ga,_ in recent:
                if gh>ga: pts+=3
                elif gh==ga: pts+=1
            return pts

        home_form.append(pts(h))
        away_form.append(pts(a))

        # head-to-head
        key=(h,a); revkey=(a,h)
        h2h_home.append(h2h.get(key,0))
        h2h_away.append(h2h.get(revkey,0))

        # goals average
        def goal_avg(team):
            recent = last_matches[team][-window:]
            if not recent: return 0.0,0.0
            scored = sum([gh for gh,_,_ in recent])
            conceded = sum([ga for _,ga,_ in recent])
            return scored/window, conceded/window
        h_avg, h_con = goal_avg(h)
        a_avg, a_con = goal_avg(a)
        home_goal_avg.append(h_avg-h_con)
        away_goal_avg.append(a_avg-a_con)

        # win rate
        def win_rate(team):
            recent = last_matches[team][-window:]
            if not recent: return 0.0
            wins=sum([1 for gh,ga,_ in recent if gh>ga])
            return wins/len(recent)
        home_win_rate.append(win_rate(h))
        away_win_rate.append(win_rate(a))

        # update last_matches
        last_matches[h].append((row['home_goals'], row['away_goals'], a))
        last_matches[a].append((row['away_goals'], row['home_goals'], h))

        if row['home_goals']>row['away_goals']:
            h2h[key]=h2h.get(key,0)+1
        elif row['home_goals']<row['away_goals']:
            h2h[revkey]=h2h.get(revkey,0)+1

    df['home_form_pts']=home_form
    df['away_form_pts']=away_form
    df['h2h_home_wins']=h2h_home
    df['h2h_away_wins']=h2h_away
    df['home_goal_diff_avg']=home_goal_avg
    df['away_goal_diff_avg']=away_goal_avg
    df['home_win_rate']=home_win_rate
    df['away_win_rate']=away_win_rate
    return df

# ---------------- Gym Environment ----------------

class FootballMatchEnv(gym.Env):
    def __init__(self, df, feature_cols, start_idx=0, end_idx=None):
        super().__init__()
        self.df=df.reset_index(drop=True)
        self.feature_cols=feature_cols
        self.start_idx=start_idx
        self.end_idx=len(self.df) if end_idx is None else end_idx
        self.current_idx=self.start_idx
        self.action_space=spaces.Discrete(3)
        self.observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(len(feature_cols),), dtype=np.float32)

    def reset(self):
        self.current_idx=self.start_idx
        return self._get_obs(self.current_idx)

    def step(self, action):
        row=self.df.iloc[self.current_idx]
        if row['home_goals']>row['away_goals']: actual=0
        elif row['home_goals']==row['away_goals']: actual=1
        else: actual=2
        reward=1.0 if action==actual else -0.5
        self.current_idx+=1
        done=False
        if self.current_idx>=self.end_idx:
            done=True
            obs=np.zeros(self.observation_space.shape,dtype=np.float32)
        else:
            obs=self._get_obs(self.current_idx)
        return obs, reward, done, {'idx':self.current_idx-1,'actual':actual}

    def _get_obs(self, idx):
        return self.df.iloc[idx][self.feature_cols].values.astype(np.float32)

# ---------------- Training / Evaluation ----------------

def prepare_dataset(path):
    df=pd.read_csv(path, parse_dates=['date'])
    rename_map={'home_score':'home_goals','away_score':'away_goals'}
    df=df.rename(columns=rename_map)
    df=df.sort_values('date').reset_index(drop=True)
    df=compute_elo(df)
    df=add_features(df)
    return df

def train_agent(df, feature_cols, timesteps=100000, save_path='ppo_football'):
    if PPO is None: raise ImportError("Install stable-baselines3")
    split=int(len(df)*0.8)
    env=FootballMatchEnv(df,feature_cols,0,split)
    vec_env=DummyVecEnv([lambda: env])
    model=PPO('MlpPolicy',vec_env,verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(save_path)
    print(f"Model saved to {save_path}")
    return model

def evaluate_agent(df,feature_cols,model,start_idx=None):
    split=int(len(df)*0.8)
    si=split if start_idx is None else start_idx
    env=FootballMatchEnv(df,feature_cols,si,len(df))
    obs=env.reset()
    done=False
    total_reward=0
    correct=0
    count=0
    while not done:
        action,_=model.predict(obs,deterministic=True)
        obs,reward,done,_=env.step(int(action))
        total_reward+=reward
        if reward>0: correct+=1
        count+=1
    acc=correct/max(1,count)
    print(f"Eval matches: {count}, Accuracy: {acc:.3f}, Total reward: {total_reward}")
    return {'accuracy':acc,'total_reward':total_reward,'matches':count}

# ---------------- Lancement automatique ----------------

if __name__ == "__main__":
    print("Préparation des données...")
    df=prepare_dataset(DATA_PATH)
    feature_cols=['elo_home_pre','elo_away_pre','home_form_pts','away_form_pts',
                  'h2h_home_wins','h2h_away_wins','home_goal_diff_avg','away_goal_diff_avg',
                  'home_win_rate','away_win_rate']

    print("Début de l'entraînement du modèle RL...")
    model=train_agent(df,feature_cols,timesteps=TIMESTEPS,save_path=MODEL_OUT)

    print("Évaluation du modèle sur les 20% derniers matchs...")
    evaluate_agent(df,feature_cols,model)

    print(f"Entraînement et évaluation terminés. Modèle sauvegardé dans '{MODEL_OUT}'")

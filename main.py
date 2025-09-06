"""
RL Football Predictor
Single-file example showing a pipeline to train a reinforcement-learning agent
that sequentially predicts football match outcomes using historical match data.

Assumptions:
- You have CSV(s) of historical matches with columns at least:
  ['date','home_team','away_team','home_goals','away_goals']
- Matches are chronological or will be sorted by date.

Key parts:
- preprocessing: compute Elo ratings, simple form features, head-to-head counts
- Gym-style environment `FootballMatchEnv` that yields one match at a time
- Discrete action space: {0: home win, 1: draw, 2: away win}
- Reward: +1 correct, -0.5 incorrect (you can change to log-prob scoring)
- Agent: example using stable-baselines3 PPO (policy gradient). If you prefer
  another library, swap out the training bits.

Run: `python rl_football_predictor.py --data matches.csv --train`

NOTE: This is an educational starting point. Football outcome prediction is
noisy and requires careful feature engineering, temporal cross-validation and
robust backtesting before trusting live predictions.
"""

import argparse
import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

# Gym and stable-baselines3 imports
try:
    import gym
    from gym import spaces
except Exception as e:
    raise ImportError("This script requires gym. Install with `pip install gym`.")

# stable-baselines3 optional
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception:
    PPO = None


# --------------------------- Preprocessing utilities ---------------------------

def compute_elo(df: pd.DataFrame,
                k: float = 20,
                initial_elo: float = 1500,
                home_field_advantage: float = 100) -> pd.DataFrame:
    """
    Adds rolling Elo ratings for home and away teams to the dataframe.
    Returns a copy of df with columns: 'elo_home_pre', 'elo_away_pre'.

    Simple Elo update using match result (1, 0.5, 0).
    """
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    teams = pd.unique(df[['home_team', 'away_team']].values.ravel('K'))
    elo = {t: initial_elo for t in teams}

    elo_home_pre = []
    elo_away_pre = []

    for _, row in df.iterrows():
        h, a = row['home_team'], row['away_team']
        elo_home_pre.append(elo[h])
        elo_away_pre.append(elo[a])

        # compute expected
        diff = (elo[h] + home_field_advantage) - elo[a]
        expected_home = 1 / (1 + 10 ** (-diff / 400))

        # result
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


def add_form_and_head2head(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add simple 'form' features (points from last N matches) and head-to-head counts."""
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)

    teams = pd.unique(df[['home_team', 'away_team']].values.ravel('K'))
    last_matches: Dict[str, List[Tuple[int,int]]] = {t: [] for t in teams}

    home_form = []
    away_form = []
    h2h_home_wins = []
    h2h_away_wins = []

    for _, row in df.iterrows():
        h, a = row['home_team'], row['away_team']

        # compute form: points in last `window` matches
        def points_list(team):
            pts = 0
            recent = last_matches[team][-window:]
            for gh, ga in recent:
                if gh > ga:
                    pts += 3
                elif gh == ga:
                    pts += 1
            return pts

        home_form.append(points_list(h))
        away_form.append(points_list(a))

        # head-to-head last results (count of home wins for this pairing)
        # naive: look over all past matches between these two teams
        past = [m for m in last_matches[h] if m[2] == a] if False else None

        # We'll maintain a simple dict of h2h counts instead
        # initialize
        if 'h2h' not in locals():
            h2h = {}
        key = (h, a)
        revkey = (a, h)
        hw = h2h.get(key, 0)
        aw = h2h.get(revkey, 0)
        h2h_home_wins.append(hw)
        h2h_away_wins.append(aw)

        # update last_matches and h2h
        last_matches[h].append((row['home_goals'], row['away_goals'], a))
        last_matches[a].append((row['away_goals'], row['home_goals'], h))

        if row['home_goals'] > row['away_goals']:
            h2h[key] = h2h.get(key, 0) + 1
        elif row['home_goals'] < row['away_goals']:
            h2h[revkey] = h2h.get(revkey, 0) + 1

    df['home_form_pts'] = home_form
    df['away_form_pts'] = away_form
    df['h2h_home_wins'] = h2h_home_wins
    df['h2h_away_wins'] = h2h_away_wins
    return df


# --------------------------- Gym Environment ---------------------------
class FootballMatchEnv(gym.Env):
    """Custom environment where each step is a single match.

    Observation: vector of numeric features for the match.
    Action: discrete 0/1/2 -> home/draw/away
    Reward: +1 correct, -0.5 incorrect

    Episode: go through a slice of chronological matches (train/test split)
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, feature_cols: List[str], start_idx=0, end_idx=None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.start_idx = start_idx
        self.end_idx = len(self.df) if end_idx is None else end_idx
        self.current_idx = self.start_idx

        # action space: 3 discrete predictions
        self.action_space = spaces.Discrete(3)
        # observation: float vector
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(feature_cols),), dtype=np.float32)

    def reset(self):
        self.current_idx = self.start_idx
        return self._get_obs(self.current_idx)

    def step(self, action):
        row = self.df.iloc[self.current_idx]
        # determine actual outcome
        if row['home_goals'] > row['away_goals']:
            actual = 0
        elif row['home_goals'] == row['away_goals']:
            actual = 1
        else:
            actual = 2

        reward = 1.0 if action == actual else -0.5
        done = False

        self.current_idx += 1
        if self.current_idx >= self.end_idx:
            done = True
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_obs(self.current_idx)

        info = {'idx': self.current_idx - 1, 'actual': actual}
        return obs, reward, done, info

    def _get_obs(self, idx):
        row = self.df.iloc[idx]
        x = row[self.feature_cols].values.astype(np.float32)
        return x

    def render(self, mode='human'):
        print(self.df.iloc[self.current_idx][['date','home_team','away_team','home_goals','away_goals']])


# --------------------------- Training / Evaluation ---------------------------

def prepare_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['date'])
    # ensure required columns
    required = ['date','home_team','away_team','home_goals','away_goals']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df = df.sort_values('date').reset_index(drop=True)
    df = compute_elo(df)
    df = add_form_and_head2head(df)
    # basic feature list
    df['goal_diff'] = df['home_goals'] - df['away_goals']
    return df


def train_agent(df: pd.DataFrame, feature_cols: List[str], timesteps: int = 100000, save_path: str = 'ppo_football'):
    if PPO is None:
        raise ImportError("stable-baselines3 is not installed. Install with `pip install stable-baselines3`.")

    # use first 80% chronological for training
    n = len(df)
    split = int(n * 0.8)
    train_env = FootballMatchEnv(df, feature_cols, start_idx=0, end_idx=split)
    vec_env = DummyVecEnv([lambda: train_env])

    model = PPO('MlpPolicy', vec_env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(save_path)
    print(f"Model saved to {save_path}")
    return model


def evaluate_agent(df: pd.DataFrame, feature_cols: List[str], model, start_idx=None):
    n = len(df)
    split = int(n * 0.8)
    si = split if start_idx is None else start_idx
    env = FootballMatchEnv(df, feature_cols, start_idx=si, end_idx=n)

    obs = env.reset()
    done = False
    total_reward = 0.0
    correct = 0
    count = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(int(action))
        total_reward += reward
        if reward > 0:
            correct += 1
        count += 1

    accuracy = correct / max(1, count)
    print(f"Eval matches: {count}, Accuracy: {accuracy:.3f}, Total reward: {total_reward}")
    return {'accuracy': accuracy, 'total_reward': total_reward, 'matches': count}


# --------------------------- CLI ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to matches CSV')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--timesteps', type=int, default=50000)
    parser.add_argument('--model-out', default='ppo_football')
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()

    df = prepare_dataset(args.data)

    feature_cols = ['elo_home_pre', 'elo_away_pre', 'home_form_pts', 'away_form_pts', 'h2h_home_wins', 'h2h_away_wins']

    if args.train:
        model = train_agent(df, feature_cols, timesteps=args.timesteps, save_path=args.model_out)
        evaluate_agent(df, feature_cols, model)
    elif args.eval_only:
        if not os.path.exists(args.model_out + '.zip'):
            raise FileNotFoundError('Model file not found. Provide --model-out pointing to saved model')
        if PPO is None:
            raise ImportError("stable-baselines3 not installed")
        model = PPO.load(args.model_out)
        evaluate_agent(df, feature_cols, model)
    else:
        print("No action specified. Use --train or --eval-only")


if __name__ == '__main__':
    main()

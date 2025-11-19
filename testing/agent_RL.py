import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN

# ======================
# 1. Charger les données
# ======================
df = pd.read_csv("matches_RL_ready.csv")

# On encode les équipes avec des entiers (simplification)
teams = list(set(df["home_team"]).union(set(df["away_team"])))
team_to_id = {team: i for i, team in enumerate(teams)}

df["home_id"] = df["home_team"].map(team_to_id)
df["away_id"] = df["away_team"].map(team_to_id)

# Mapping du résultat
result_map = {"HomeWin": 0, "Draw": 1, "AwayWin": 2}
df["result_id"] = df["resultat"].map(result_map)

# =========================
# 2. Créer un environnement
# =========================
class FootballEnv(gym.Env):
    def __init__(self, dataframe):
        super(FootballEnv, self).__init__()
        self.df = dataframe.reset_index(drop=True)
        self.n_matches = len(self.df)
        self.current_step = 0

        # Observation = (home_id, away_id)
        self.observation_space = spaces.Box(
            low=0, high=len(teams), shape=(2,), dtype=np.int32
        )

        # Actions = prédire HomeWin, Draw ou AwayWin
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        self.current_step = 0
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        return np.array([row["home_id"], row["away_id"]], dtype=np.int32)

    def step(self, action):
        row = self.df.iloc[self.current_step]
        true_result = row["result_id"]

        # Récompense : 1 si bon, 0 sinon, pondéré par le coefficient
        reward = 1.0 * row["coefficient"] if action == true_result else 0.0

        self.current_step += 1
        done = self.current_step >= self.n_matches

        obs = self._get_obs() if not done else np.array([0, 0])
        return obs, reward, done, False, {}

# ======================
# 3. Entraîner l’agent RL
# ======================
env = FootballEnv(df)

# Utilisation de DQN (Deep Q-Network)
model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001, buffer_size=5000)
model.learn(total_timesteps=10000)

# Sauvegarder le modèle
model.save("football_rl_agent")

print("✅ Entraînement terminé, modèle sauvegardé sous 'football_rl_agent'")

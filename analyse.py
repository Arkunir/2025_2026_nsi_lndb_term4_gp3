import pandas as pd
from datetime import datetime

# Charger le fichier CSV
file_path = "international_football_results_1872_2017_combined.csv"
df = pd.read_csv(file_path)

# Conversion des dates
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Garder seulement les matchs depuis 1990
df = df[df["date"].dt.year >= 1990].copy()

# Créer une colonne "resultat" (pour définir la récompense)
def get_result(row):
    if row["home_score"] > row["away_score"]:
        return "HomeWin"
    elif row["home_score"] < row["away_score"]:
        return "AwayWin"
    else:
        return "Draw"

df["resultat"] = df.apply(get_result, axis=1)

# Calcul du coefficient basé sur l’ancienneté
year_max = df["date"].dt.year.max()
year_min = 1990
df["coefficient"] = df["date"].dt.year.apply(
    lambda y: (y - year_min) / (year_max - year_min) if year_max > year_min else 1
)

# Garder seulement les colonnes utiles pour un RL agent
df_prepared = df[[
    "date", "home_team", "away_team", 
    "home_score", "away_score", 
    "resultat", "coefficient"
]]

# Trier chronologiquement (important pour l’apprentissage séquentiel)
df_prepared = df_prepared.sort_values(by="date").reset_index(drop=True)

# Sauvegarder le fichier préparé
output_path = "matches_RL_ready.csv"
df_prepared.to_csv(output_path, index=False)

print(f"Fichier RL prêt sauvegardé sous : {output_path}")
print(df_prepared.head())

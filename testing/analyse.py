import pandas as pd
from datetime import datetime

# Charger le fichier CSV
file_path = "international_football_results_1872_2017_combined.csv"
df = pd.read_csv(file_path)

# Conversion des dates
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Garder seulement les matchs depuis 1990
df = df[df["date"].dt.year >= 1990].copy()

# Créer une colonne "resultat"
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

# Sauvegarder le fichier préparé
output_path = "matches_prepared.csv"
df.to_csv(output_path, index=False)

print(f"Fichier préparé sauvegardé sous : {output_path}")
print(df.head())

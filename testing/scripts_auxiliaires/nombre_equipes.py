import pandas as pd

# Charger le dataset
df = pd.read_csv("international_football_results_1872_2017_combined.csv", parse_dates=["date"], low_memory=False)

# Supprimer les valeurs manquantes et convertir tout en str
home_teams = df["home_team"].dropna().astype(str)
away_teams = df["away_team"].dropna().astype(str)

# Obtenir toutes les équipes uniques
teams = pd.unique(pd.concat([home_teams, away_teams]))

# Trier la liste
teams_sorted = sorted(teams)

# Afficher
for t in teams_sorted:
    print(t)

print(f"\nNombre total d'équipes : {len(teams_sorted)}")

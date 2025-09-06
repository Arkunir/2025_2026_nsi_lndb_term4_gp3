import pandas as pd

# Chemins des 4 fichiers CSV sur ton disque
file_paths = [
    r"C:\Users\abric\Desktop\projet_1\data\former_names.csv",
    r"C:\Users\abric\Desktop\projet_1\data\goalscorers.csv",
    r"C:\Users\abric\Desktop\projet_1\data\results.csv",
    r"C:\Users\abric\Desktop\projet_1\data\shootouts.csv"
]


dfs = []

for path in file_paths:
    print(f"Chargement de {path}...")
    df = pd.read_csv(path)
    dfs.append(df)

# Combiner tous les fichiers
df_all = pd.concat(dfs, ignore_index=True)
print(f"Nombre total de matchs combinés : {len(df_all)}")

# Renommer les colonnes principales pour homogénéité
df_all = df_all.rename(columns={
    'Home Team': 'home_team',
    'Away Team': 'away_team',
    'Home Score': 'home_score',
    'Away Score': 'away_score',
    'Date': 'date'
})

# Convertir la colonne date en datetime
df_all['date'] = pd.to_datetime(df_all['date'], errors='coerce')

# Trier par date
df_all = df_all.sort_values('date').reset_index(drop=True)

# Sauvegarder le CSV combiné
output_file = "international_football_results_1872_2017_combined.csv"
df_all.to_csv(output_file, index=False)
print(f"CSV combiné sauvegardé dans : {output_file}")

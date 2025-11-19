import pandas as pd
import numpy as np
import joblib
import re
from datetime import datetime

# --- 0. Chargement du Modèle et des Données Sources (Identique) ---

try:
    model = joblib.load('best_football_model_v8.joblib')
    print("Modèle IA 'best_football_model_v8.joblib' chargé avec succès.")
except FileNotFoundError:
    print("ERREUR : Le fichier 'best_football_model_v8.joblib' n'a pas été trouvé.")
    exit()

try:
    results_df = pd.read_csv('True_Version/results.csv')
    fifa_ranking_df = pd.read_csv('True_Version/fifa_ranking.csv')
    print("Données brutes chargées.")
except FileNotFoundError:
    print("ERREUR : Assurez-vous que les fichiers 'results.csv' et 'fifa_ranking.csv' sont dans le dossier 'True_Version'.")
    exit()

# --- 1. Dictionnaire de Traduction (Identique) ---
country_translation_map = {
    "Antigua-et-Barbuda": "Antigua and Barbuda", "États-Unis": "United States", "Uruguay": "Uruguay",
    "Colombie": "Colombia", "Australie": "Australia", "Équateur": "Ecuador", "Nouvelle-Zélande": "New Zealand",
    "Mexique": "Mexico", "Paraguay": "Paraguay", "Venezuela": "Venezuela", "Canada": "Canada",
    "Costa Rica": "Costa Rica", "Guatemala": "Guatemala", "Surinam": "Suriname", "Haïti": "Haiti",
    "Nicaragua": "Nicaragua", "Jamaïque": "Jamaica", "Curaçao": "Curacao", "Panama": "Panama",
    "Salvador": "El Salvador", "Trinité-et-Tobago": "Trinidad and Tobago", "Bermudes": "Bermudes"
}

def translate_country(french_name):
    return country_translation_map.get(french_name, french_name)

# --- 2. Fonctions de Calcul de Features (Identiques) ---
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

# --- 3. Extraction des Données depuis le HTML (Amélioré) ---

html_content = """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Paris – BetFoot</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">

  <style>
    :root{
      --red:#E30613;
      --dark:#121212;
      --muted:#6b6b6b;
      --bg:#0f1113;
      --card:#151617;
      --white:#ffffff;
      --gain:#1db954;
      --glass: rgba(255,255,255,0.03);
      font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
    }
    *{box-sizing:border-box}
    body{margin:0;background:linear-gradient(180deg,var(--bg),#071014);color:var(--white);min-height:100vh}
    header{display:flex;align-items:center;justify-content:space-between;padding:18px 24px}
    .brand{display:flex;align-items:center;gap:12px}
    .logo{width:46px;height:46px;border-radius:8px;background:var(--red);display:flex;align-items:center;justify-content:center;font-weight:700}
    nav{display:flex;gap:18px;align-items:center}
    nav a{color:rgba(255,255,255,0.9);text-decoration:none;font-weight:600}
    nav a.active{color:var(--red);border-bottom:2px solid var(--red)}

    .container{max-width:1200px;margin:18px auto;padding:0 18px;display:flex;gap:18px}
    .left{flex:1}
    aside.ticket{width:360px;background:linear-gradient(180deg, #0b0c0d, #0e0f11);border-radius:12px;padding:16px;min-height:220px;align-self:flex-start}

    .card{
      background:var(--card);
      padding:18px;
      border-radius:12px;
      margin-bottom:18px;
      box-shadow:0 0 14px rgba(0,0,0,0.25);
    }

    .match-line{
      display:flex;
      align-items:center;
      justify-content:space-between;
      margin-bottom:10px;
    }

    .teams{
      font-size:18px;
      font-weight:700;
    }

    .odds{display:flex;gap:8px}
    .odd{
      background:rgba(255,255,255,0.04);
      padding:8px 14px;
      border-radius:8px;
      cursor:pointer;
      font-weight:600;
      user-select:none;
    }
    .odd.selected{outline:2px solid rgba(255,255,255,0.08);transform:translateY(-2px)}

    .ticket-btn{
      background:var(--red);
      padding:10px 14px;
      border:none;
      color:white;
      border-radius:8px;
      cursor:pointer;
      font-weight:700;
      margin-top:10px;
    }

    /* ticket */
    .ticket h3{margin:0 0 8px 0}
    .selection{display:flex;align-items:center;justify-content:space-between;padding:8px 0;border-bottom:1px dashed rgba(255,255,255,0.04)}
    .stake{display:flex;gap:8px;align-items:center;margin-top:12px}
    input[type=number]{background:transparent;border:1px solid rgba(255,255,255,0.06);padding:8px;border-radius:8px;color:var(--white);width:120px}
    .potential{font-weight:700;color:var(--gain)}
    .small-btn{background:transparent;border:1px solid rgba(255,255,255,0.06);padding:8px 10px;border-radius:8px;color:var(--white);cursor:pointer}

    .warning{background:rgba(227,6,19,0.08);color:var(--red);padding:8px;border-radius:8px;margin-top:12px;font-size:13px}

    @media (max-width:1000px){
      .container{flex-direction:column;padding:0 12px}
      aside.ticket{width:100%}
    }

  </style>
</head>
<body>

<header>
  <div class="brand" style="max-width:1200px;margin:auto;display:flex;align-items:center;justify-content:space-between;">
    <div style="display:flex;align-items:center;gap:12px">
      <div class="logo">BF</div>
      <div>
        <div style="font-weight:700">BetFoot</div>
        <div style="font-size:12px;color:var(--muted)">Reproduction Betclic — Version Foot</div>
      </div>
    </div>
    <nav>
      <a href="interface.html">Accueil</a>
      <a href="live.html">Matchs en direct</a>
      <a href="paris.html" class="active">Paris</a>
      <a href="resultats.html">Résultats</a>
      <a href="compte.html">Mon compte</a>
    </nav>
  </div>
</header>

<div class="container">
  <div class="left">
    <h2 style="margin-top:6px">Paris disponibles</h2>
    <p style="color:var(--muted)"></p>

    <!-- ---------- AMICAUX (capture 1) ---------- -->
    <div class="card match-row" data-id="m_a1" data-home="Antigua-et-Barbuda" data-away="Guyane" data-odds="1.90,3.10,3.70">
      <div class="match-line">
        <div class="teams">Antigua-et-Barbuda <span style="color:var(--muted)">vs</span> Guyane</div>
        <div class="odds">
          <div class="odd" data-odd="1.90">1.90</div>
          <div class="odd" data-odd="3.10">3.10</div>
          <div class="odd" data-odd="3.70">3.70</div>
        </div>
      </div>
      <button class="ticket-btn add-first">Ajouter au ticket</button>
    </div>

    <div class="card match-row" data-id="m_a2" data-home="États-Unis" data-away="Uruguay" data-odds="2.00,3.25,3.40">
      <div class="match-line">
        <div class="teams">États-Unis <span style="color:var(--muted)">vs</span> Uruguay</div>
        <div class="odds">
          <div class="odd" data-odd="2.00">2.00</div>
          <div class="odd" data-odd="3.25">3.25</div>
          <div class="odd" data-odd="3.40">3.40</div>
        </div>
      </div>
      <button class="ticket-btn add-first">Ajouter au ticket</button>
    </div>

    <div class="card match-row" data-id="m_a3" data-home="Colombie" data-away="Australie" data-odds="1.65,3.40,4.60">
      <div class="match-line">
        <div class="teams">Colombie <span style="color:var(--muted)">vs</span> Australie</div>
        <div class="odds">
          <div class="odd" data-odd="1.65">1.65</div>
          <div class="odd" data-odd="3.40">3.40</div>
          <div class="odd" data-odd="4.60">4.60</div>
        </div>
      </div>
      <button class="ticket-btn add-first">Ajouter au ticket</button>
    </div>

    <div class="card match-row" data-id="m_a4" data-home="Équateur" data-away="Nouvelle-Zélande" data-odds="1.55,3.70,5.10">
      <div class="match-line">
        <div class="teams">Équateur <span style="color:var(--muted)">vs</span> Nouvelle-Zélande</div>
        <div class="odds">
          <div class="odd" data-odd="1.55">1.55</div>
          <div class="odd" data-odd="3.70">3.70</div>
          <div class="odd" data-odd="5.10">5.10</div>
        </div>
      </div>
      <button class="ticket-btn add-first">Ajouter au ticket</button>
    </div>

    <div class="card match-row" data-id="m_a5" data-home="Mexique" data-away="Paraguay" data-odds="2.10,3.05,3.25">
      <div class="match-line">
        <div class="teams">Mexique <span style="color:var(--muted)">vs</span> Paraguay</div>
        <div class="odds">
          <div class="odd" data-odd="2.10">2.10</div>
          <div class="odd" data-odd="3.05">3.05</div>
          <div class="odd" data-odd="3.25">3.25</div>
        </div>
      </div>
      <button class="ticket-btn add-first">Ajouter au ticket</button>
    </div>

    <div class="card match-row" data-id="m_a6" data-home="Venezuela" data-away="Canada" data-odds="2.50,3.15,2.70">
      <div class="match-line">
        <div class="teams">Venezuela <span style="color:var(--muted)">vs</span> Canada</div>
        <div class="odds">
          <div class="odd" data-odd="2.50">2.50</div>
          <div class="odd" data-odd="3.15">3.15</div>
          <div class="odd" data-odd="2.70">2.70</div>
        </div>
      </div>
      <button class="ticket-btn add-first">Ajouter au ticket</button>
    </div>

    <!-- ---------- CONCACAF (capture 2) ---------- -->
    <div style="height:8px"></div>

    <div class="card match-row" data-id="m_c1" data-home="Costa Rica" data-away="Honduras" data-odds="2.20,3.00,3.00">
      <div class="match-line">
        <div class="teams">Costa Rica <span style="color:var(--muted)">vs</span> Honduras</div>
        <div class="odds">
          <div class="odd" data-odd="2.20">2.20</div>
          <div class="odd" data-odd="3.00">3.00</div>
          <div class="odd" data-odd="3.00">3.00</div>
        </div>
      </div>
      <button class="ticket-btn add-first">Ajouter au ticket</button>
    </div>

    <div class="card match-row" data-id="m_c2" data-home="Guatemala" data-away="Surinam" data-odds="1.70,3.40,4.20">
      <div class="match-line">
        <div class="teams">Guatemala <span style="color:var(--muted)">vs</span> Surinam</div>
        <div class="odds">
          <div class="odd" data-odd="1.70">1.70</div>
          <div class="odd" data-odd="3.40">3.40</div>
          <div class="odd" data-odd="4.20">4.20</div>
        </div>
      </div>
      <button class="ticket-btn add-first">Ajouter au ticket</button>
    </div>

    <div class="card match-row" data-id="m_c3" data-home="Haïti" data-away="Nicaragua" data-odds="1.85,3.20,3.90">
      <div class="match-line">
        <div class="teams">Haïti <span style="color:var(--muted)">vs</span> Nicaragua</div>
        <div class="odds">
          <div class="odd" data-odd="1.85">1.85</div>
          <div class="odd" data-odd="3.20">3.20</div>
          <div class="odd" data-odd="3.90">3.90</div>
        </div>
      </div>
      <button class="ticket-btn add-first">Ajouter au ticket</button>
    </div>

    <div class="card match-row" data-id="m_c4" data-home="Jamaïque" data-away="Curaçao" data-odds="2.10,3.00,3.15">
      <div class="match-line">
        <div class="teams">Jamaïque <span style="color:var(--muted)">vs</span> Curaçao</div>
        <div class="odds">
          <div class="odd" data-odd="2.10">2.10</div>
          <div class="odd" data-odd="3.00">3.00</div>
          <div class="odd" data-odd="3.15">3.15</div>
        </div>
      </div>
      <button class="ticket-btn add-first">Ajouter au ticket</button>
    </div>

    <div class="card match-row" data-id="m_c5" data-home="Panama" data-away="Salvador" data-odds="1.55,3.60,5.20">
      <div class="match-line">
        <div class="teams">Panama <span style="color:var(--muted)">vs</span> Salvador</div>
        <div class="odds">
          <div class="odd" data-odd="1.55">1.55</div>
          <div class="odd" data-odd="3.60">3.60</div>
          <div class="odd" data-odd="5.20">5.20</div>
        </div>
      </div>
      <button class="ticket-btn add-first">Ajouter au ticket</button>
    </div>

    <div class="card match-row" data-id="m_c6" data-home="Trinité-et-Tobago" data-away="Bermudes" data-odds="1.95,3.25,3.60">
      <div class="match-line">
        <div class="teams">Trinité-et-Tobago <span style="color:var(--muted)">vs</span> Bermudes</div>
        <div class="odds">
          <div class="odd" data-odd="1.95">1.95</div>
          <div class="odd" data-odd="3.25">3.25</div>
          <div class="odd" data-odd="3.60">3.60</div>
        </div>
      </div>
      <button class="ticket-btn add-first">Ajouter au ticket</button>
    </div>

  </div>

  <!-- TICKET À DROITE (option A) -->
  <aside class="ticket" id="ticket">
    <h3>Ticket de pari</h3>
    <div id="selections">
      <div style="color:var(--muted);font-size:13px">Aucune sélection</div>
    </div>

    <div class="stake">
      <label for="stake">Mise (crédits)</label>
      <input type="number" id="stake" min="1" value="10">
    </div>
    <div style="margin-top:8px">Gain potentiel : <span class="potential" id="potential">0</span> crédits</div>

    <button class="ticket-btn" id="place-bet" style="margin-top:12px">Placer le pari</button>

    <div style="display:flex;gap:8px;margin-top:12px;align-items:center;justify-content:space-between">
      <div>
        <div style="font-size:13px;color:var(--muted)">Solde virtuel</div>
        <div style="font-weight:700"><span id="balance">1000</span> crédits</div>
      </div>
      <div style="display:flex;flex-direction:column;gap:8px">
        <button class="small-btn" id="recharge">+500</button>
        <button class="small-btn" id="history-btn">Historique</button>
      </div>
    </div>

    <div id="history" style="margin-top:12px;display:none"></div>

    <div class="warning">Transactions factices uniquement – aucun argent réel.</div>
  </aside>

</div>

<script>
  // Ticket logic (adapted from interface.html)
  const selections = [];
  const selectionsEl = document.getElementById('selections');
  const potentialEl = document.getElementById('potential');
  const stakeInput = document.getElementById('stake');
  const balanceEl = document.getElementById('balance');
  const historyEl = document.getElementById('history');
  const placeBetBtn = document.getElementById('place-bet');

  // Persisted data
  let balance = localStorage.getItem('userBalance') !== null ? Number(localStorage.getItem('userBalance')) : 1000;
  let history = [];

  // initialize balance display
  function initBalance(){
    balanceEl.textContent = balance;
  }
  initBalance();

  // Render selections in ticket
  function renderSelections(){
    selectionsEl.innerHTML = '';
    if(selections.length===0){
      selectionsEl.innerHTML = '<div style="color:var(--muted);font-size:13px">Aucune sélection</div>';
      potentialEl.textContent = '0';
      return;
    }
    selections.forEach((s, idx)=>{
      const div = document.createElement('div');
      div.className='selection';
      div.innerHTML = `<div style="min-width:0">
        <div style="font-size:14px"><strong>${s.home}</strong> vs <strong>${s.away}</strong></div>
        <div style="font-size:13px;color:var(--muted)">Cote choisie: ${s.odd}</div>
      </div>
      <div style="display:flex;flex-direction:column;align-items:flex-end">
        <button class="small-btn remove" data-idx="${idx}">Supprimer</button>
      </div>`;
      selectionsEl.appendChild(div);
    });
    updatePotential();
  }

  function updatePotential(){
    const stake = Number(stakeInput.value) || 0;
    const combined = selections.reduce((acc,s)=>acc * Number(s.odd), 1);
    const pot = (stake * combined).toFixed(2);
    potentialEl.textContent = pot;
  }

  // Add a selection (prevent duplicates)
  function addSelection(obj){
    if(selections.some(s => s.id === obj.id)){
      alert('Match déjà dans le ticket');
      return false;
    }
    selections.push(obj);
    renderSelections();
    return true;
  }

  // Add-first-btn behavior: add first odd of the match
  document.querySelectorAll('.add-first').forEach(btn=>{
    btn.addEventListener('click', e=>{
      const row = e.target.closest('.match-row');
      const id = row.dataset.id;
      const home = row.dataset.home;
      const away = row.dataset.away;
      const odds = row.dataset.odds.split(',');
      const odd = odds[0];
      addSelection({id,home,away,odd});
    });
  });

  // Click on individual odd to choose it and add
  document.querySelectorAll('.odd').forEach(el=>{
    el.addEventListener('click', e=>{
      const odd = e.currentTarget.dataset.odd || e.currentTarget.textContent;
      const row = e.currentTarget.closest('.match-row');
      const id = row.dataset.id;
      const home = row.dataset.home;
      const away = row.dataset.away;

      // Visual feedback: toggle selected on that row's odds
      row.querySelectorAll('.odd').forEach(o=>o.classList.remove('selected'));
      e.currentTarget.classList.add('selected');

      if(selections.some(s=>s.id===id)){
        alert('Match déjà dans le ticket — supprimez-le pour changer la cote.');
        return;
      }
      addSelection({id,home,away,odd});
    });
  });

  // Remove selection
  selectionsEl.addEventListener('click', e=>{
    if(e.target.matches('button.remove')){
      const idx = Number(e.target.dataset.idx);
      if(!isNaN(idx)){
        selections.splice(idx,1);
        renderSelections();
      }
    }
  });

  stakeInput.addEventListener('input', updatePotential);

  // Place bet (fictive)
  placeBetBtn.addEventListener('click', () => {
    const stake = Number(stakeInput.value) || 0;
    if (selections.length === 0) {
        alert('Aucune sélection dans le ticket.');
        return;
    }
    if (stake <= 0) {
        alert('Entrez une mise valide.');
        return;
    }
    if (stake > balance) {
        alert('Solde insuffisant. Rechargez vos crédits fictifs.');
        return;
    }

    // Demander les scores pour chaque match
    const predictions = [];

    for (let s of selections) {
        const score = prompt(`Score prévu pour ${s.home} vs ${s.away}\nFormat: 2-1`);
        if (!score || !score.includes("-")) {
            alert("Format invalide. Exemple: 2-1");
            return;
        }

        const [homeScore, awayScore] = score.split("-").map(n => Number(n.trim()));
        if (isNaN(homeScore) || isNaN(awayScore)) {
            alert("Scores invalides.");
            return;
        }

        predictions.push({
            matchId: s.id,
            home: s.home,
            away: s.away,
            predictedHome: homeScore,
            predictedAway: awayScore,
            odd: s.odd
        });
    }

    // Mise à jour du solde
    const pot = Number(potentialEl.textContent) || 0;
    balance -= stake;
    localStorage.setItem('userBalance', balance);
    balanceEl.textContent = balance;

    // Enregistrer le pari dans l'historique
    const record = {
        date: new Date().toLocaleString(),
        selections: JSON.parse(JSON.stringify(selections)),
        predictions: predictions,
        stake: stake,
        potential: pot
    };

    history.unshift(record);
    localStorage.setItem('betHistory', JSON.stringify(history));

    // Sauvegarder les prédictions dans un stockage séparé pour "Résultats"
    let allPred = localStorage.getItem('matchPredictions');
    allPred = allPred ? JSON.parse(allPred) : [];

    allPred.push({
        betDate: record.date,
        predictions: predictions
    });

    localStorage.setItem('matchPredictions', JSON.stringify(allPred));

    alert('Pari placé ! Prévisions de score enregistrées.');

    // Reset du ticket
    selections.length = 0;
    renderSelections();
    renderHistory();
});


  // Recharge button
  document.getElementById('recharge').addEventListener('click', ()=>{
    balance += 500;
    localStorage.setItem('userBalance', balance);
    balanceEl.textContent = balance;
    alert('+500 crédits fictifs ajoutés.');
  });

  // History toggle
  document.getElementById('history-btn').addEventListener('click', ()=>{
    if(historyEl.style.display === 'none'){
      historyEl.style.display = 'block';
      renderHistory();
    } else {
      historyEl.style.display = 'none';
    }
  });

  function renderHistory(){
    // try load saved history
    const stored = localStorage.getItem('betHistory');
    if(stored) history = JSON.parse(stored);
    if(history.length===0){
      historyEl.innerHTML = '<div style="color:var(--muted);font-size:13px">Aucun pari effectué</div>';
      return;
    }
    historyEl.innerHTML = history.map((h, idx) => {
      const sel = h.selections.map(s=>`${s.home} vs ${s.away} (${s.odd})`).join('<br>');
      return `<div style="padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.03)">
                <div style="font-size:13px;color:var(--muted)">${h.date}</div>
                <div style="font-weight:700">Mise: ${h.stake} crédits — Potentiel: ${h.potential} crédits</div>
                <div style="font-size:13px;margin-top:6px">${sel}</div>
              </div>`;
    }).join('');
  }

  // Initialize from localStorage
  document.addEventListener('DOMContentLoaded', ()=>{
    const storedBalance = localStorage.getItem('userBalance');
    if(storedBalance !== null){
      balance = Number(storedBalance);
      balanceEl.textContent = balance;
    }
    const storedHistory = localStorage.getItem('betHistory');
    if(storedHistory){
      history = JSON.parse(storedHistory);
    }
    renderSelections();
  });

  // Keep balance in sync if changed elsewhere (compte.html)
  window.addEventListener('storage', (e)=>{
    if(e.key === 'userBalance'){
      balance = Number(e.newValue);
      balanceEl.textContent = balance;
    }
    if(e.key === 'betHistory'){
      const stored = localStorage.getItem('betHistory');
      history = stored ? JSON.parse(stored) : [];
      renderHistory();
    }
  });

</script>

</body>
</html>

"""

# Regex pour extraire les informations des matchs, y compris les cotes
match_pattern = re.compile(
    r'<div class="card match-row"[^>]*data-home="([^"]*)"[^>]*data-away="([^"]*)"[^>]*data-odds="([^"]*)"',
    re.IGNORECASE
)

matches = re.findall(match_pattern, html_content)

if not matches:
    print("ERREUR : Aucun match trouvé dans le code HTML fourni.")
    exit()

# --- 4. Logique de Pari Automatisé avec Intégration des Cotes ---

# NOUVEAU : Poids pour la combinaison des prédictions
# 0.7 pour l'IA, 0.3 pour les cotes du marché. Vous pouvez ajuster ces valeurs.
WEIGHT_MODEL = 0.7
WEIGHT_ODDS = 0.3

def odds_to_probabilities(odds):
    """Convertit une liste de cotes en probabilités normalisées."""
    # La probabilité implicite est 1 / cote
    implied_probs = [1 / odd for odd in odds]
    total_implied_prob = sum(implied_probs)
    # Normaliser pour que la somme soit égale à 1
    normalized_probs = [p / total_implied_prob for p in implied_probs]
    return normalized_probs

print("\n" + "="*50)
print("     DÉMARRAGE DU BOT DE PARI AUTOMATISÉ (AVEC INTÉGRATION DES COTES)")
print(f"     Pondération : {WEIGHT_MODEL*100:.0f}% IA / {WEIGHT_ODDS*100:.0f}% Cotes du Marché")
print("="*50)

bankroll = 1000.0
base_stake = 10.0
result_map = {0: 'Match Nul', 1: 'Victoire Domicile', 2: 'Victoire Extérieur'}

for i, match_data in enumerate(matches):
    home_team_raw, away_team_raw, odds_str = match_data
    
    # Nettoyer et parser les noms des équipes et les cotes
    home_team = translate_country(home_team_raw.strip())
    away_team = translate_country(away_team_raw.strip())
    odds = list(map(float, odds_str.split(',')))

    print(f"\n--- Analyse du match {i+1}/{len(matches)} : {home_team_raw} vs {away_team_raw} ---")

    # Préparation des features pour le modèle IA
    today = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
    home_rank = get_fifa_rank(home_team, today)
    away_rank = get_fifa_rank(away_team, today)

    if pd.isna(home_rank) or pd.isna(away_rank):
        print(f"ERREUR : Impossible de trouver le classement FIFA pour '{home_team}' ou '{away_team}'. Match ignoré.")
        continue

    match_features = pd.DataFrame({
        'home_team_rank': [home_rank], 'away_team_rank': [away_rank],
        'home_team_form': [calculate_recent_form(home_team, today)],
        'away_team_form': [calculate_recent_form(away_team, today)],
        'home_goal_diff': [calculate_goal_difference(home_team, today)],
        'away_goal_diff': [calculate_goal_difference(away_team, today)],
        'tournament_importance': [1], # Par défaut, considéré comme match amical
        'is_neutral': [0], # Par défaut, pas de terrain neutre
        'h2h_points_diff': [get_h2h_points_diff(home_team, away_team, today)]
    })

    # Prédiction de l'IA
    model_probs = model.predict_proba(match_features)[0]
    
    # NOUVEAU : Conversion des cotes en probabilités du marché
    market_probs = odds_to_probabilities(odds)

    # NOUVEAU : Combinaison des probabilités
    # On fait une moyenne pondérée des probabilités de l'IA et du marché
    final_probs = (model_probs * WEIGHT_MODEL) + (market_probs * WEIGHT_ODDS)
    
    # Identifier le résultat le plus probable et sa probabilité finale
    predicted_index = np.argmax(final_probs)
    confidence = final_probs[predicted_index]
    
    # Calcul de la mise proportionnelle à la confiance finale
    stake = base_stake * confidence

    # Vérifier si le solde est suffisant
    if stake > bankroll:
        print(f"SOLDE INSUFFISANT. Mise requise ({stake:.2f}) supérieure au solde ({bankroll:.2f}). Pari ignoré.")
        continue

    # "Placer le pari"
    bankroll -= stake
    
    # Affichage des résultats détaillés
    print(f"Prédiction de l'IA : {result_map[np.argmax(model_probs)]} (Confiance: {np.max(model_probs):.2%})")
    print(f"Probabilités du Marché : Victoire Domicile: {market_probs[1]:.2%} | Nul: {market_probs[0]:.2%} | Victoire Extérieur: {market_probs[2]:.2%}")
    print(f"Probabilités FINALES (combinées) : Victoire Domicile: {final_probs[1]:.2%} | Nul: {final_probs[0]:.2%} | Victoire Extérieur: {final_probs[2]:.2%}")
    print(f"Confiance finale : {confidence:.2%}")
    print(f"MISE AUTOMATISÉE : {stake:.2f} crédits sur {result_map[predicted_index]}")
    print(f"Nouveau solde : {bankroll:.2f} crédits")

# --- 5. Résumé Final ---
print("\n" + "="*50)
print("     SIMULATION TERMINÉE")
print("="*50)
print(f"Solde initial : 1000.00 crédits")
print(f"Solde final : {bankroll:.2f} crédits")
print(f"Résultat net : {bankroll - 1000:.2f} crédits")
if bankroll > 1000:
    print("🎉 Le bot est globalement gagnant !")
else:
    print("😞 Le bot est globalement perdant.")
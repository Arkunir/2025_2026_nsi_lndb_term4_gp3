#!/usr/bin/env python3
# fetch_last_100_international_matches.py
# Nécessite: requests, beautifulsoup4, pandas
# Optionnel (si la page est JS-heavy): selenium, webdriver-manager

import time
import csv
import sys
from typing import List, Tuple
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Si la méthode requests échoue à cause de JS, régler à True et installer selenium
USE_SELENIUM = False

# URL ciblée (page "Friendly International" ou page résultats internationaux)
FLASH_URL = "https://www.flashscore.com/football/world/friendly/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def fetch_with_requests(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.text

def parse_flashscore_html(html: str) -> List[Tuple[str,str,str]]:
    """
    Retourne une liste de tuples (date, home_team home_score-away_score away_team).
    Le format exact d'extraction dépend du DOM du site.
    """
    soup = BeautifulSoup(html, "html.parser")
    matches = []

    # Flashscore organise souvent les journées par blocs. Cherchons les lignes de match.
    # NOTE: le sélecteur ci-dessous peut nécessiter adaptation si Flashscore change son HTML.
    for match_block in soup.select(".event__match"):
        try:
            date_block = match_block.find_previous(class_="event__time")  # approximatif
            # Extraire équipes et score
            home = match_block.select_one(".event__participant--home")
            away = match_block.select_one(".event__participant--away")
            score = match_block.select_one(".event__score--home")  # approximatif
            # alternative pour score
            score_full = match_block.select_one(".event__scores")
            if home and away:
                home_text = home.get_text(strip=True)
                away_text = away.get_text(strip=True)
                # chercher score
                score_text = match_block.get_text(separator=" | ", strip=True)
                # Nettoyage simple : essayer d'extraire pattern 'X - Y'
                import re
                m = re.search(r"(\d+)\s*[-–]\s*(\d+)", score_text)
                if m:
                    hs, as_ = m.group(1), m.group(2)
                    # date approximative - flashscore often shows day headings; fallback to 'unknown'
                    # We'll extract nearest date heading if present
                    date_heading = match_block.find_parent()
                    date_str = "unknown"
                    # attempt to find a date label above
                    prev = match_block.find_previous(lambda tag: tag.name in ["div","span"] and ("date" in (tag.get("class") or []) or "event__time" in (tag.get("class") or [])))
                    if prev:
                        date_str = prev.get_text(strip=True)
                    matches.append((date_str, f"{home_text} {hs}-{as_} {away_text}"))
        except Exception:
            continue
        if len(matches) >= 100:
            break

    return matches

def save_matches_csv(matches: List[Tuple[str,str]] , out="last_100_international_matches.csv"):
    df = pd.DataFrame(matches, columns=["date","scoreline"])
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} matches to {out}")

# If JS page: selenium path (optional)
def fetch_with_selenium(url: str) -> str:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=opts)
    driver.get(url)
    time.sleep(5)  # attendre le chargement JS
    html = driver.page_source
    driver.quit()
    return html

def main():
    url = FLASH_URL
    print("Fetching international matches from:", url)
    try:
        if USE_SELENIUM:
            html = fetch_with_selenium(url)
        else:
            html = fetch_with_requests(url)
    except Exception as e:
        print("Erreur lors du fetch initial:", e)
        if not USE_SELENIUM:
            print("Essai avec Selenium...")
            try:
                html = fetch_with_selenium(url)
            except Exception as e2:
                print("Echec Selenium:", e2)
                sys.exit(1)
        else:
            sys.exit(1)

    matches = parse_flashscore_html(html)
    # Si pas assez de résultats extraits, on peut essayer une autre source (ESPN)
    if len(matches) < 100:
        print(f"Extrait seulement {len(matches)} matches — tentative via ESPN...")
        # appel simple ESPN (structure peut varier fortement)
        try:
            espn_html = fetch_with_requests("https://www.espn.com/soccer/scoreboard/_/league/fifa.friendly")
            # tentative d'extraction basique (à adapter)
            soup2 = BeautifulSoup(espn_html, "html.parser")
            for m in soup2.select(".scoreboard"):
                if len(matches) >= 100:
                    break
                # Extract teams and score in a generic way
                txt = m.get_text(" | ", strip=True)
                matches.append(("unknown", txt))
        except Exception:
            pass

    # Enfin tronquer à 100 si nécessaire
    matches = matches[:100]
    if not matches:
        print("Aucun match trouvé automatiquement. Le sélecteur HTML doit être adapté à la page cible.")
        sys.exit(1)

    # Sauvegarde et affichage
    save_matches_csv(matches)
    print("--- Aperçu ---")
    for i, (d, s) in enumerate(matches[:20], 1):
        print(f"{i:02d}. {d} — {s}")

if __name__ == "__main__":
    main()

BetFoot: Plateforme de Prédiction et de Simulation de Paris Sportifs

Ce projet est une plateforme qui utilise un modèle de machine learning pour prédire les résultats de matchs de football et simuler des paris sportifs via une interface web.

Fonctionnalités principales

Prédiction des résultats de matchs (victoire domicile, nul, victoire extérieur) à l'aide d'un modèle d'intelligence artificielle.
Combinaison des prédictions de l'IA avec les cotes du marché pour une analyse améliorée.
Simulation de paris automatiques où le script place des mises en fonction de sa confiance en la prédiction.
Calcul des gains et des pertes basé sur les résultats réels des matchs.
Interface web interactive pour consulter les matchs, placer des paris virtuels et gérer un compte utilisateur.

Quelques détails sur le fonctionnement

Le script Python analyse une liste de matchs et de cotes.
Pour chaque match, il extrait des données statistiques pertinentes comme le classement FIFA, la forme récente des équipes, ou la différence de buts à partir de fichiers CSV.
Ces données sont ensuite fournies à un modèle de machine learning pré-entraîné pour générer des probabilités.
Ces probabilités sont combinées avec celles dérivées des cotes du marché.
Le script place alors un pari sur l'issue la plus probable et, une fois les résultats réels connus, il calcule le gain ou la perte pour mettre à jour un solde virtuel.

Architecture du projet

Le projet est structuré en trois couches principales :
un backend Python pour la logique de prédiction et de simulation,
un frontend HTML/CSS/JavaScript pour l'interface utilisateur,
et une couche de données composée de fichiers CSV (résultats, classements FIFA) et du modèle pré-entraîné.

Technologies utilisées

Backend: Python, Pandas, NumPy, Scikit-learn (via le modèle), Joblib.
Frontend: HTML5, CSS3, JavaScript.
Stockage de données: Fichiers CSV, localStorage du navigateur.

Imports Python utilisés dans le projet

import pandas as pd
Utilisé pour la manipulation et l'analyse de données, notamment pour lire les fichiers CSV de résultats et de classements FIFA, et pour créer les structures de données (DataFrame) nécessaires au modèle.
import numpy as np
Utilisé pour les calculs numériques, en particulier pour manipuler les tableaux de probabilités générés par le modèle de machine learning.
import joblib
Utilisé pour charger le modèle de machine learning pré-entraîné qui est sauvegardé dans un fichier .joblib.
import re
Utilisé pour les expressions régulières, permettant d'extraire de manière structurée les informations des matchs (équipes, cotes) directement depuis le code HTML fourni.
from datetime import datetime
Utilisé pour obtenir la date actuelle, ce qui est nécessaire pour rechercher les données les plus récentes (comme le classement FIFA ou la forme de l'équipe) avant un match donné.

Présentation du projet

https://chat.z.ai/space/w0suc9gf57u1-ppt
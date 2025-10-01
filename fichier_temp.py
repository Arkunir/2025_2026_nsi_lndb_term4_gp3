

# === client.py ===
"""
Client console pour Last-Man-Standing
Usage: python3 client.py <IP_du_serveur>

Le client affiche les infos et attend les invitations à répondre. Il gère les nombres interdits quand le serveur les envoie.
"""

import socket
import threading
import json
import sys
import time

if len(sys.argv) < 2:
    print('Usage: python3 client.py <IP_du_serveur>')
    sys.exit(1)

SERVER = sys.argv[1]
PORT = 5000

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.connect((SERVER, PORT))
except Exception as e:
    print('Impossible de se connecter:', e)
    sys.exit(1)

name = input('Tape ton pseudo: ').strip()
if not name:
    name = 'Joueur'

# envoie du join
sobj = {'type':'join','name': name}
sock.sendall((json.dumps(sobj, ensure_ascii=False)+'\n').encode())

# shared state for the input thread
current_forbidden = set()
answer_event = threading.Event()
answer_value = {'v': None}
expecting_answer = {'val': False}


def send_line(obj):
    try:
        sock.sendall((json.dumps(obj, ensure_ascii=False)+'\n').encode())
    except Exception as e:
        print('Erreur envoi:', e)


def input_thread_fn():
    # thread permanent qui attend quand expecting_answer est vrai
    while True:
        # attendre qu'on s'attende à une réponse
        while not expecting_answer['val']:
            time.sleep(0.1)
        # on attend l'entrée, mais on laisse le main gérer le timeout
        try:
            s = input()  # l'utilisateur tape quelque chose
        except EOFError:
            s = ''
        if not expecting_answer['val']:
            continue
        s = s.strip()
        # validation
        try:
            v = int(s)
            if v < 0 or v > 100:
                print('Entier entre 0 et 100 requis. Réessayez:')
                continue
            if v in current_forbidden:
                print('Nombre interdit pour cette manche. Réessayez:')
                continue
            # ok
            answer_value['v'] = v
            answer_event.set()
        except ValueError:
            print('Entier requis. Réessayez:')
            continue


def listen_thread():
    buf = b""
    while True:
        try:
            data = sock.recv(1024)
            if not data:
                print('\nConnexion au serveur perdue')
                break
            buf += data
            while b"\n" in buf:
                line, _, buf = buf.partition(b"\n")
                try:
                    m = json.loads(line.decode())
                except Exception:
                    continue
                handle_server_message(m)
        except Exception as e:
            print('Erreur receive:', e)
            break


def handle_server_message(m):
    t = m.get('type')
    if t == 'info':
        print('\n[INFO]', m.get('text'))
    elif t == 'start_game':
        mult = m.get('multiplier')
        it = m.get('initial_time')
        print(f"\n[GAME START] Multiplicateur initial = {mult}, temps par round = {it}s")
    elif t == 'round_start':
        r = m.get('round')
        mult = m.get('multiplier')
        time_allowed = m.get('time')
        forbidden = m.get('forbidden', [])
        # set forbidden
        global current_forbidden
        current_forbidden.clear()
        for x in forbidden:
            current_forbidden.add(x)
        print(f"\n[ROUND {r}] multiplier={mult}, temps={time_allowed}s")
        if forbidden:
            print('Nombres interdits cette manche:', sorted(forbidden))
        print('Tapez un entier entre 0 et 100 et appuyez sur entrée:')
        # prepare to capture input
        answer_event.clear()
        expecting_answer['val'] = True
        # wait for answer or timeout
        start = time.time()
        while time.time() - start < time_allowed:
            if answer_event.wait(timeout=0.1):
                # got answer
                v = answer_value['v']
                send_line({'type':'answer','value': v})
                answer_value['v'] = None
                answer_event.clear()
                break
        else:
            print('\nTemps écoulé, pas de réponse envoyée')
        expecting_answer['val'] = False
        current_forbidden.clear()
    elif t == 'round_result':
        rnd = m.get('round')
        target = m.get('target')
        closest = m.get('closest')
        eliminated = m.get('eliminated')
        lives = m.get('lives', [])
        active = m.get('active_rules', [])
        new_rules = m.get('new_rules', [])
        print(f"\n[RESULTAT Manche {rnd}]: cible = {target}")
        print('Plus proches:', closest)
        if eliminated:
            print('Eliminés cette manche:', eliminated)
        if new_rules:
            print('Nouvelles règles activées:', new_rules)
        print('Vies:')
        for p in lives:
            print(f" - {p['name']}: {p['lives']}")
        if active:
            print('Règles actives:', active)
    else:
        print('\n[MSG]', m)


# lancer threads
threading.Thread(target=input_thread_fn, daemon=True).start()
threading.Thread(target=listen_thread, daemon=True).start()

# boucle principale: rester vivant
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print('Déconnexion...')
    sock.close()

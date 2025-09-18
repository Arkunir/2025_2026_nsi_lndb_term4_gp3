# interface.py
# Reproduction Tkinter de interface.html (page principale)
import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
from datetime import datetime

STORAGE = "storage.json"

MATCHES = [
    {"id": "m1", "home": "France", "away": "Brésil", "odds": [1.85, 3.20, 4.10], "date": "2025-10-12T21:00:00"},
    {"id": "m2", "home": "Argentine", "away": "Portugal", "odds": [2.10, 3.00, 3.50], "date": "2025-11-20T18:45:00"},
    {"id": "m3", "home": "Espagne", "away": "Allemagne", "odds": [1.95, 3.40, 4.00], "date": "2025-12-05T17:30:00"},
    {"id": "m4", "home": "Italie", "away": "Pays-Bas", "odds": [2.40, 3.10, 2.80], "date": "2026-03-28T20:00:00"},
]

def load_storage():
    if not os.path.exists(STORAGE):
        data = {"balance": 1000, "history": []}
        with open(STORAGE, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return data
    with open(STORAGE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_storage(data):
    with open(STORAGE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

class BetApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BetFoot — Reproduction")
        self.configure(bg="#071014")
        self.geometry("1100x700")
        self.minsize(900,600)
        self.storage = load_storage()
        self.balance = self.storage.get("balance", 1000)
        self.history = self.storage.get("history", [])
        self.selections = []  # list of dicts {id, home, away, odd}
        self._create_styles()
        self._build_ui()
        self.render_selections()
        self.update_balance_label()

    def _create_styles(self):
        style = ttk.Style(self)
        # Use default theme and tweak
        try:
            style.theme_use("clam")
        except:
            pass
        style.configure("TFrame", background="#0f1113")
        style.configure("Card.TFrame", background="#151617", relief="flat")
        style.configure("Header.TLabel", background="#0f1113", foreground="#ffffff", font=("Inter", 14, "bold"))
        style.configure("Muted.TLabel", background="#151617", foreground="#9aa0a6", font=("Inter", 9))
        style.configure("Red.TButton", background="#E30613", foreground="#fff", font=("Inter", 10, "bold"))
        style.map("Red.TButton",
                  background=[("active", "#b0050f")])
        style.configure("Small.TButton", background="#0b0c0d", foreground="#fff", borderwidth=1)
        style.configure("TLabel", background="#0f1113", foreground="#ffffff")
        style.configure("Balance.TLabel", background="#151617", foreground="#ffffff", font=("Inter", 12, "bold"))

    def _build_ui(self):
        # Header
        header = ttk.Frame(self, padding=(12,10), style="TFrame")
        header.pack(fill="x")
        brand = ttk.Frame(header, style="TFrame")
        brand.pack(side="left", padx=12)
        logo = tk.Label(brand, text="BF", bg="#E30613", fg="#fff", width=3, height=1, font=("Inter", 12, "bold"))
        logo.pack(side="left", padx=(0,8))
        ttk.Label(brand, text="BetFoot", style="Header.TLabel").pack(side="left")
        ttk.Label(brand, text="Reproduction Betclic — Version Foot", style="Muted.TLabel").pack(side="left", padx=8)

        nav = ttk.Frame(header, style="TFrame")
        nav.pack(side="right", padx=12)
        for name in ["Accueil","Matchs en direct","Paris","Résultats","Mon compte"]:
            b = ttk.Button(nav, text=name, style="Small.TButton")
            b.pack(side="left", padx=6)

        # Main container
        container = ttk.Frame(self, padding=12, style="TFrame")
        container.pack(fill="both", expand=True)

        top = ttk.Frame(container, style="TFrame")
        top.pack(fill="both", expand=True)

        main = ttk.Frame(top, style="TFrame")
        main.pack(side="left", fill="both", expand=True)

        # Hero / Matches slider & list
        hero = ttk.Frame(main, padding=12, style="Card.TFrame")
        hero.pack(fill="x", pady=(0,12))

        header_row = ttk.Frame(hero, style="Card.TFrame")
        header_row.pack(fill="x")
        ttk.Label(header_row, text="Prochains matchs internationaux", style="Header.TLabel").pack(side="left")
        quick_frame = ttk.Frame(header_row, style="Card.TFrame")
        quick_frame.pack(side="right")
        ttk.Button(quick_frame, text="Parier maintenant", style="Red.TButton", command=lambda: self.matches_canvas.yview_moveto(0)).pack(side="left", padx=6)
        ttk.Button(quick_frame, text="Voir les cotes", style="Small.TButton", command=self.flash_odds).pack(side="left", padx=6)

        # Slider (a simple horizontal frame with canvas)
        slider_wrapper = ttk.Frame(hero, style="Card.TFrame")
        slider_wrapper.pack(fill="x", pady=8)
        canvas = tk.Canvas(slider_wrapper, height=120, bg="#151617", highlightthickness=0)
        canvas.pack(side="left", fill="x", expand=True)
        self.slider_canvas = canvas
        self._draw_slider()

        # Matches list
        matches_frame = ttk.Frame(main, padding=8, style="Card.TFrame")
        matches_frame.pack(fill="both", expand=True)
        ttk.Label(matches_frame, text="Tous les matchs", style="Header.TLabel").pack(anchor="w", pady=(0,8))

        # Scrollable matches list
        matches_canvas = tk.Canvas(matches_frame, bg="#151617", highlightthickness=0)
        scrollbar = ttk.Scrollbar(matches_frame, orient="vertical", command=matches_canvas.yview)
        matches_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        matches_canvas.pack(side="left", fill="both", expand=True)
        self.matches_inner = ttk.Frame(matches_canvas, style="Card.TFrame")
        self.matches_inner.bind("<Configure>", lambda e: matches_canvas.configure(scrollregion=matches_canvas.bbox("all")))
        matches_canvas.create_window((0,0), window=self.matches_inner, anchor="nw")
        self.matches_canvas = matches_canvas

        for m in MATCHES:
            self._add_match_row(self.matches_inner, m)

        # Account panel (small) on main column
        account = ttk.Frame(main, padding=10, style="Card.TFrame")
        account.pack(fill="x", pady=12)
        ttk.Label(account, text="Mon compte", style="Header.TLabel").pack(anchor="w")
        bal_frame = ttk.Frame(account, style="Card.TFrame")
        bal_frame.pack(fill="x", pady=6)
        ttk.Label(bal_frame, text="Solde virtuel", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        self.balance_label = ttk.Label(bal_frame, text=f"{self.balance} crédits", style="Balance.TLabel")
        self.balance_label.grid(row=1, column=0, sticky="w")
        btns = ttk.Frame(bal_frame, style="Card.TFrame")
        btns.grid(row=0, column=1, rowspan=2, sticky="e")
        ttk.Button(btns, text="Recharger en crédits fictifs", style="Small.TButton", command=self.recharge).pack(padx=4, pady=2)
        ttk.Button(btns, text="Historique des paris", style="Small.TButton", command=self.toggle_history).pack(padx=4, pady=2)
        self.history_panel = ttk.Frame(account, style="Card.TFrame")
        self.history_panel.pack(fill="x", pady=(8,0))
        self.history_panel.pack_forget()

        # Ticket (aside)
        ticket = ttk.Frame(top, width=360, padding=12, style="Card.TFrame")
        ticket.pack(side="right", fill="y")
        ttk.Label(ticket, text="Ticket de pari", style="Header.TLabel").pack(anchor="w")
        self.selections_container = ttk.Frame(ticket, style="Card.TFrame")
        self.selections_container.pack(fill="both", expand=True, pady=8)
        stake_frame = ttk.Frame(ticket, style="Card.TFrame")
        stake_frame.pack(fill="x", pady=6)
        ttk.Label(stake_frame, text="Mise (crédits)").pack(side="left")
        self.stake_var = tk.StringVar(value="10")
        stake_entry = ttk.Entry(stake_frame, textvariable=self.stake_var, width=10)
        stake_entry.pack(side="left", padx=8)
        stake_entry.bind("<KeyRelease>", lambda e: self.update_potential())
        ttk.Label(ticket, text="Gain potentiel :").pack(anchor="w")
        self.potential_label = ttk.Label(ticket, text="0", style="Header.TLabel")
        self.potential_label.pack(anchor="w", pady=(0,8))
        ttk.Button(ticket, text="Placer le pari", style="Red.TButton", command=self.place_bet).pack(fill="x")
        ttk.Label(ticket, text="Transactions factices uniquement – aucun argent réel.", style="Muted.TLabel").pack(anchor="w", pady=(8,0))

    def _draw_slider(self):
        # Draw simple cards horizontally
        canvas = self.slider_canvas
        canvas.delete("all")
        x = 8
        for m in MATCHES:
            w = 240
            h = 100
            rect = canvas.create_rectangle(x, 8, x+w, 8+h, fill="#111214", outline="#2a2a2a", width=1)
            canvas.create_text(x+16, 24, anchor="nw", text=f"{m['home']}  vs  {m['away']}", fill="#fff", font=("Inter", 10, "bold"))
            canvas.create_text(x+16, 48, anchor="nw", text=f"{m['date'].replace('T',' — ')}", fill="#9aa0a6", font=("Inter", 9))
            # odds
            ox = x+16
            for odd in m['odds']:
                canvas.create_rectangle(ox-2, 72, ox+48, 96, fill="#0f1113", outline="#222")
                canvas.create_text(ox+22, 82, text=str(odd), fill="#fff")
                ox += 56
            x += w + 12
        canvas.configure(scrollregion=(0,0,x,120))

    def _add_match_row(self, parent, m):
        frame = ttk.Frame(parent, padding=8, style="Card.TFrame")
        frame.pack(fill="x", pady=6)
        left = ttk.Frame(frame, style="Card.TFrame")
        left.pack(side="left", fill="x", expand=True)
        ttk.Label(left, text=f"{m['home']}  vs  {m['away']}", style="Header.TLabel").pack(anchor="w")
        ttk.Label(left, text=datetime.fromisoformat(m['date']).strftime("%d %b %Y - %H:%M"), style="Muted.TLabel").pack(anchor="w")

        right = ttk.Frame(frame, style="Card.TFrame")
        right.pack(side="right")
        oddsf = ttk.Frame(right, style="Card.TFrame")
        oddsf.pack(side="left", padx=4)
        for odd in m["odds"]:
            b = ttk.Button(oddsf, text=str(odd), style="Small.TButton",
                           command=lambda o=odd, mm=m: self.add_selection_with_odd(mm, o))
            b.pack(side="left", padx=4)
        ttk.Button(right, text="Ajouter au ticket", style="Small.TButton", command=lambda mm=m: self.add_selection_default(mm)).pack(side="left", padx=8)

    # Logic
    def add_selection_default(self, match):
        odd = match["odds"][0]
        self._add_selection(match["id"], match["home"], match["away"], odd)

    def add_selection_with_odd(self, match, odd):
        self._add_selection(match["id"], match["home"], match["away"], odd)

    def _add_selection(self, id_, home, away, odd):
        if any(s["id"] == id_ for s in self.selections):
            messagebox.showinfo("Info", "Match déjà dans le ticket")
            return
        self.selections.append({"id": id_, "home": home, "away": away, "odd": str(odd)})
        self.render_selections()

    def render_selections(self):
        for child in self.selections_container.winfo_children():
            child.destroy()
        if not self.selections:
            ttk.Label(self.selections_container, text="Aucune sélection", style="Muted.TLabel").pack()
            self.potential_label.config(text="0")
            return
        for idx, s in enumerate(self.selections):
            f = ttk.Frame(self.selections_container, style="Card.TFrame")
            f.pack(fill="x", pady=4)
            left = ttk.Frame(f, style="Card.TFrame")
            left.pack(side="left", fill="x", expand=True)
            ttk.Label(left, text=f"{s['home']} vs {s['away']}", font=("Inter", 10, "bold")).pack(anchor="w")
            ttk.Label(left, text=f"Cote choisie: {s['odd']}", style="Muted.TLabel").pack(anchor="w")
            btn = ttk.Button(f, text="Supprimer", style="Small.TButton", command=lambda i=idx: self.remove_selection(i))
            btn.pack(side="right")

        self.update_potential()

    def remove_selection(self, idx):
        try:
            self.selections.pop(idx)
        except IndexError:
            pass
        self.render_selections()

    def update_potential(self):
        try:
            stake = float(self.stake_var.get())
        except:
            stake = 0.0
        combined = 1.0
        for s in self.selections:
            try:
                combined *= float(s["odd"])
            except:
                combined *= 1.0
        pot = round(stake * combined, 2)
        self.potential_label.config(text=str(pot))

    def place_bet(self):
        if not self.selections:
            messagebox.showwarning("Attention", "Aucune sélection dans le ticket.")
            return
        try:
            stake = float(self.stake_var.get())
        except:
            messagebox.showwarning("Attention", "Entrez une mise valide.")
            return
        if stake <= 0:
            messagebox.showwarning("Attention", "Entrez une mise valide.")
            return
        if stake > self.balance:
            messagebox.showwarning("Attention", "Solde insuffisant. Rechargez vos crédits fictifs.")
            return
        pot = float(self.potential_label.cget("text") or 0)
        # perform "fake" placement
        self.balance -= stake
        self.update_balance_label()
        record = {"date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "selections": self.selections.copy(), "stake": stake, "potential": pot}
        self.history.insert(0, record)
        self._sync_storage()
        messagebox.showinfo("Pari placé", f"Pari placé (fictif). Gain potentiel: {pot} crédits.")
        self.selections.clear()
        self.render_selections()

    def update_balance_label(self):
        self.balance_label.config(text=f"{int(self.balance)} crédits")
        self.storage["balance"] = int(self.balance)
        save_storage(self.storage)

    def _sync_storage(self):
        self.storage["balance"] = int(self.balance)
        self.storage["history"] = self.history
        save_storage(self.storage)

    def recharge(self):
        # mimic +500 credits
        self.balance += 500
        self._sync_storage()
        self.update_balance_label()
        messagebox.showinfo("Recharge", "500 crédits fictifs ajoutés !")

    def toggle_history(self):
        if self.history_panel.winfo_ismapped():
            self.history_panel.pack_forget()
            return
        for child in self.history_panel.winfo_children():
            child.destroy()
        if not self.history:
            ttk.Label(self.history_panel, text="Aucun pari effectué", style="Muted.TLabel").pack()
        else:
            for h in self.history:
                frame = ttk.Frame(self.history_panel, style="Card.TFrame")
                frame.pack(fill="x", pady=4)
                ttk.Label(frame, text=h["date"], style="Muted.TLabel").pack(anchor="w")
                ttk.Label(frame, text=f"Mise: {h['stake']} crédits — Potentiel: {h['potential']} crédits", font=("Inter", 9, "bold")).pack(anchor="w")
                sels = "\n".join([f"{s['home']} vs {s['away']} ({s['odd']})" for s in h["selections"]])
                ttk.Label(frame, text=sels, style="Muted.TLabel").pack(anchor="w")
        self.history_panel.pack(fill="x", pady=(8,0))

    def flash_odds(self):
        # small visual feedback by flashing the slider canvas background
        c = self.slider_canvas
        orig = c["bg"]
        def _flash(step=0):
            if step % 2 == 0:
                c.config(bg="#0d0f10")
            else:
                c.config(bg=orig)
            if step < 4:
                c.after(80, lambda: _flash(step+1))
        _flash()

if __name__ == "__main__":
    app = BetApp()
    app.mainloop()

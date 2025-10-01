# compte.py
# Reproduction Tkinter de compte.html (compte + admin panel)
import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
from datetime import datetime

STORAGE = "storage.json"

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

class CompteApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mon compte — BetFoot")
        self.configure(bg="#070707")
        self.geometry("820x640")
        self.storage = load_storage()
        self.balance = int(self.storage.get("balance", 1000))
        self.history = self.storage.get("history", [])
        self._create_styles()
        self._build_ui()
        self.update_balance_display()

    def _create_styles(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except:
            pass
        style.configure("TFrame", background="#1a1a1a")
        style.configure("Card.TFrame", background="#1a1a1a")
        style.configure("Header.TLabel", background="#1a1a1a", foreground="#fff", font=("Inter", 14, "bold"))
        style.configure("Muted.TLabel", background="#1a1a1a", foreground="#cfcfcf")
        style.configure("Red.TButton", background="#E30613", foreground="#fff", font=("Inter", 10, "bold"))
        style.map("Red.TButton", background=[("active", "#b0050f")])

    def _build_ui(self):
        header = ttk.Frame(self, padding=10, style="TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="BetFoot", font=("Inter", 16, "bold"), background="#070707", foreground="#fff").pack(side="left")
        # container
        container = ttk.Frame(self, padding=12, style="TFrame")
        container.pack(fill="both", expand=True)

        main_card = ttk.Frame(container, padding=16, style="Card.TFrame")
        main_card.pack(fill="both", expand=True)

        ttk.Label(main_card, text="Espace Mon compte", style="Header.TLabel").pack(anchor="w")
        bal = ttk.Frame(main_card, style="Card.TFrame")
        bal.pack(fill="x", pady=8)
        self.balance_label = ttk.Label(bal, text="", style="Header.TLabel")
        self.balance_label.pack(side="left")
        ttk.Button(bal, text="Recharger 500 crédits fictifs", style="Red.TButton", command=self.recharge).pack(side="right")

        ttk.Label(main_card, text="Historique des paris", style="Header.TLabel").pack(anchor="w", pady=(12,0))

        # History table (Treeview)
        columns = ("date", "stake", "potential", "selections")
        self.tree = ttk.Treeview(main_card, columns=columns, show="headings", height=8)
        self.tree.heading("date", text="Date")
        self.tree.heading("stake", text="Mise")
        self.tree.heading("potential", text="Gain potentiel")
        self.tree.heading("selections", text="Sélections")
        self.tree.column("date", width=140)
        self.tree.column("stake", width=80, anchor="center")
        self.tree.column("potential", width=120, anchor="center")
        self.tree.column("selections", width=400)
        self.tree.pack(fill="both", expand=True, pady=8)
        self.render_history()

        # Admin box
        self.admin_box = ttk.Frame(self, padding=12, style="Card.TFrame")
        self.admin_box.pack(fill="x", pady=(8,10))
        ttk.Label(self.admin_box, text="Espace Administrateur", style="Header.TLabel").pack()
        form = ttk.Frame(self.admin_box, style="Card.TFrame")
        form.pack(pady=6)
        ttk.Label(form, text="Identifiant:", style="Muted.TLabel").grid(row=0, column=0, sticky="e", padx=6, pady=2)
        self.admin_user = ttk.Entry(form)
        self.admin_user.grid(row=0, column=1, pady=2)
        ttk.Label(form, text="Mot de passe:", style="Muted.TLabel").grid(row=1, column=0, sticky="e", padx=6, pady=2)
        self.admin_pass = ttk.Entry(form, show="*")
        self.admin_pass.grid(row=1, column=1, pady=2)
        ttk.Button(form, text="Se connecter", style="Red.TButton", command=self.admin_login).grid(row=2, column=0, columnspan=2, pady=6)

        self.admin_msg = ttk.Label(self.admin_box, text="", style="Muted.TLabel")
        self.admin_msg.pack()

        # Admin panel (hidden)
        self.admin_panel = ttk.Frame(self, padding=12, style="Card.TFrame")
        # inside admin_panel: modify balance, view history with remove buttons, logout
        ttk.Label(self.admin_panel, text="Tableau de bord Admin", style="Header.TLabel").pack()
        mod_frame = ttk.Frame(self.admin_panel, style="Card.TFrame")
        mod_frame.pack(pady=8)
        ttk.Label(mod_frame, text="Modifier le solde:", style="Muted.TLabel").grid(row=0, column=0, padx=6)
        self.admin_balance_entry = ttk.Entry(mod_frame, width=15)
        self.admin_balance_entry.grid(row=0, column=1, padx=6)
        ttk.Button(mod_frame, text="Mettre à jour", style="Red.TButton", command=self.update_balance_admin).grid(row=0, column=2, padx=6)

        ttk.Label(self.admin_panel, text="Historique des paris (admin)", style="Muted.TLabel").pack(anchor="w")
        self.admin_history_container = ttk.Frame(self.admin_panel, style="Card.TFrame")
        self.admin_history_container.pack(fill="both", expand=True, pady=6)

        ttk.Button(self.admin_panel, text="Se déconnecter", style="Small.TButton", command=self.logout_admin).pack(pady=6)

    def render_history(self):
        # clear tree
        for i in self.tree.get_children():
            self.tree.delete(i)
        if not self.history:
            return
        for rec in self.history:
            sels = " | ".join([f"{s['home']} vs {s['away']} ({s['odd']})" for s in rec.get("selections", [])])
            self.tree.insert("", "end", values=(rec.get("date",""), rec.get("stake",""), rec.get("potential",""), sels))

    def update_balance_display(self):
        self.balance_label.config(text=f"Solde fictif : {self.balance} crédits")
        # keep storage in sync
        self.storage["balance"] = int(self.balance)
        save_storage(self.storage)

    def recharge(self):
        self.balance += 500
        self.storage["balance"] = int(self.balance)
        save_storage(self.storage)
        self.update_balance_display()
        messagebox.showinfo("Recharge", "500 crédits fictifs ajoutés !")

    # Admin logic
    def admin_login(self):
        user = self.admin_user.get().strip()
        pw = self.admin_pass.get().strip()
        if user == "admin" and pw == "admin123":
            self.admin_msg.config(text="Connexion réussie !", foreground="#00ff00")
            self.admin_box.pack_forget()
            self.admin_panel.pack(fill="x", padx=12, pady=(0,12))
            self.admin_balance_entry.delete(0, tk.END)
            self.admin_balance_entry.insert(0, str(self.balance))
            self.render_admin_history()
        else:
            self.admin_msg.config(text="Identifiant ou mot de passe incorrect.", foreground="#ff3b3b")

    def logout_admin(self):
        self.admin_panel.pack_forget()
        self.admin_box.pack(fill="x", pady=(8,10))
        self.admin_msg.config(text="")

    def render_admin_history(self):
        for child in self.admin_history_container.winfo_children():
            child.destroy()
        if not self.history:
            ttk.Label(self.admin_history_container, text="Aucun pari effectué", style="Muted.TLabel").pack()
            return
        for idx, h in enumerate(self.history):
            f = ttk.Frame(self.admin_history_container, style="Card.TFrame", padding=6)
            f.pack(fill="x", pady=4)
            ttk.Label(f, text=h.get("date",""), style="Muted.TLabel").pack(anchor="w")
            ttk.Label(f, text=f"Mise: {h.get('stake','')} — Potentiel: {h.get('potential','')}", font=("Inter", 9, "bold")).pack(anchor="w")
            sels = "\n".join([f"{s['home']} vs {s['away']} ({s['odd']})" for s in h.get("selections", [])])
            ttk.Label(f, text=sels, style="Muted.TLabel").pack(anchor="w")
            ttk.Button(f, text="Supprimer ce pari", style="Red.TButton", command=lambda i=idx: self.remove_bet(i)).pack(anchor="e", pady=4)

    def remove_bet(self, idx):
        if idx < 0 or idx >= len(self.history):
            return
        if not messagebox.askyesno("Confirmer", "Supprimer ce pari ?"):
            return
        self.history.pop(idx)
        self.storage["history"] = self.history
        save_storage(self.storage)
        self.render_history()
        self.render_admin_history()

    def update_balance_admin(self):
        val = self.admin_balance_entry.get().strip()
        try:
            n = int(val)
            if n < 0:
                raise ValueError
        except:
            messagebox.showwarning("Erreur", "Entrez une valeur valide !")
            return
        self.balance = n
        self.storage["balance"] = self.balance
        save_storage(self.storage)
        self.update_balance_display()
        messagebox.showinfo("Mis à jour", "Solde mis à jour !")

    def render_history(self):
        # keep treeview in sync
        for i in self.tree.get_children():
            self.tree.delete(i)
        if not self.history:
            return
        for rec in self.history:
            sels = " | ".join([f"{s['home']} vs {s['away']} ({s['odd']})" for s in rec.get("selections", [])])
            self.tree.insert("", "end", values=(rec.get("date",""), rec.get("stake",""), rec.get("potential",""), sels))

if __name__ == "__main__":
    app = CompteApp()
    app.mainloop()

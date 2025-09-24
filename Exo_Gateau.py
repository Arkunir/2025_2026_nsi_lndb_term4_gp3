# gateau = {
#     "nom" : "gâteau",
#     "ingredients" : ["sucre", "farine", "lait", "oeuf"],
#     "temps_de_cuisson" : 30,
#     "temperature_de_cuisson" : 150
# }

# def se_presenter():
#     texte1 = f"Je suis un {gateau["nom"]} et mes ingrédients sont {gateau["ingredients"]}"
#     print(texte1)

# def cuire():
#     texte2 = f"Le gateau cuit pendant {gateau["temps_de_cuisson"]} minutes et à {gateau["temperature_de_cuisson"]} degrés Celsius"
#     print(texte2)

# se_presenter()
# cuire()

class Gateau():
    def __init__(self, nom, ingredients, temps_de_cuisson):
        self.nom = nom
        self.ingredients = ingredients
        self.temps_de_cuisson = temps_de_cuisson
    
    def se_presenter(self):
        texte = f"\nLe gateau s'appelle (self.nom).\n" \
            + f"Il contient (self.ingredients).\n" \
            + f"Le temps de cuisson est de (self.temps_de_cuisson) minutes."
        print(texte)
    
    def cuire(self):
        texte = f" \nCuisson spéciale pour le (self.nom) :" \
        + f"- Enfourner à 180°C pendant (self.temps_de_cuisson) minutes."
        print(texte)

my_cake = Gateau("marbré", ["beurre", "oeufs", "sucre", "chocolat"])
my_cake.presenter()
my_cake.cuire(24)

class Gateau_anniversaire(Gateau):
    def __init__(self, nom, ingredients, temps_de_cuisson, nombre_bougies):
        super().__init__(nom, ingredients, temps_de_cuisson)
        self.nombre_bougies = nombre_bougies

    def cuire(self, alpha):
        texte = f" \nCuisson spéciale pour le (self.nom) :" \
        + f"- Enfourner à 200°C pendant (self.temps_de_cuisson) minutes." \
        + f"Le paramètre est (alpha)"
        print(texte)

my_birthday_cake = Gateau_anniversaire("tarte à l'orange", \
                                       ["farine", "oeufs", "sucre", "oranges"], \
                                        50, 55)
my_birthday_cake.presenter_bis()
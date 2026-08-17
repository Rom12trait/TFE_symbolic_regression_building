# Travail de fin d'études: Symbolic Regression for Interpretable Modeling of Building Thermal Dynamics

## Description
Cette étude explore une approche de modélisation interprétable de la dynamique thermique des bâtiments, basée sur la régression symbolique, afin d’optimiser efficacement les systèmes HVAC dans un contexte marqué par la transition énergétique et l’électrification rapide. Face à la croissance continue de la superficie bâtie mondiale et à la montée en puissance des pompes à chaleur, les bâtiments deviennent des acteurs majeurs du système électrique. Cette évolution soulève des défis techniques et économiques, notamment en matière de flexibilité et d’interactions avec les marchés de l’électricité à court terme (ex. marché day-ahead avec granularité de 15 minutes).

L’objectif spécifique de ce travail est de développer des modèles dynamiques légers, robustes et surtout interprétables, capables de prédire la température intérieure d’une maison résidentielle unifamiliale à deux étages, modélisée comme une seule zone thermique. Ces modèles utilisent comme variables d’entrée la température extérieure, la température intérieure instantanée et la puissance thermique HVAC, avec pour sortie la température intérieure prédite 15 minutes après. Ils sont ensuite intégrés dans une optimisation coûts liée aux signaux tarifaires du marché day-ahead (15 min) et validés ex-post avec l'outil EnergyPlus.

L’étude s’appuie sur deux jeux de données distincts : une « Année Classique », caractérisée par des consignes stables, et une « Année Dynamique », favorisant une excitation riche des dynamiques thermiques. Plusieurs modèles benchmark ont été développés : modèle persistant, modèles physiques simplifiés 1R1C, régressions linéaires et quadratiques, ainsi que des modèles issus de la régression symbolique (PySR) linéaire et non linéaire.

## Résultats
Les résultats montrent que la régression symbolique, particulièrement dans sa version non linéaire cubique entraînée sur l’année dynamique, parvient à concilier interprétabilité et précision, en assurant une meilleure stabilité temporelle lors de simulations récursives et en réduisant significativement l’erreur sur les puissances HVAC. En optimisation ex-post, le modèle linéaire issu de la RS année dynamique permet quant à lui d’obtenir les coûts énergétiques les plus bas sur les journées testées. Les résultats obtenus avec la régression symbolique surpassent les performances des modèles classiques développés dans cette étude, tels que la régression linéaire, la régression quadratique et le modèle persistant, en termes de stabilité temporelle, de précision sur la puissance HVAC et de qualité des trajectoires générées dans un contexte d’optimisation intégrée aux signaux tarifaires du marché.

L’étude met en lumière le potentiel prometteur de la régression symbolique, tout en soulignant sa nature essentiellement statistique et la nécessité d’explorations futures incluant une recherche plus développée de la RS, ainsi que d’autres méthodes d’apprentissage plus « boîtes noires » pour une comparaison complète.


## Structure du projet
- `src/` : code source Python comprenant les modèles thermiques (régression symbolique PySR dans symbolic_models.py, modèles de régression dans regression_models.py, 1R1C dans physical_models), l’optimiseur HVAC dans hvac_optimizer.py. L'API d'EnergyPlus a été construit au moyen des fichiers api_validator.py, building_model.py et opti.py. Le sous-dossier dataset contient des codes pour créer données de consigne thermostat et schedulefile pour EnergyPlus. 
- `dataset/` : contient toutes les données liées au modèle d'habitation, la météorologie, les données d'entrainement d'EnergyPlus ainsi que les prix du marché Day-Ahead utilisé.
- `optiV4/` : résultats des optimisations sur les 12 jours, fichiers Excel de synthèse et graphiques pour chaque modèle.  
- `ApiV4/` : résultats des validations ex-post sur les 12 jours, fichiers Excel de synthèse et graphiques pour chaque modèle.
- `results/` : résultats linéaires pour l'année classique dans LinearV2_annee_classique ainsi que les résultats linéaires et non-linéaire de l'année dynamique dans LinearV2_annee_dyn.
- `outputs/` : résultats de PySR durant mon exploration de celui-ci quand j'essayais d'obtenir une équation quadratique (les dossiers pysr_square), une équation logarithmique (dossier pysr_log). Equation quadratique avec delta T en sortie (au lieu de Tzone après 15min) (dossiers pysr_quad_delta) ou avec un gradiant sur Tout (Tzone-Tout) en entrée (dossiers pysr_quad_gradientTout). Le dossier pysr_quad_iter200_pop30 reprend l'équation cubique trouvée par PySR. Recherche d'équations exponentielles (dossiers pysr_exp) : le plus intéressant est pysr_exp_n150_p20 qui est ma meilleure équation exponentielle trouvée.


Dans le main est repris tout le code pour l'identification et l'entrainement des modèles, l'optimisation des coûts et la validation ex-post par API EnergyPlus.


## Installation et pré-requis

1. Clonez ce dépôt :  
   ```bash
   git clone https://github.com/Rom12trait/TFE_symbolic_regression_building.git

2. Créez et activez un environnement virtuel Python 3.8+ (venv).
3. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
4. Assurez-vous d’avoir installé Julia (utilisé par PySR) et Pyomo, Gurobi (pour la programmation linéaire) ou Ipopt (pour la programmation non-linéaire).

## Usage

parler de comment utiliser surtout avec runfile, name...

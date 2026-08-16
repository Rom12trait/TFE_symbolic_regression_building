# Travail de fin d'études: Symbolic Regression for Interpretable Modeling of Building Thermal Dynamics

## Description
Ce travail de fin d’études porte sur la modélisation interprétable de la dynamique thermique des bâtiments à l’aide de la régression symbolique, visant à améliorer la gestion et l’optimisation des systèmes de chauffage, ventilation et climatisation (HVAC) dans un contexte de transition énergétique et d’électrification massive. Face à la croissance continue de la superficie bâtie mondiale et à la montée en puissance des pompes à chaleur, les bâtiments deviennent des acteurs majeurs du système électrique. Cette évolution soulève des défis techniques et économiques, notamment en matière de flexibilité et d’interactions avec les marchés de l’électricité à court terme (marché day-ahead avec granularité de 15 minutes).

L’objectif principal est de développer des modèles dynamiques légers, robustes et surtout interprétables, capables de prédire la température intérieure d’une maison résidentielle unifamiliale à deux étages, modélisée comme une seule zone thermique. Ces modèles utilisent comme variables d’entrée la température extérieure, la température intérieure instantanée et la puissance thermique HVAC, avec pour sortie la température intérieure prédite à 15 minutes.

L’étude s’appuie sur deux jeux de données distincts : une « Année Classique », caractérisée par des consignes stables, et une « Année Dynamique », favorisant une excitation riche des dynamiques thermiques. Plusieurs modèles benchmark ont été développés : modèle persistant, modèles physiques simplifiés 1R1C, régressions linéaires et quadratiques, ainsi que des modèles issus de la régression symbolique (PySR) linéaire et non linéaire.

## Résultats
Les résultats montrent que la régression symbolique, particulièrement dans sa version non linéaire cubique entraînée sur l’année dynamique, parvient à concilier interprétabilité et précision, en assurant une meilleure stabilité temporelle lors de simulations récursives et en réduisant significativement l’erreur sur les puissances HVAC. En optimisation ex-post, le modèle linéaire issu de PySR année dynamique permet quant à lui d’obtenir les coûts énergétiques les plus bas sur les journées testées. Ces approches surpassent les limites des modèles plus traditionnels en termes de robustesse, adaptabilité et exploitation dans des cadres d’optimisation intégrée au marché.

L’étude met en lumière le potentiel prometteur de la régression symbolique, tout en soulignant sa nature essentiellement statistique et la nécessité d’explorations futures incluant d’autres méthodes d’apprentissage plus « boîtes noires » pour une comparaison complète.



## Structure du projet
Dans le main est repris tout le code pour l'identification, l'optimisation et la validation ex-post EnergyPlus.
Le dataset contient toutes les données liées au modèle d'habitation, la météorologie, les données d'entrainement d'EnergyPlus ainsi que les prix du marché Day-Ahead utilisé.
OptiV4 reprend tout les résultats et graphiques de l'optimisation pour les modèles.
ApiV4 reprend les résultats de validation pour chaque modèle.
Le dossier results contient les résultats linéaires pour l'année classique dans LinearV2_annee_classique ainsi que les résultats linéaires et non-linéaire de l'année dynamique dans LinearV2_annee_dyn.
Outputs reprend les résultats de toutes les simulations PySR effectué.

## Installation et pré-requis

## Usage

parler de comment utiliser surtout avec runfile, name...

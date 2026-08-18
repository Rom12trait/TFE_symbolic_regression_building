# Travail de fin d'études: Symbolic Regression for Interpretable Modeling of Building Thermal Dynamics

## Contexte
Cette étude explore une approche de modélisation interprétable de la dynamique thermique des bâtiments, basée sur la régression symbolique, afin d’optimiser efficacement les systèmes HVAC dans un contexte marqué par la transition énergétique et l’électrification rapide. Face à la croissance continue de la superficie bâtie mondiale et à la montée en puissance des pompes à chaleur, les bâtiments deviennent des acteurs majeurs du système électrique. Cette évolution soulève des défis techniques et économiques, notamment en matière de flexibilité et d’interactions avec les marchés de l’électricité à court terme (ex. marché day-ahead avec granularité de 15 minutes).

L’objectif spécifique de ce travail est de développer des modèles dynamiques légers, robustes et surtout interprétables, capables de prédire la température intérieure d’une maison résidentielle unifamiliale à deux étages, modélisée comme une seule zone thermique. Ces modèles utilisent comme variables d’entrée la température extérieure, la température intérieure instantanée et la puissance thermique HVAC, avec pour sortie la température intérieure prédite 15 minutes après. Ils sont ensuite intégrés dans une optimisation coûts liée aux signaux tarifaires du marché day-ahead (15 min) et validés ex-post avec l'outil EnergyPlus.

L’étude s’appuie sur deux jeux de données distincts : une « Année Classique », caractérisée par des consignes stables, et une « Année Dynamique », favorisant une excitation riche des dynamiques thermiques. Plusieurs modèles benchmarks ont été développés pour les comparer : modèle persistant, modèles physiques simplifiés 1R1C, régressions linéaires et quadratiques, ainsi que des modèles issus de la régression symbolique (implémentés avec l'outil PySR) linéaire et non linéaire.



## Structure du projet

L'architecture du dépôt est organisée de la manière suivante :

```text
├── doc/                      # contient le rapport du travail de fin d'étude
├── dataset/                  # Données d'entrée (bâtiment, météo, prix Day-Ahead)
│   ├── Meteo                 # données météorologiques
│   ├── ModeleHabitation/     # Fichiers géométriques (.idf) et dictionnaires (.idd)  + année dynamique
│   └── output_energyplus/    # Séries temporelles brutes d'EnergyPlus (15 min)
├── src/                      # Code source de l'application
│   ├── dataset/              # Codes pour créer données de consigne thermostat et schedulefile pour EnergyPlus.
│   ├── physical_models.py    # Modèle physique simplifié (1R1C/Réseau RC) et modèle persistant
│   ├── regression_models.py  # Modèles de régression (Linéaire, Quadratique)
│   ├── symbolic_models.py    # Modèles de Régression Symbolique (PySR)
│   ├── hvac_optimizer.py     # Formulation de l'optimisation mathématique (Pyomo)
│   ├── api_validator.py      # Couplage et exécution de l'API EnergyPlus
│   ├── building_model.py     # Configuration de la structure de l'enveloppe
│   ├── opti.py               # Contient la classe simulateur EnergyPlus
│   └── communs.py            # Fonctions utilitaires (métriques, exports Excel, aggrégation,  conversion latex...)
├── results/                  # Résultats d'entraînement des modèles classiques
│   ├── LinearV2_annee_classique/
│   └── LinearV2_annee_dyn/
├── outputs/                  # Historique d'exploration de la régression symbolique
│   ├── pysr_square_*/        # Recherche d'équations quadratiques (pysr_quad_iter200_pop30 contient l'équation cubique)
│   ├── pysr_log_*/           # Essais de formulations logarithmiques
│   ├── pysr_quad_delta_*/    # Formulation en ΔT (au lieu de Tzone_next)
│   ├── pysr_quad_gradientTout_*/ # Formulation avec (Tzone-Tout) comme entrée
│   └── pysr_exp_*/           # Formulations exponentielles (ex: pysr_exp_n150_p20)
├── optiV4/                   # Résultats des optimisations de coûts (12 jours)
├── ApiV4/                    # Validations ex-post de contrôle thermique (12 jours)
├── main.py                   # Script principal (Pipeline complet de bout en bout)
├── requirements.txt          # Liste des dépendances Python à installer
└── README.md                 # Project documentation
```

### Description des répertoires principaux

* **`src/` (Moteur algorithmique)** : Regroupe la logique métier. La physique et la modélisation mathématique sont isolées dans leurs scripts respectifs, tandis que la gestion des contraintes et des coûts énergétiques est déportée dans l'optimiseur.
* **`results/` & `outputs/` (Phases d'identification)** : Stockent les sorties d'apprentissage. `results/` fige les coefficients des structures classiques alors que `outputs/` recense l'évolution des expressions analytiques testées lors des mutations génétiques de PySR.
* **`optiV4/` & `ApiV4/` (Phases applicatives)** : Centralisent la validation finale. Ces répertoires contiennent les fichiers Excel de synthèse ainsi que les graphiques comparatifs des trajectoires de température intérieure générés à l'issue des scénarios.
* **`main.py`** : Point d'entrée unique du programme. Il orchestre séquentiellement le chargement des données, l'ajustement des paramètres, la planification HVAC par modèle, et l'évaluation de robustesse thermodynamique.

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

L'ensemble du pipeline est centralisé dans le fichier `main.py`. Son exécution se divise en 4 étapes clés, configurables à l'aide des variables globales situées au début du script.

### ️ Configuration globale
Avant de lancer le script, ajustez les variables principales selon vos besoins :
* `runfile` : Nom de la session d'étude actuelle (sert à nommer le dossier d'export dans `results/`).
* `yeartype` : Type de structure d'équation PySR pour l'optimiseur (`'dynamique'`,`'classique'`, `'cube'`, `'exp'`).
* `version` & `name` : Métadonnées pour l'organisation et l'archivage automatique.

---

### Les 4 étapes du Pipeline

#### 1. Chargement et prétraitement des données
Le script charge les fichiers de données selon le type d'année choisi (Classique ou Dynamique).
* Les matrices de features $X$ (`Tzone`, `Tout`, `Qhvac`) et de cible $y$ (`Tzone_next`) sont extraites.
* Un découpage `train_test_split` est appliqué.
* ️**Attention :** L'argument `shuffle` doit impérativement être configuré sur `False` lors de l'utilisation de PySR, car l'ordre chronologique des pas de temps (15 min) est essentiel pour sa fonction de coût temporelle.

#### 2. Identification et entraînement des modèles
Cette étape est séparée en deux processus distincts :

* **Modèles classiques & Benchmarks :**
  Les modèles physiques (`RC`), de régression linéaire (`Linear`) et quadratique (`Quadratic`) sont déclarés dans le dictionnaire `models`. Une boucle `for` automatise leur entraînement et exporte les fichiers de métriques Excel directement dans `results/{runfile}/`. Le modèle de persistance (Naïf) est évalué à ce moment via la méthode `.benchmark()` de la classe RCModel.
* **Régression Symbolique (PySR) :**
  Les hyperparamètres d'exploration génétique sont configurés dans la liste de dictionnaires `simulations_config`. La boucle dédiée permet d'enchaîner plusieurs runs longs en arrière-plan. Les résultats d'exploration brute sont archivés séparément dans le dossier `outputs/`.

#### 3. Résolution du problème d'optimisation HVAC
Le modèle entraîné est couplé à la classe `HVACOptimizer` pour minimiser le coût financier sur le marché Day-Ahead (granularité 15 min) sur une sélection de 12 journées types.
* **Intégration des équations :** Les structures linéaires et quadratiques sont lues automatiquement. Les équations complexes trouvées par PySR doivent être implémentées manuellement dans l'optimiseur (cubique, exponentielle) et activées via la variable `yeartype`. ️**Attention :** pour effectuer l'optimisation pour PySR, il faut qu'un modèle PySR soit instancié dans le dictionnaire `models` afin qu'il puisse choisir l'équation. 
* **Résolveurs mathématiques (Solvers) :** 
  * `Gurobi` est utilisé pour les formulations linéaires (permettant la gestion stricte des variables binaires d'exclusion Chauffage/Refroidissement).
  * `Ipopt` prend le relais pour les modèles non linéaires. 
* **Post-traitement des prix négatifs :** `Ipopt` ne gérant pas les variables binaires, un algorithme de post-traitement corrige les cas de simultanéité (activation concomitante du chaud et du froid lors des prix négatifs) en annulant la puissance minoritaire et en ajustant la puissance restante pour maintenir $Q_{HVAC}$ et la température. Les coûts sont ajustés aussi.

#### 4. Validation Ex-post via l'API EnergyPlus
Les trajectoires de températures de zone optimales ($T_{zone}$) calculées à l'étape précédente sont injectées comme consignes au sein du simulateur thermodynamique de référence **EnergyPlus**. Le script génère alors automatiquement les graphiques comparatifs finaux et agrège les indicateurs de performance dans le dossier `ApiV4/` sous fichier excel.


## Algorithme d'Optimisation HVAC

Le problème d'optimisation prédictive implémenté dans la classe `HVACOptimizer` (via Pyomo) suit la structure mathématique formalisée ci-dessous :

```text
Algorithm: Predictive Building HVAC Cost Optimization
──────────────────────────────────────────────────────────────────────────────────────────
1: Inputs: 
   - Vecteur de prix de l'électricité : p = [prices_t] pour t ∈ [0, 95] (pas de 15 min)
   - Vecteur météo : T_out = [Tout_t] pour t ∈ [0, 95]
   - Condition initiale : T_init (Température de la zone au pas t=0)
   - Modèle thermique identifié : f(·) ∈ {Linéaire, Quadratique, PySR}

2: Initialisation des paramètres de l'enveloppe et du système :
   - Puissances max : Ph_max, Pc_max 
   - Rendements HVAC : η_h, η_c
   - Températures de confort : tmin = 20°C, tmax = 24°C
   - Coefficient de pénalité financière : γ_confort = 1000.0

3: Déclaration des Variables de Décision :
   - P_heating[t] ∈ [0, Ph_max],  P_cooling[t] ∈ [0, Pc_max]   (∀ t ∈ [0, 95])
   - T_zone[t] ∈ ℝ,  λ_low[t] ≥ 0,  λ_high[t] ≥ 0            (∀ t ∈ [0, 96])
   - z ∈ {0, 1} (Variable binaire d'exclusion Chauffage/Refroidissement)

4: Condition aux limites :
   - T_zone[0] = T_init

5: Boucle temporelle (Horizon de 24h, 96 pas de temps) :
   for t = 1 allant de 1 à 96 faire
       a. Dynamique thermique :
          T_zone[t] = f(T_zone[t-1], T_out[t-1], Q_hvac[t-1])
          où Q_hvac[t-1] = η_h * P_heating[t-1] - η_c * P_cooling[t-1]

       b. Limites de confort souples (Soft Constraints via Slacks) :
          T_zone[t] + λ_low[t]  ≥ tmin
          T_zone[t] - λ_high[t] ≤ tmax

       c. Contraintes d'exclusion Big-M (Si mode linéaire activé) :
          P_heating[t-1] ≤ Ph_max * z
          P_cooling[t-1] ≤ Pc_max * (1 - z)
   end

6: Résolution du Problème d'Optimisation :
   Minimiser Fonction_Objectif = Coût_Électricité + Pénalités_Confort
   
   Coût_Électricité = ∑ [ p[t] * (P_heating[t] + P_cooling[t]) / 1000 * Δt ] pour t allant de 0 à 95
   Pénalités_Confort = ∑ [ γ_confort * (λ_low[t] + λ_high[t]) ] pour t allant de 0 à 96

7: Sélection du Solveur :
   si mode == 'linear' alors
       Résolution du programme linéaire mixte via Gurobi
   sinon
       Résolution du programme non-linéaire continu via Ipopt
   fin si

8: Outputs: Trajectoires optimales P_heating*, P_cooling*, T_zone*
──────────────────────────────────────────────────────────────────────────────────────────
```

### Points clés mis en valeur dans le README

* **Agnosticisme du modèle `f(·)`** : L'algorithme montre clairement comment l'optimiseur s'interface avec n'importe quel modèle entraîné à l'étape précédente. Les variables Pyomo sont injectées dynamiquement dans l'expression symbolique générée (qu'il s'agisse des coefficients d'une régression linéaire ou d'un arbre d'équations PySR).
* **Gestion du confort par relaxation (Slacks λ)** : Pour éviter que le solveur ne déclare le problème "infaisable" (Infeasible) lors de conditions météo extrêmes, les barrières thermiques de 20°C et 24°C sont dites "souples". Toute violation applique une lourde pénalité financière virtuelle (\(\gamma = 1000\)) dans la fonction objectif.
* **Exclusivité logique des actionneurs** : La variable binaire \(z\) empêche mathématiquement le système de chauffer et refroidir simultanément.

## Résultats

## Performances comparatives (Synthèse sur 12 jours)

Le tableau ci-dessous présente la synthèse comparative des performances entre l'optimisation théorique et la validation Ex-Post, pour chaque modèle utilisé. Les valeurs de coûts et d'énergies sont cumulées sur l'ensemble des 12 journées de test, tandis que les métriques d'erreur représentent des moyennes pour chaque modèle.

| Métrique | RL A.C. | RL A.D. | RS Lin. A.C. | RS Lin. A.D. | Régression quadratique A.D. | RS-cubique A.D. | RS Exp. A.D. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cost Opti (€)** | -0.64 | 12.23 | -1.39 | 5.93 | 8.23 | 8.26 | 6.18 |
| **Cost Ex-post (€)** | 14.91 | 14.55 | 15.08 | 13.73 | 14.67 | 15.70 | 16.57 |
| **Diff (€)** | 15.55 | 2.32 | 16.47 | 7.80 | 6.44 | 7.44 | 10.40 |
| **RMSE Temp (°C)** | 0.04 | 0.04 | 0.04 | 0.13 | 0.04 | 0.20 | 0.30 |
| **RMSE Power (W)** | 921.39 | 1033.8 | 1119.5 | 478.4 | 907.7 | 460.9 | 718.7 |
| **MAPE Power (%)** | 184.9 | 184.5 | 200.6 | 78.9 | 189.1 | 60.8 | 130.1 |
| **Energie Opti (kWh)** | 99.1 | 197.0 | 130.2 | 89.2 | 190.7 | 109.4 | 165.9 |
| **Energie Ex-post (kWh)** | 164.0 | 163.1 | 164.8 | 168.6 | 164.0 | 182.8 | 184.9 |

>  **Légende :** 
> * $RMSE_{Temp} = f(T_{zone, opti}, T_{zone,ex-post})$
> * **MAPE** : Erreur absolue moyenne en pourcentage (*Mean Absolute Percentage Error*).
> * **A.C.** : Année Classique \| **A.D.** : Année Dynamique.
> * **RL** : Régression Linéaire \| **RS** : Régression Symbolique.
> 
> **Génération automatique :** L'extraction des données, l'agrégation sur les 12 jours et la mise en forme de ces lignes de résultats sont entièrement automatisées dans le projet via la fonction `communs.generate_tfe_summary_line()`.

Les résultats montrent que la régression symbolique, particulièrement dans sa version non linéaire cubique entraînée sur l’année dynamique, parvient à concilier interprétabilité et précision, en assurant une meilleure stabilité temporelle lors de simulations récursives et en réduisant significativement l’erreur sur les puissances HVAC. En optimisation ex-post, le modèle linéaire issu de la RS année dynamique permet quant à lui d’obtenir les coûts énergétiques les plus bas sur les journées testées. Les résultats obtenus avec la régression symbolique surpassent les performances des modèles classiques développés dans cette étude, tels que la régression linéaire, la régression quadratique et le modèle persistant, en termes de stabilité temporelle, de précision sur la puissance HVAC et de qualité des trajectoires générées dans un contexte d’optimisation intégrée aux signaux tarifaires du marché.

L’étude met en lumière le potentiel prometteur de la régression symbolique, tout en soulignant sa nature essentiellement statistique et la nécessité d’explorations futures incluant une recherche plus développée de la RS, ainsi que d’autres méthodes d’apprentissage plus « boîtes noires » pour une comparaison complète.


## Licence
Ce projet est distribué sous la licence MIT.

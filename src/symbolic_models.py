import numpy as np
import sympy
import os
import time
from pathlib import Path
from pysr import PySRRegressor
from sklearn.preprocessing import StandardScaler
from src.new_base_model import BaseModel


class PySRThermalModel(BaseModel):
    def __init__(self, niterations=80, run_id="pysr_run", populations=15, population_size=30,maxsize=20, parsimony=0.002, complexity_of_constants=1, **kwargs):
        super().__init__(**kwargs)
        # On définit un chemin absolu propre avec des slashes classiques '/'
        #pysr_output_dir = "C:/Users/Corentin/pysr_outputs"
        #os.makedirs(pysr_output_dir, exist_ok=True)
        #pysr_dir = "C:\\pysrresults"
        self.run_id = run_id
        self.pysr_dir = os.path.abspath("C:/pysr_outputs")
        #unix_clean_path = "/pysr_outputs"
        multi_step_loss = """
function my_trajectory_loss(tree, dataset, options)
    X = dataset.X  # Matrix{Float32} de taille (3 x n_samples)
    y = dataset.y  # Vector{Float32} de taille (n_samples)
    n_samples = dataset.n
    
        horizon = 48  # Simulation récursive sur x/4 heure (pas de 15 minutes)
    total_error = 0.0
    count = 0
    
    X_step = Matrix{Float32}(undef, 3, 1)
    saut = 1000
    start_offset = rand(1:saut)
    for i in start_offset:saut:(n_samples - horizon)
        # Sécurité pour ne pas déborder du tableau à cause de l'offset
        if i < 1
            continue
        end
        # Température initiale réelle au début de l'horizon de 1 heure
        T_sim = X[1, i]
        
        for h in 0:(horizon - 1)
            idx = i + h
            
            # Construction d'une vraie Matrix{Float32} de taille (3 x 1) pour PySR
            X_step[1, 1] = T_sim
            X_step[2, 1] = X[2, idx]
            X_step[3, 1] = X[3, idx]
            
            # Évaluation du pas de temps par l'arbre PySR
            pred_step, step_flag = eval_tree_array(tree, X_step, options)
            if !step_flag
                return Inf
            end
            
            # --- CORRECTION DU CRASH DE TYPE ---
            # pred_step est un Vector{Float32} à 1 élément. 
            # On extrait explicitement la valeur scalaire numérique via [1] 
            # pour que T_sim reste un simple nombre Float32 au pas suivant.
            T_sim = pred_step[1]
            
            # Accumulation de l'écart quadratique trajectoire
            total_error += (T_sim - y[idx])^2
            count += 1
        end
    end
    
    return total_error / count
end
"""
        multi_step_loss_dt_tout = """
        function my_physics_informed_loss(tree, dataset, options)
            X = dataset.X  # Matrix{Float32} de taille (3 x n_samples) -> [T_réel, DT_réel, Q_réel]
            y = dataset.y  # Vector{Float32} de taille (n_samples) -> T_next
            n_samples = dataset.n

            horizon = 48  # 12 heures
            total_error = 0.0
            count = 0

            X_step = Matrix{Float32}(undef, 3, 1)
            saut = 1000  
            start_offset = rand(1:saut)

            for i in start_offset:saut:(n_samples - horizon)
                if i < 1
                    continue
                end

                T_sim = X[1, i]  # Température de zone initiale simulée

                for h in 0:(horizon - 1)
                    idx = i + h

                    # --- DÉDUCTION DE TOUT SANS COLONNE SUPPLÉMENTAIRE ---
                    # Puisque DT_réel = T_réel - Tout_réel, alors Tout_réel = T_réel - DT_réel
                    T_real_idx = X[1, idx]
                    DT_real_idx = X[2, idx]
                    Tout_real = T_real_idx - DT_real_idx

                    # Recalcul du gradient dynamique simulé
                    DT_sim = T_sim - Tout_real

                    # Remplissage des 3 entrées de l'arbre PySR
                    X_step[1, 1] = T_sim
                    X_step[2, 1] = DT_sim  # Transmis sous le nom "DT"
                    X_step[3, 1] = X[3, idx] # Transmis sous le nom "Q"

                    pred_step, step_flag = eval_tree_array(tree, X_step, options)
                    if !step_flag
                        return Inf
                    end

                    T_sim = pred_step[1]
                    total_error += (T_sim - y[idx])^2
                    count += 1
                end
            end

            return total_error / count
        end
        """

        multi_step_loss_deltaT = """
function my_delta_trajectory_loss(tree, dataset, options)
    X = dataset.X  # Matrix{Float32} (3 x n_samples) -> [T_scaled, Tout_scaled, Q_scaled]
    y = dataset.y  # Vector{Float32} (n_samples) -> Delta_T_scaled_réel
    n_samples = dataset.n
    
    horizon = 48  
    total_error = 0.0
    count = 0
    
    X_step = Matrix{Float32}(undef, 3, 1)
    saut = 1000  
    start_offset = rand(1:saut)
    
    
    
    for i in start_offset:saut:(n_samples - horizon)
        if i < 1
            continue
        end
        
        # On initialise la trajectoire avec la valeur scaled de départ
        T_sim_scaled = X[1, i]  
        
        for h in 0:(horizon - 1)
            idx = i + h
            
            # Remplissage des 3 entrées normalisées : [T_sim_scaled, Tout_scaled, Q_scaled]
            X_step[1, 1] = T_sim_scaled
            X_step[2, 1] = X[2, idx]  
            X_step[3, 1] = X[3, idx]  
            
            pred_delta_vec, step_flag = eval_tree_array(tree, X_step, options)
            if !step_flag
                return Inf
            end
            
            pred_delta_scaled = pred_delta_vec[1]
            
            # 1. Comparaison statistique directe sur l'erreur du Delta normalisé
            total_error += (pred_delta_scaled - y[idx])^2
            count += 1
            
            T_sim_scaled = pred_delta_scaled + T_sim_scaled
            
        end
    end
    
    return total_error / count
end
"""
        #os.makedirs(pysr_dir, exist_ok=True)

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.degree = 1
        self.model = PySRRegressor(
            niterations=niterations,
            populations=populations,
            population_size=population_size,
            parsimony=parsimony,
            maxsize=maxsize,
            parallelism="multithreading",
            binary_operators=["+", "*", "-"], # "conduct(x, y) = x - y",  # Calcule instantanément un gradient thermique (ex: T - Tout)
                                                #"hvac_eff(x, y) = x * (y^2)"  # Multiplie la puissance par le carré d'une température
            unary_operators= ["square"],
            model_selection= "best",
            verbosity= 1,
            elementwise_loss= "L2DistLoss()",
            #loss_function=multi_step_loss,
            crossover_probability= 0.04,
            complexity_of_constants= complexity_of_constants,  # Favorise l'intercept
            complexity_of_variables= 1,

            # 2. DICTIONNAIRE DE TRADUCTION POUR PYTHON (SymPy)
            # Indispensable pour que get_sympy_expression() sache lire tes opérateurs Julia
            #extra_sympy_mappings={
            #    "conduct": lambda x, y: x - y,
            #    "hvac_eff": lambda x, y: x * (y ** 2)
            #},
            nested_constraints={
                "*": {"square": 0},
                "square": {"square": 0, "*": 0}
                #"square": {"square": 0, "*": 0},
                #"exp" : {"exp": 0},
                #"log": {"log": 0}
            },
            constraints= {
        '*': (1,1),       # Empêche de multiplier des blocs trop complexes entre eux
        #'/': (-1, -1)
            }, # {'*': (1, 1)}
            # SÉCURITÉ THERMODYNAMIQUE ABSOLUE
            # On interdit explicitement à PySR de multiplier T avec Q, ou Q avec Q.
            # On autorise uniquement les multiplications logiques (ex: T * Tout pour les pertes).

            #select_k_features= 3,
            batching= False,
            batch_size= 32,

            output_directory= None,
            run_id= self.run_id,
            # On désactive la gestion des dossiers temporaires dynamiques complexes
            temp_equation_file=False,


            **kwargs
        )

    def fit(self, X, y):

        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        # 2. Extraction des coefficients réels du StandardScaler Python
        # Index 0 correspond à la colonne de Tzone
        mean_T = self.scaler_X.mean_[0]
        std_T = self.scaler_X.scale_[0]

        # Pour la cible delta_y (Delta T)
        mean_dy = self.scaler_y.mean_[0]
        std_dy = self.scaler_y.scale_[0]

        # 3. ÉCRITURE DYNAMIQUE DE LA FONCTION JULIA AVEC LES VRAIS CHIFFRES
        multi_step_loss_delta_simple = f"""
        function my_delta_trajectory_loss(tree, dataset, options)
            X = dataset.X  # Matrix{{Float32}} (3 x n_samples) -> [T_scaled, Tout_scaled, Q_scaled]
            y = dataset.y  # Vector{{Float32}} (n_samples) -> Delta_T_scaled_réel
            n_samples = dataset.n

            horizon = 48  # Horizon de 12 heures
            total_error = 0.0
            count = 0

            X_step = Matrix{{Float32}}(undef, 3, 1)
            saut = 1000  
            start_offset = rand(1:saut)

            # Injection directe des coefficients calculés par Python (Format Float32)
            mean_T = {float(mean_T)}f0
            std_T  = {float(std_T)}f0
            mean_dy = {float(mean_dy)}f0
            std_dy  = {float(std_dy)}f0

            for i in start_offset:saut:(n_samples - horizon)
                if i < 1
                    continue
                end

                T_sim_scaled = X[1, i]  

                for h in 0:(horizon - 1)
                    idx = i + h

                    X_step[1, 1] = T_sim_scaled
                    X_step[2, 1] = X[2, idx]  
                    X_step[3, 1] = X[3, idx]  

                    pred_delta_vec, step_flag = eval_tree_array(tree, X_step, options)
                    if !step_flag
                        return Inf
                    end

                    pred_delta_scaled = pred_delta_vec[1]

                    total_error += (pred_delta_scaled - y[idx])^2
                    count += 1

                    # --- LA RÉCURRENCE THERMODYNAMIQUE PARFAITE ---
                    T_sim_real = T_sim_scaled * std_T + mean_T
                    delta_T_real = pred_delta_scaled * std_dy + mean_dy

                    T_sim_real += delta_T_real
                    T_sim_scaled = (T_sim_real - mean_T) / std_T
                end
            end

            return total_error / count
        end
        """
        # 4. On écrase la fonction de perte du modèle avec notre version chiffrée
        #self.model.loss_function = multi_step_loss_delta_simple

        start = time.perf_counter()
        self.model.fit(X_scaled, y_scaled)
        self.is_fitted = True
        return time.perf_counter() - start

    def predict(self, X):
        X_scaled = self.scaler_X.transform(X)
        start = time.perf_counter()
        y_scaled = self.model.predict(X_scaled)
        y_real = self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
        return y_real, time.perf_counter() - start

    def predict_step(self, T, Tout, Q):
        if not hasattr(self, '_fast_fn'):
            expr = self.get_sympy_expression()
            vars = sympy.symbols('T Tout Q')
            # On utilise cse=True pour accélérer le calcul si l'expression est complexe
            self._fast_fn = sympy.lambdify(vars, expr, modules='numpy', cse=True)

        # On appelle la fonction et on force la conversion en float
        # On ajoute [0] ou float() au cas où lambdify retourne un array
        val = self._fast_fn(T, Tout, Q)
        return float(np.array(val).item())

    def predict_24h(self, T_initial, Tout_vector, Q_vector):
        start = time.perf_counter()
        # Utilisation de lambdify pour transformer l'expression SymPy en fonction rapide
        expr = self.get_sympy_expression()
        T, Tout, Q = sympy.symbols('T Tout Q')
        f_fast = sympy.lambdify((T, Tout, Q), expr, 'numpy')

        n = len(Tout_vector)
        t_preds = np.zeros(n)
        curr_t = T_initial
        for k in range(n):
            curr_t = f_fast(curr_t, Tout_vector[k], Q_vector[k])
            t_preds[k] = curr_t
        return t_preds, time.perf_counter() - start

    def get_sympy_expression(self):
        """
        Extrait proprement l'équation symbolique de PySR en appliquant
        la dénormalisation inverse à partir des scalers de l'enveloppe.
        """
        import sympy as sp

        # 1. On force la sélection sur l'indice de la meilleure équation (best_equation)
        # Cela évite de prendre l'équation de complexité 7 si elle est aberrante.
        # Si besoin, tu peux aussi forcer l'index manuellement: idx = 2 pour la complexité 5
        try:
            sympy_expr_scaled = self.model.sympy()
        except:
            # Sécurité si l'appel direct échoue sur ta version
            idx = self.model.equations_.index.max()  # ou force la ligne souhaitée du dataframe
            sympy_expr_scaled = self.model.equations_.loc[idx, 'sympy_format']

        # Variable symboliques normalisées utilisées par PySR (généralement x0, x1, x2)
        # On crée des alias locaux pour faire la substitution de dénormalisation
        x0, x1, x2 = sp.symbols('x0 x1 x2')

        # 2. Récupération des paramètres de normalisation (Moyenne et Écart-type)
        # pour reconstruire l'inversion : Variable_réelle = (Variable_scaled * std) + mean
        mean_X = self.scaler_X.mean_
        std_X = self.scaler_X.scale_
        mean_y = self.scaler_y.mean_[0]
        std_y = self.scaler_y.scale_[0]

        # 3. Formules de substitution inverses
        # T_scaled (x0) = (T_réel - mean) / std  ==> On exprime les scaled en fonction des réels
        T_s, Tout_s, Q_s = sp.symbols('T Tout Q')

        substitutions = {
            x0: (T_s - mean_X[0]) / std_X[0],
            x1: (Tout_s - mean_X[1]) / std_X[1],
            x2: (Q_s - mean_X[2]) / std_X[2]
        }

        # 4. Application de la substitution dans l'arbre scaled
        expr_with_real_X = sympy_expr_scaled.subs(substitutions)

        # 5. Dénormalisation de la variable de sortie (y_real = y_scaled * std_y + mean_y)
        expr_real_final = expr_with_real_X * std_y + mean_y

        # Simplification algébrique finale par SymPy
        return sp.simplify(expr_real_final)

    def get_parameters_dict(self):
        best = self.model.get_best()
        return {"equation": str(best['equation']), "loss": float(best['loss'])}

    def simulate_yearly_24h_gradient_tout(self, X_full, steps_per_day=96):
        # X_full contient [Tzone, Tout, Qhvac]
        x_scaled = self.scaler_X.transform(X_full)

        raw_expr = self.model.sympy()
        x0, x1, x2 = sympy.symbols('x0 x1 x2')
        f_scaled = sympy.lambdify((x0, x1, x2), raw_expr, modules='numpy')

        n = len(X_full)
        predictions_scaled = np.zeros(n)

        T_zone_real_all = X_full[:, 0]
        T_out_real_all = X_full[:, 1]
        x2_scaled = x_scaled[:, 2]  # Q scaled

        mean_T = self.scaler_X.mean_[0]
        std_T = self.scaler_X.scale_[0]

        # Paramètres pour reconstruire le Delta T réel
        mean_dy = self.scaler_y.mean_[0]
        std_dy = self.scaler_y.scale_[0]

        start = time.perf_counter()

        for i in range(0, n, steps_per_day):
            # Initialisation de la journée avec la vraie Tzone
            t_zone_actuelle_real = T_zone_real_all[i]
            predictions_scaled[i] = (t_zone_actuelle_real - mean_T) / std_T

            end = min(i + steps_per_day, n)
            for t in range(i, end - 1):
                # 1. Calcul du DT conduction réel
                dt_conduction_real = t_zone_actuelle_real - T_out_real_all[t]

                # 2. Normalisation flash des entrées de l'arbre
                x0_in = (t_zone_actuelle_real - mean_T) / std_T
                x1_in = (dt_conduction_real - self.scaler_X.mean_[1]) / self.scaler_X.scale_[1]

                # 3. Prédiction du Delta T normalisé
                delta_t_scaled = f_scaled(x0_in, x1_in, x2_scaled[t])

                # 4. Dénormalisation du Delta T en °C
                delta_t_real = delta_t_scaled * std_dy + mean_dy

                # 5. Reconstruction de la température future (Physique de dérive)
                t_zone_actuelle_real = t_zone_actuelle_real + delta_t_real

                # 6. Stockage au format scaled pour la sortie standard
                predictions_scaled[t + 1] = (t_zone_actuelle_real - mean_T) / std_T

        # Dénormalisation globale finale pour coller à ta classe de base
        predictions_real = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

        # Correction finale : comme ta classe de base compare predictions_real à X[:,0] (qui est la Tzone réelle),
        # predictions_real contient maintenant la trajectoire reconstruite °C par °C.
        elapsed = time.perf_counter() - start
        return predictions_real, elapsed

    def simulate_yearly_24h(self, X_full, steps_per_day=96):
        x_scaled = self.scaler_X.transform(X_full)

        # Récupération de l'équation rapide (sur données scaled) utilisé avant
        # equation_str = self.model.get_best()["equation"]
        # f_scaled = eval(f"lambda x0, x1, x2: {equation_str}")
        raw_expr = self.model.sympy()
        # x0, x1, x2 correspondent aux features normalisées transmises à PySR
        x0, x1, x2 = sympy.symbols('x0 x1 x2')
        f_scaled = sympy.lambdify((x0, x1, x2), raw_expr, modules='numpy')

        n = len(X_full)
        predictions_scaled = np.zeros(n)
        # Extraire les colonnes pour un accès direct (vitesse maximale)
        x1_scaled = x_scaled[:, 1]
        x2_scaled = x_scaled[:, 2]
        start = time.perf_counter()

        for i in range(0, n, steps_per_day):
            t_zone_actuelle_scaled = x_scaled[i, 0]
            predictions_scaled[i] = t_zone_actuelle_scaled

            end = min(i + steps_per_day, n)
            for t in range(i, end - 1):
                t_next_scaled = f_scaled(t_zone_actuelle_scaled, x1_scaled[t], x2_scaled[t])
                predictions_scaled[t + 1] = t_next_scaled
                t_zone_actuelle_scaled = t_next_scaled

        predictions_real = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        elapsed = time.perf_counter() - start
        return predictions_real, elapsed

    def simulate_yearly_24h_deltat(self, X_full, steps_per_day=96):
        # X_full contient [Tzone, Tout, Qhvac]
        x_scaled = self.scaler_X.transform(X_full)

        # Extraction de la fonction SymPy
        raw_expr = self.model.sympy()
        x0, x1, x2 = sympy.symbols('x0 x1 x2')
        f_scaled = sympy.lambdify((x0, x1, x2), raw_expr, modules='numpy')

        n = len(X_full)
        predictions_scaled = np.zeros(n)

        # Profils normalisés fixes issus directement du dataset (vitesse maximale)
        x1_scaled_all = x_scaled[:, 1]  # Tout normalisé complet
        x2_scaled_all = x_scaled[:, 2]  # Qhvac normalisé complet

        # Paramètres d'échelle de l'état (Tzone)
        mean_T = self.scaler_X.mean_[0]
        std_T = self.scaler_X.scale_[0]

        # Paramètres d'échelle de la cible (Delta T)
        mean_dy = self.scaler_y.mean_[0]
        std_dy = self.scaler_y.scale_[0]

        start = time.perf_counter()

        for i in range(0, n, steps_per_day):
            # Étape A : Initialisation de la journée (en réel)
            t_zone_actuelle_real = X_full[i, 0]
            predictions_scaled[i] = (t_zone_actuelle_real - mean_T) / std_T

            end = min(i + steps_per_day, n)
            for t in range(i, end - 1):
                # Étape B : Normalisation flash de la température de zone simulée
                x0_in = (t_zone_actuelle_real - mean_T) / std_T

                # Étape C : Prédiction du Delta T normalisé par l'arbre PySR
                # On passe Tout_scaled et Q_scaled réels stables du pas t
                delta_t_scaled = f_scaled(x0_in, x1_scaled_all[t], x2_scaled_all[t])

                # Étape D : Dénormalisation flash du Delta T en °C réels
                delta_t_real = delta_t_scaled * std_dy + mean_dy

                # Étape E : Reconstruction physique de la trajectoire (Dérive cumulative)
                t_zone_actuelle_real = t_zone_actuelle_real + delta_t_real

                # Étape F : Stockage au format normalisé pour coller à ta structure de base
                predictions_scaled[t + 1] = (t_zone_actuelle_real - mean_T) / std_T

        # Inversion globale finale de la trajectoire
        predictions_real = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

        elapsed = time.perf_counter() - start
        return predictions_real, elapsed


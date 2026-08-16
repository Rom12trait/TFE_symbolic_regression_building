import pyomo.environ as pyo
import sympy
import numpy as np


class HVACOptimizer:
    def __init__(self, thermal_model, building_params=None):
        """
        :param thermal_model: Un objet hérité de BaseModel (RC, Polynomial ou PySR)
        :param building_params: Dictionnaire contenant Ph_max, Pc_max, eta_h, eta_c, etc.
        """
        self.thermal_model = thermal_model
        # Paramètres par défaut si non fournis
        self.params = building_params or {
            "Ph_max": 7120.17/1.86,
            "Pc_max": 7120.17/3.73,
            "eta_h": 1.86, #3.8378
            "eta_c": 3.73, #4.3039
            "dt": 0.25,
            "tmin": 20,
            "tmax": 24,
            "penalite_confort": 1000.0  # Coût fictif par °C de violation par quart d'heure
        }
        # Préparation de la fonction Pyomo à partir de SymPy
        self._prepare_thermal_function()

    def _prepare_thermal_function(self):
        if self.thermal_model.__class__.__name__ == 'PySRThermalModel':
            self.sympy_expr = None
        else:
            self.sympy_expr = self.thermal_model.get_sympy_expression()
            self.sym_T, self.sym_Tout, self.sym_Q = sympy.symbols('T Tout Q')

        # On crée une fonction Python qui reconstruit l'expression
        # en utilisant les opérateurs de Pyomo sur les variables Pyomo.
        # modules=[] force l'utilisation des opérateurs standards (+, -, *, **)
        #self.thermal_function = sympy.lambdify(
        #    (self.sym_T, self.sym_Tout, self.sym_Q),
        #    self.expr,
        #    modules=[]
        #)

    def solve(self, prices_vector, Tout_vector, T_initial, year = 'dynamique', mode ='linear'):
        model = pyo.ConcreteModel()

        # 1. Indices
        model.T = pyo.RangeSet(0, 95)
        model.T_instants = pyo.RangeSet(0, 96)

        # 2. Variables
        model.P_heating = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, self.params["Ph_max"]))
        model.P_cooling = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, self.params["Pc_max"]))
        model.T_zone = pyo.Var(model.T_instants, domain=pyo.Reals) #, bounds=(self.params["tmin"], self.params["tmax"])
        model.z = pyo.Var(domain=pyo.Binary)  # Mode de fonctionnement sur la journée

        # --- VARIABLES DE RELÂCHEMENT (SLACKS) POUR LES 96 PAS ---
        # lambda1 (sous les 20°C) et lambda2 (au-dessus des 24°C)
        model.slack_low = pyo.Var(model.T_instants, domain=pyo.NonNegativeReals)
        model.slack_high = pyo.Var(model.T_instants, domain=pyo.NonNegativeReals)

        # 3. Expressions intermédiaires
        def qhvac_rule(m, t):
            return (self.params["eta_h"] * m.P_heating[t]) - (self.params["eta_c"] * m.P_cooling[t])

        model.Qhvac = pyo.Expression(model.T, rule=qhvac_rule)

        def real_cost_rule(m):
            return sum(prices_vector[t] * (m.P_heating[t] + m.P_cooling[t]) / 1000 * self.params["dt"] for t in m.T)

        model.real_cost = pyo.Expression(rule=real_cost_rule)

        def t_min(m,t):
            return  self.params["tmin"] - m.slack_low[t]
        model.low_borne = pyo.Expression(model.T_instants, rule=t_min)
        def t_max(m,t):
            return  self.params["tmax"] + m.slack_high[t]
        model.high_borne = pyo.Expression(model.T_instants, rule=t_max)

        def phvac_total_rule(m, t):
            return m.P_heating[t] + m.P_cooling[t]  # + Pfans_vector[t]

        model.Phvac = pyo.Expression(model.T, rule=phvac_total_rule)

        # 4. Objectif : Minimiser le coût
        def objective_rule(m):
            cout_elec = sum(prices_vector[t] * (m.P_heating[t] + m.P_cooling[t]) / 1000 * self.params["dt"] for t in m.T)
            penalite_tot = sum(self.params["penalite_confort"] * (m.slack_low[t] + m.slack_high[t]) for t in m.T)
            return cout_elec + penalite_tot

        model.cost = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        # --- CONTRAINTES DE CONFORT SOUPLES (96 pas) ---
        # T_zone[t] >= 20 - slack_low[t]  =>  T_zone[t] + slack_low[t] >= 20
        def comfort_low_rule(m, t):
            return m.T_zone[t] + m.slack_low[t] >= self.params["tmin"]

        model.comfort_low = pyo.Constraint(model.T_instants, rule=comfort_low_rule)

        # T_zone[t] <= 24 + slack_high[t] =>  T_zone[t] - slack_high[t] <= 24
        def comfort_high_rule(m, t):
            return m.T_zone[t] - m.slack_high[t] <= self.params["tmax"]

        model.comfort_high = pyo.Constraint(model.T_instants, rule=comfort_high_rule)

        # 5. Contraintes de non-simultanéité (via variable binaire z)
        def heat_excl_rule(m, t):
            return m.P_heating[t] <= self.params["Ph_max"] * m.z

        model.heat_excl = pyo.Constraint(model.T, rule=heat_excl_rule)

        def cool_excl_rule(m, t):
            return m.P_cooling[t] <= self.params["Pc_max"] * (1 - m.z)

        model.cool_excl = pyo.Constraint(model.T, rule=cool_excl_rule)




        # 6. Dynamique thermique (L'automatisation est ici !)
        def thermal_dynamics_rule(m, t):
            if t == 0:
                # On force la valeur numérique pour éviter tout conflit
                val_init = float(np.array(T_initial).flatten()[0])
                return m.T_zone[0] == val_init

            # --- CAS SPÉCIFIQUE PYSR (BRUTE FORCE) ---
            if self.thermal_model.__class__.__name__ == 'PySRThermalModel' and year == 'exp':
                # ÉQUATION EXPONENTIELLE NON LINÉAIRE (Modèle Ipopt)
                return m.T_zone[t] == (
                        0.284656230113608 * pyo.exp(0.143741769601487 * float(Tout_vector[t - 1]) -
                                                    (0.000517122740535654 * m.Qhvac[t - 1] + 0.0141820235751735) ** 2) +
                        21.7321770029164 -
                        3.63209106019174 * pyo.exp(-(0.000517122740535654 * m.Qhvac[t - 1] +
                                                     0.308082902268653 * m.T_zone[t - 1] - 5.84234646606035) ** 2)
                )
            if self.thermal_model.__class__.__name__ == 'PySRThermalModel' and year == 'cube':
                # ÉQUATION QUADRATIQUE/CUBIQUE DE HAUTE PRÉCISION (R² = 0.7040)
                # On utilise les opérateurs natifs de Pyomo (**) compatibles avec Ipopt
                return m.T_zone[t] == (
                        1.73225180795747e-8 * (m.Qhvac[t - 1] ** 2) +
                        0.000367698095454799 * m.Qhvac[t - 1] -
                        0.0153468178001472 * (m.T_zone[t - 1] ** 3) +
                        1.00054755529384 * (m.T_zone[t - 1] ** 2) -
                        20.7438507472603 * m.T_zone[t - 1] +
                        0.0754394535100545 * float(Tout_vector[t - 1]) +
                        156.662748052745
                )
            if self.thermal_model.__class__.__name__ == 'PySRThermalModel' and year == 'dynamique':
                # On écrit l'équation manuellement comme demandé
                return m.T_zone[t] == (
                        0.952113783794049 * m.T_zone[t - 1] +
                        0.0287562819434527 * Tout_vector[t - 1] +
                        0.000135946989606225 * m.Qhvac[t - 1] +
                        0.709746652516824
                )
            if self.thermal_model.__class__.__name__ == 'PySRThermalModel' and year == 'classique':
                return m.T_zone[t] == (
                        0.994436486020745 * m.T_zone[t - 1] +
                        0.000557823545350589 * Tout_vector[t - 1] +
                        1.50530253314913e-8 * m.Qhvac[t - 1] +
                        0.120920836775699
                )

            # On récupère l'expression SymPy
            expr_sympy = self.thermal_model.get_sympy_expression()
            T_sym, Tout_sym, Q_sym = sympy.symbols('T Tout Q')
            # Reconstruction manuelle de l'expression pour Pyomo
            # On remplace les symboles SymPy par les objets Pyomo
            subs = {
                T_sym: m.T_zone[t - 1],
                Tout_sym: float(Tout_vector[t - 1]),
                Q_sym: m.Qhvac[t - 1]
            }

            # Utilisation de la fonction SymPy pour créer l'objet Pyomo
            # sympify=False empêche SymPy de vouloir convertir les variables Pyomo en floats
            try:
                t_next_expr = expr_sympy.subs(subs)
                # On force Pyomo à traiter cela comme une égalité de contrainte
                return m.T_zone[t] - t_next_expr == 0
            except Exception:
                # Si .subs() échoue encore à cause de types internes,
                # on utilise lambdify SANS module (Python pur)
                f = sympy.lambdify((T_sym, Tout_sym, Q_sym), expr_sympy, modules=[])
                return m.T_zone[t] == f(m.T_zone[t - 1], Tout_vector[t - 1], m.Qhvac[t - 1])

        model.dynamics = pyo.Constraint(model.T_instants, rule=thermal_dynamics_rule)

        # 7. Résolution
        # Si le modèle est non-linéaire (PySR ou Quadratic), on force Ipopt
        if mode == 'linear':
            # Gurobi est ok pour le linéaire
            opt = pyo.SolverFactory('gurobi')
        else:
            opt = pyo.SolverFactory('ipopt')
            # Pour tout ce qui est quadratique ou symbolique non-linéaire
            #opt = pyo.SolverFactory('couenne', solver_io='asl', executable = "C:/Users/Corentin/PycharmProjects/thermal-dynamics-symbolic-regression/.venv/Scripts/couenne.exe") #ipopt
            # Temps de calcul maximum de 60 secondes par journée
            #opt.options['time_limit'] = 60
            #opt = pyo.SolverFactory('asl')
            #opt.set_executable(
             #   "C:/Users/Corentin/PycharmProjects/thermal-dynamics-symbolic-regression/.venv/Scripts/couenne.exe")
            #opt.options['solver'] = 'couenne'
            # Options reconnues par l'interface ASL native

        results = opt.solve(model, tee=True)
        return model, results

    def post_process_hvac_simultaneity(self, model_resolved, prices_vector, eta_h=1.86, eta_c=3.73, dt=0.25):
        """
        Supprime la simultanéité chauffage/climatisation induite par Ipopt (notamment lors des prix négatifs),
        ajuste les puissances électriques pour conserver le Qhvac net intact (préservant T_zone),
        et recalcule les coûts corrigés.
        """

        T_indices = list(model_resolved.T)

        # 1. Listes de réception pour les profils corrigés
        p_heat_corr = []
        p_cool_corr = []
        cout_instant_corr = []

        total_cost_corr = 0.0

        for t in T_indices:
            # Récupération des valeurs optimisées par Ipopt
            ph_raw = pyo.value(model_resolved.P_heating[t])
            pc_raw = pyo.value(model_resolved.P_cooling[t])
            prix_t = prices_vector[t]

            # 2. Calcul du Qhvac net visé par le solveur
            q_hvac_net = (eta_h * ph_raw) - (eta_c * pc_raw)

            # 3. Application de l'équivalence physique sans simultanéité
            if q_hvac_net > 1e-3:  # Besoin de Chauffage net (seuil pour filtrer le bruit numérique)
                ph_new = q_hvac_net / eta_h
                pc_new = 0.0
            elif q_hvac_net < -1e-3:  # Besoin de Climatisation nette
                ph_new = 0.0
                pc_new = -q_hvac_net / eta_c
            else:  # Flux quasi nul ou nul
                ph_new = 0.0
                pc_new = 0.0

            # 4. Recalcul du coût pour ce pas de temps (Wh -> kW => /1000)
            # Cout = Puissance(W) * prix(€/kWh) / 1000 * dt(h)
            puissance_appelee = ph_new + pc_new
            cost_t = prix_t * (puissance_appelee) / 1000.0 * dt

            # Stockage
            p_heat_corr.append(ph_new)
            p_cool_corr.append(pc_new)
            cout_instant_corr.append(cost_t)

            total_cost_corr += cost_t

        # 5. Extraction de T_zone (inchangée car Qhvac est conservé)
        t_zone_opt = [pyo.value(model_resolved.T_zone[t]) for t in model_resolved.T_instants]

        return p_heat_corr, p_cool_corr, t_zone_opt, cout_instant_corr, total_cost_corr

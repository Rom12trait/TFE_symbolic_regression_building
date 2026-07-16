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

        # --- DYNAMIQUE THERMIQUE (MÉTHODE SYMBOLE DIRECTE) ---
        # On récupère l'expression SymPy
        expr_sympy = self.thermal_model.get_sympy_expression()
        T_sym, Tout_sym, Q_sym = sympy.symbols('T Tout Q')

        # 6. Dynamique thermique (L'automatisation est ici !)
        def thermal_dynamics_rule(m, t):
            if t == 0:
                # On force la valeur numérique pour éviter tout conflit
                val_init = float(np.array(T_initial).flatten()[0])
                return m.T_zone[0] == val_init

            # --- CAS SPÉCIFIQUE PYSR (BRUTE FORCE) ---
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
            # Pour tout ce qui est quadratique ou symbolique non-linéaire
            opt = pyo.SolverFactory('ipopt')

        results = opt.solve(model)
        return model, results
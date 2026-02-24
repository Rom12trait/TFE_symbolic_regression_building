from eppy.modeleditor import IDF
import numpy as np
import math
from pathlib import Path
import json


def load_idf(idf_path, idd_path):
    IDF.setiddname(idd_path)
    return IDF(idf_path)

#obtenir les surfaces extérieures
def get_external_surfaces(idf):
    surfaces = []
    for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
        if s.Outside_Boundary_Condition.lower() == "outdoors":
            surfaces.append(s)
    return surfaces



#calculs des U
def compute_u_value(construction, idf):
    r_total = 0.0
    # Outside layer
    layer_names = [construction.Outside_Layer]
    # Layers 2 → 10
    for i in range(2, 11):
        layer = getattr(construction, f"Layer_{i}", None)
        if layer:
            layer_names.append(layer)

    for layer_name in layer_names:
        mat = (
            idf.getobject("MATERIAL", layer_name)
            or idf.getobject("MATERIAL:NOMASS", layer_name)
            or idf.getobject("MATERIAL:AIRGAP", layer_name)
        )

        if mat is None:
            print(f"⚠️ Matériau non reconnu : {layer_name}")
            continue

        mat_type = mat.obj[0].upper()
        if mat_type == "MATERIAL":
            r_total += mat.Thickness / mat.Conductivity

        elif mat_type == "MATERIAL:AIRGAP":
            r_total += mat.Thermal_Resistance

        elif mat_type == "MATERIAL:NOMASS":
            r_total += mat.Thermal_Resistance
        else:
            print(f"⚠️ Type matériau ignoré : {mat_type}")
            continue

    if r_total == 0:
        raise ValueError("Résistance totale nulle → problème construction")
    return 1.0 / r_total


def compute_h_windows(idf):
    h = 0.0
    for w in idf.idfobjects["window"]:
            u = 1.590008
            length = float(w.Length)
            height = float(w.Height)
            multiplier = float(w.Multiplier)
            h += u * length * height* multiplier

    print(f"window h= {h:.4f}")
    return h



def distance_2d(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def get_surface_vertices(surface):
    vertices = []
    n = int(surface.Number_of_Vertices)
    for i in range(1, n + 1):
        x = getattr(surface, f"Vertex_{i}_Xcoordinate")
        y = getattr(surface, f"Vertex_{i}_Ycoordinate")
        vertices.append((x, y))
    return vertices

def compute_polygon_perimeter(vertices):
    perimeter = 0.0
    for i in range(len(vertices)):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % len(vertices)]
        perimeter += distance_2d(p1, p2)
    return perimeter

def get_heated_zones(idf):
    zones = set()

    for eq in idf.idfobjects["ZONECONTROL:THERMOSTAT"]:
        zones.add(eq.Zone_or_ZoneList_Name)

    return list(zones)

def compute_slab_perimeter(idf, heated_zones=None):
    """
    idf           : objet eppy IDF
    heated_zones  : liste des noms de zones chauffées (optionnel)
    """
    total_perimeter = 0.0

    for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"]:

        # uniquement planchers
        if s.Surface_Type.lower() != "Floor":
            continue

        # en contact avec le sol
        if s.Outside_Boundary_Condition.lower() != "GroundSlabPreprocessorAverage":
            continue

        #uniquement zones chauffées (si spécifié)
        if heated_zones is not None and s.Zone_Name not in heated_zones:
            continue

        vertices = get_surface_vertices(s)
        perimeter = compute_polygon_perimeter(vertices)
        total_perimeter += perimeter
        print(f"perimeter: {total_perimeter:.4f}")
    return total_perimeter



def compute_slab_loss(idf, f_slab=0.4):
    heated_zones = get_heated_zones(idf)
    perimeter = compute_slab_perimeter(idf, heated_zones)
    print("perimeter: ", perimeter)
    x= perimeter * f_slab
    print(f"slabloss = {x:.4f}")
    return x

def compute_h_wall(idf):
    h = 0.0
    surfaces = get_external_surfaces(idf)

    for s in surfaces:
        construction = idf.getobject("CONSTRUCTION", s.Construction_Name)
        u = compute_u_value(construction, idf)
        a = s.area
        h += u * a
        print(
            s.Name,
            f"A={s.area:.2f}",
            f"U={u:.3f}",
            f"H={u * s.area:.2f}"
        )

    return h

#ventilation et infiltration
AIR_DENSITY = 1.2
CP_AIR = 1005

def compute_h_vent(volume, ach):
    v_dot = ach * volume / 3600
    x= AIR_DENSITY * CP_AIR * v_dot
    print(f"x ={x:.4f}")
    return x

def compute_r(idf):

    H_trans = compute_h_wall(idf) + compute_h_windows(idf) + compute_slab_loss(idf)
    H_vent = compute_h_vent(volume=350, ach=0.4)
    R = 1 / (H_trans + H_vent)
    return R

#Calcul de C
def slab_capacity(thickness=0.12, rho=2300, cp=880):
    x = 12.1330909462833
    y = 9.09981820971244
    a_slab= x*y
    volume = a_slab * thickness
    return rho * cp * volume



def wall_capacity(area, thickness, rho, cp, alpha=0.25):
    return alpha * rho * cp * area * thickness

def compute_total_capacity(slab_c, wall_c_list):
    return slab_c + sum(wall_c_list)


#class RC

class RCmodel:
    def __init__(self,R,C,dt,random_state):
        self.R = R
        self.C = C
        self.dt = dt
        self.random_state = random_state

        # Paramètres physiques
        self.Ca = 6.8e5
        self.Cm = 2.42e7
        self.R1 = 0.0010
        self.R2 = 0.0086

        # Matrices continues
        self.A = np.array([
            [-1 / (self.Ca * self.R1), 1 / (self.Ca * self.R1)],
            [1 / (self.Cm * self.R1), -(1 / (self.Cm * self.R1) + 1 / (self.Cm * self.R2))]
        ])

        self.B = np.array([
            [0, 1 / self.Ca],
            [1 / (self.Cm * self.R2), 0]
        ])

        self.I = np.eye(2)

        # Pré-calcul matrice inverse (constante)
        self.M = np.linalg.inv(self.I - self.dt * self.A)

# coefficients
    @property
    def a(self):
        return 1- (self.dt / (self.R * self.C))
    @property
    def b(self):
        return self.dt / (self.R * self.C)
    @property
    def c(self):
        return self.dt / self.C

#predictions
    def predict(self, Tzone, Tout, Qhvac):

        n = len(Tout)
        T_pred = np.zeros(n)
        T_pred[0] = Tzone[0]

        for k in range(n-1):

            T_pred[k+1] = (
                    self.a * Tzone[k]
                    + self.b * Tout[k]
                    + self.c * Qhvac[k]
            )

        return T_pred

    def predict_free(self,Tzone, Tout, Qhvac):

        n = len(Tout)
        T_pred = np.zeros(n)
        T_pred[0] = (
                self.a * Tzone[0]
                + self.b * Tout[0]
                + self.c * Qhvac[0]
        )
        for k in range(n-1):

            T_pred[k+1] = (
                    self.a * T_pred[k]
                    + self.b * Tout[k]
                    + self.c * Qhvac[k]
            )

        return T_pred

    def simulate_by_day(self, T_data, Tout, Qhvac, steps_per_day=96):

        n = len(T_data)
        T_pred = np.zeros(n)

        for start in range(0, n, steps_per_day):

            end = min(start + steps_per_day, n)

            T_pred[start] = T_data[start]

            for k in range(start, end - 1):
                T_pred[k + 1] = (
                        self.a * T_pred[k]
                        + self.b * Tout[k]
                        + self.c * Qhvac[k]
                )

        return T_pred

    def simulate_2r2c(self, Tout, Qhvac, Ta_init=20.0, Tm_init=20.0):
        # Capacité thermique air (J/K)
        Ca = 6.8e5

        # Capacité thermique masse (J/K)
        Cm = 2.42e7

        # Résistance air ↔ masse (K/W)
        R1 = 0.0010

        # Résistance masse ↔ extérieur (K/W)
        R2 = 0.0087
        """
        Simulation du modèle 2R2C

        Parameters
        ----------
        Tout : array
            Température extérieure [°C]
        Qhvac : array
            Puissance HVAC [W]
        Ta_init : float
            Température air initiale
        Tm_init : float
            Température masse initiale

        Returns
        -------
        Ta_pred : array
            Température air prédite
        Tm_pred : array
            Température masse prédite
        """

        n = len(Tout)

        Ta_pred = np.zeros(n)
        Tm_pred = np.zeros(n)

        Ta_pred[0] = Ta_init
        Tm_pred[0] = Tm_init

        for k in range(n - 1):
            # -------- AIR --------
            dTa = (
                    (Tm_pred[k] - Ta_pred[k]) / R1
                    + Qhvac[k]
            )

            Ta_pred[k + 1] = Ta_pred[k] + (self.dt / Ca) * dTa

            # -------- MASSE --------
            dTm = (
                    (Ta_pred[k] - Tm_pred[k]) / R1
                    + (Tout[k] - Tm_pred[k]) / R2
            )

            Tm_pred[k + 1] = Tm_pred[k] + (self.dt / Cm) * dTm

        return Ta_pred, Tm_pred

    def simulate_euler_implicite(self, Tout, Qhvac, Ta_init=20.0, Tm_init=20.0):

        n = len(Tout)
        X = np.zeros((2, n))

        X[:, 0] = [Ta_init, Tm_init]

        for k in range(n - 1):
            U_next = np.array([Tout[k + 1], Qhvac[k + 1]])

            X[:, k + 1] = self.M @ (
                    X[:, k] + self.dt * (self.B @ U_next)
            )

        Ta = X[0, :]
        Tm = X[1, :]

        return Ta, Tm


    def benchmark(self, Tzone, timestep_minutes=15, day_step=4):

        steps_per_hour = 60 // timestep_minutes
        steps_per_day = 24 * steps_per_hour  # 96 si 15 min
        shift_steps = steps_per_day  # horizon 24h

        values = Tzone

        y_true = []
        y_pred = []

        total_days = len(values) // steps_per_day

        for day in range(0, total_days - 1, day_step):

            start_idx = day * steps_per_day
            end_idx = start_idx + steps_per_day

            # prédiction = valeurs du jour courant
            pred_day = values[start_idx:end_idx]

            # vérité = valeurs 1 pas de temps plus tard
            true_day = values[start_idx + 1: end_idx + 1]

            if len(true_day) == steps_per_day:
                y_pred.append(pred_day)
                y_true.append(true_day)

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)

        return y_true, y_pred

    def get_parameters_dict(self):
        """
        Retourne tous les paramètres nécessaires
        à la reproduction du modèle.
        """
        return {
            "model_type": "RC",
            "R": self.R,
            "C": self.C,
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "random_state": self.random_state,
        }

    def save_parameters(self, directory):
        """
        Sauvegarde les paramètres du modèle
        dans un fichier JSON.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        params = self.get_parameters_dict()

        file_path = directory / "rc_model_parameters.json"

        with open(file_path, "w") as f:
            json.dump(params, f, indent=4)

        print(f"Paramètres sauvegardés dans : {file_path}")

    def save_modelrc(self,filepath, R, C, dt, a, b, c):
        data = {
            "model": "RC",
            "R_K_per_W": R,
            "C_J_per_K": C,
            "dt_s": dt,
            "a": a,
            "b": b,
            "c": c
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)



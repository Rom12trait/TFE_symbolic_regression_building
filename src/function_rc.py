from eppy.modeleditor import IDF
import numpy as np
import math
from pathlib import Path
import json
import time

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

    h_trans = compute_h_wall(idf) + compute_h_windows(idf) + compute_slab_loss(idf)
    h_vent = compute_h_vent(volume=350, ach=0.4)
    r = 1 / (h_trans + h_vent)
    return r

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
    def __init__(self,res,capa,dt,random_state):
        self.R = res
        self.C = capa
        self.dt = dt
        self.random_state = random_state


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
    def predict(self, t_zone, t_out, q_hvac):

        n = len(t_out)
        t_pred = np.zeros(n)
        t_pred[0] = t_zone[0]
        start = time.perf_counter()

        for k in range(n-1):

            t_pred[k+1] = (
                    self.a * t_zone[k]
                    + self.b * t_out[k]
                    + self.c * q_hvac[k]
            )
        elapsed = time.perf_counter() - start

        return t_pred, elapsed

    def simulate_by_day(self, t_data, t_out, q_hvac, steps_per_day=96):

        n = len(t_data)
        t_pred = np.zeros(n)
        start_time = time.perf_counter()

        for start in range(0, n, steps_per_day):

            end = min(start + steps_per_day, n)

            t_pred[start] = t_data[start]

            for k in range(start, end - 1):
                t_pred[k + 1] = (
                        self.a * t_pred[k]
                        + self.b * t_out[k]
                        + self.c * q_hvac[k]
                )
        elapsed = time.perf_counter() - start_time

        return t_pred, elapsed

    def benchmark(self, t_zone, timestep_minutes=15, day_step=4):

        steps_per_hour = 60 // timestep_minutes
        steps_per_day = 24 * steps_per_hour  # 96 si 15 min

        values = t_zone

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

    def save_modelrc(self,filepath, res, capa, dt, a, b, c):
        data = {
            "model": "RC",
            "R_K_per_W": res,
            "C_J_per_K": capa,
            "dt_s": dt,
            "a": a,
            "b": b,
            "c": c
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)



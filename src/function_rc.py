from eppy.modeleditor import IDF
import numpy as np
import math


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
        if w.Construction_Name.lower() == "Exterior Window":
            u = 1.590008
            h += u * w.Length * w.Height
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
    return perimeter * f_slab

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
    return AIR_DENSITY * CP_AIR * v_dot

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
    def __init__(self,R,C,dt):
        self.R = R
        self.C = C
        self.dt = dt

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
        T_pred[0] = Tzone[0]

        for k in range(n-1):

            T_pred[k+1] = (
                    self.a * T_pred[k]
                    + self.b * Tout[k]
                    + self.c * Qhvac[k]
            )

        return T_pred

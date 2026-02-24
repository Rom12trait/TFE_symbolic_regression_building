"""
Calcul automatique du modèle 1R1C depuis un fichier IDF
Nécessite : pip install eppy
"""
from src.function_rc import load_idf, slab_capacity, compute_h_wall, compute_h_vent, compute_slab_loss, compute_h_windows

from src.function_rc import load_idf
from eppy.modeleditor import IDF
import os

# ==========================
# 1) CHEMINS
# ==========================

idf = load_idf(
    "dataset/modèle habitation/US+SF+CZ4C+hp+slab+IECC_2024_Brussels_airport_V2420.idf",
    "C:/Users/Corentin/energyplus/Energy+.idd"
)

# ==========================
# 2) FONCTIONS UTILITAIRES
# ==========================

def polygon_area(coords):
    """Calcule l'aire d'un polygone 3D projeté (méthode shoelace simplifiée XY)."""
    area = 0
    for i in range(len(coords)):
        x1, y1 = coords[i][0], coords[i][1]
        x2, y2 = coords[(i + 1) % len(coords)][0], coords[(i + 1) % len(coords)][1]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2


def get_construction_uvalue(construction_name):
    """Calcule U = 1 / somme(R couches)"""
    construction = idf.getobject("CONSTRUCTION", construction_name)
    r_total = 0

    for layer in construction.fieldvalues[2:]:
        if layer == "":
            continue

        mat = idf.getobject("MATERIAL", layer)
        if mat:
            thickness = float(mat.Thickness)
            conductivity = float(mat.Conductivity)
            r_total += thickness / conductivity
            continue

        mat_nomass = idf.getobject("MATERIAL:NOMASS", layer)
        if mat_nomass:
            r_total += float(mat_nomass.Thermal_Resistance)

    if r_total == 0:
        return None

    return 1 / r_total


# ==========================
# 3) SURFACES FENÊTRES
# ==========================

A_window_total = 0

for win in idf.idfobjects["WINDOW"]:
    length = float(win.Length)
    height = float(win.Height)
    multiplier = float(win.Multiplier)
    A_window_total += length * height * multiplier

# ==========================
# 4) SURFACES MURS + PLAFOND
# ==========================

A_wall_total = 0
A_ceiling = 0
U_wall = None
U_ceiling = None
C_wall_surface = 150000      # hypothèse inertie mur
C_ceiling_surface = 35000    # hypothèse plafond

for surf in idf.idfobjects["BUILDINGSURFACE:DETAILED"]:

    coords = []
    for i in range(1, int(surf.Number_of_Vertices) + 1):
        x = float(getattr(surf, f"Vertex_{i}_Xcoordinate"))
        y = float(getattr(surf, f"Vertex_{i}_Ycoordinate"))
        z = float(getattr(surf, f"Vertex_{i}_Zcoordinate"))
        coords.append((x, y, z))

    area = polygon_area(coords)

    if surf.Surface_Type.lower() == "wall" and surf.Outside_Boundary_Condition.lower() == "outdoors":
        A_wall_total += area
        U_wall = get_construction_uvalue(surf.Construction_Name)

    if surf.Surface_Type.lower() == "ceiling":
        A_ceiling += area
        U_ceiling = get_construction_uvalue(surf.Construction_Name)

# Surface mur opaque réelle
A_wall_opaque = A_wall_total - A_window_total

# ==========================
# 5) CALCUL UA
# ==========================

U_window = 1.590008  # si simple glazing défini dans IDF

UA_walls = U_wall * A_wall_opaque
UA_ceiling = U_ceiling * A_ceiling
UA_windows = U_window * A_window_total

UA_total = UA_walls + UA_ceiling + UA_windows

# ==========================
# 6) R, C et TAU
# ==========================

R_eq = 1 / UA_total

C_walls = C_wall_surface * A_wall_opaque
C_ceiling = C_ceiling_surface * A_ceiling
C_eq = C_walls + C_ceiling
CEQUIVALENT = slab_capacity()
tau_seconds = R_eq * CEQUIVALENT
tau_hours = tau_seconds / 3600
tau_days = tau_hours / 24

# ==========================
# 7) RÉSULTATS
# ==========================

print("===== RÉSULTATS AUTOMATIQUES =====")
print(f"Surface fenêtres : {A_window_total:.2f} m²")
print(f"Surface murs totaux : {A_wall_total:.2f} m²")
print(f"Surface murs opaques : {A_wall_opaque:.2f} m²")
print(f"Surface plafond : {A_ceiling:.2f} m²")
print(f"UA total : {UA_total:.2f} W/K")
print(f"R_eq : {R_eq:.6f} K/W")
print(f"C_eq : {C_eq:.2e} J/K")
print(f"myC : {CEQUIVALENT:.2e} J/K")

print(f"Constante de temps : {tau_hours:.2f} h ({tau_days:.2f} jours)")
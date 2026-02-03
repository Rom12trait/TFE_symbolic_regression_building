from eppy.modeleditor import IDF

def load_idf(idf_path, idd_path):
    IDF.setiddname(idd_path)
    return IDF(idf_path)

#obtenir toutes les surfaces extérieures
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
        mat = idf.getobject("MATERIAL", layer_name)
        if not mat:
            continue

        r_layer = mat.Thickness / mat.Conductivity
        r_total += r_layer

    return 1.0 / r_total



def compute_h_windows(idf):
    h = 0.0
    for w in idf.idfobjects["window"]:
        if w.Construction_Name.lower() == "Exterior Window":
            u = 1.590008
            h += u * w.Length * w.Height
    return h

def compute_slab_loss(x,y, f_slab=0.4):
    perimeter = 2*(x + y)
    return perimeter * f_slab

def compute_h_wall(idf):
    h = 0.0
    surfaces = get_external_surfaces(idf)

    for s in surfaces:
        construction = idf.getobject("CONSTRUCTION", s.Construction_Name)
        u = compute_u_value(construction, idf)
        a = s.area
        h += u * a

    return h

#ventilation et infiltration
AIR_DENSITY = 1.2
CP_AIR = 1005

def compute_h_vent(volume, ach):
    v_dot = ach * volume / 3600
    return AIR_DENSITY * CP_AIR * v_dot



#Calcul de C
def slab_capacity(a_slab, thickness=0.12, rho=2300, cp=880):
    volume = a_slab * thickness
    return rho * cp * volume



def wall_capacity(area, thickness, rho, cp, alpha=0.25):
    return alpha * rho * cp * area * thickness

def compute_total_capacity(slab_c, wall_c_list):
    return slab_c + sum(wall_c_list)
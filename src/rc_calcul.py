
from src.function_rc import load_idf, slab_capacity, compute_h_wall, compute_h_vent, compute_slab_loss, compute_h_windows

idf = load_idf(
    "../dataset/ModeleHabitation/US+SF+CZ4C+hp+slab+IECC_2024_Brussels_airport_V2420.idf",
    "C:/Users/Corentin/energyplus/Energy+.idd"
)

x= 12.1330909462833
y= 9.09981820971244


H_trans = compute_h_wall(idf) + compute_h_windows(idf) + compute_slab_loss(idf)
print(f"H_trans = {H_trans:.4f}")
H_vent = compute_h_vent(volume=560, ach=0)
print(f"H_vent = {H_vent:.4f}")
R = 1 / (H_trans + H_vent)

C = slab_capacity()

print(f"R = {R:.4f} K/W")
print(f"C = {C/1e6:.2f} MJ/K")
print(f"RC = {R*C/3600/24:.2f} days")




from function_rc import load_idf, slab_capacity, compute_h_wall, compute_h_vent, compute_slab_loss, compute_h_windows

idf = load_idf(
    "data/US+SF+CZ4C+hp+slab+IECC_2024_Brussels_airport_V2420.idf",
    "C:/Users/Corentin/energyplus/Energy+.idd"
)

x= 12.1330909462833
y= 9.09981820971244


H_trans = compute_h_wall(idf) + compute_h_windows(idf) + compute_slab_loss(x,y)
H_vent = compute_h_vent(volume=350, ach=0.4)

R = 1 / (H_trans + H_vent)

A_slab = x*y
C = slab_capacity(A_slab)

print(f"R = {R:.4f} K/W")
print(f"C = {C/1e6:.2f} MJ/K")
print(f"RC = {R*C/3600/24:.2f} days")



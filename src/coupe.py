from src.communs import plot_model_cuts_comparison

# 1. Tu entres tes équations dénormalisées exactes du rapport régression linéaire.
equation_lineaire = lambda T, Tout, Q: 0.9580522597868761 * T + 0.00827508874643919 * Tout + 3.6005573137969787e-06 * Q + 0.8155276129783608

equation_quadratique = lambda T, Tout, Q: (
    6.51755e-6 * Q
    - 0.0008876 * T * Tout
    + 0.965467 * T
    + 0.00038276 * Tout**2
    + 0.0204 * Tout
    + 0.67623
)

# 2. Tu lances le tracé en donnant directement les fonctions et le dossier cible
plot_model_cuts_comparison(
    model_linear_func=equation_lineaire,
    model_quadratic_func=equation_quadratique,
    output_dir="../apiV4/comparaisons_modeles_finales"
)
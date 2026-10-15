"""Case modules. A case defines configure(), reference_profiles(z), exner_profile(theta0, z),
init_perturbations(s, g) and print_rh_profile(s, g); Simulation and State dispatch on cfg.CASE."""

import importlib

CASES = ("land_cu", "single_cu")


def get(name: str):
    """Case module for cfg.CASE, or None for the built-in "default" sounding."""
    return importlib.import_module(f"cases.{name}") if name in CASES else None

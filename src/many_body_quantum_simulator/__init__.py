"""Two-mode bosonic many-body simulator with a Chebyshev propagator."""

from importlib.metadata import version

from .two_mode import (
    Sx,
    Sy,
    Sz,
    get_H_matrix,
    get_H_tridiagonal,
    apply_H_vectorized,
    get_energy,
    spectral_bounds,
    check_spectral_window,
    evaluate_state,
    evolve_state,
    get_ground_state,
)
from .representations import (
    spin_coherent_state,
    husimi_q,
    wigner,
)

__all__ = [
    "Sx",
    "Sy",
    "Sz",
    "get_H_matrix",
    "get_H_tridiagonal",
    "apply_H_vectorized",
    "get_energy",
    "spectral_bounds",
    "check_spectral_window",
    "evaluate_state",
    "evolve_state",
    "get_ground_state",
    "spin_coherent_state",
    "husimi_q",
    "wigner",
]

__version__ = version("many-body-quantum-simulator")

# many-body-quantum-simulator

[![Tests](https://github.com/serranolio/many-body-quantum-simulator/actions/workflows/tests.yml/badge.svg)](https://github.com/serranolio/many-body-quantum-simulator/actions/workflows/tests.yml)

Quantum many-body simulation of two-mode bosonic systems with conserved
particle number, such as two-component or double-well Bose–Einstein
condensates (bosonic Josephson junctions), the Lipkin–Meshkov–Glick model, and
one-axis-twisting spin squeezing. States are represented in the Fock basis and
propagated in real or imaginary time with a Chebyshev expansion of the
evolution operator.

## Model

For $N$ bosons distributed between two modes $L$ and $R$, the state is
expanded in the Fock basis $|n_L, n_R = N - n_L\rangle$, $n_L = 0, \dots, N$
(array index $i = n_L$). The normalized collective spin operators are

$$
S_x = \frac{a_L^\dagger a_R + a_R^\dagger a_L}{N}, \qquad
S_y = \frac{a_R^\dagger a_L - a_L^\dagger a_R}{iN}, \qquad
S_z = \frac{n_R - n_L}{N},
$$

with eigenvalues in $[-1, 1]$ and $[S_x, S_y] = \tfrac{2i}{N} S_z$. The
Hamiltonian ($\hbar = 1$) is

$$
H(t) = \Lambda(t)\, S_z^2 + \Omega(t)\, S_x + \delta(t)\, S_z ,
$$

where $\Lambda$ is the nonlinear (interaction) term, $\Omega$ the coupling
and $\delta$ the detuning. $H$ is real and tridiagonal in the Fock basis, so
memory and cost per time step scale linearly with $N$.

## Numerical method

Each time step applies $e^{-iH\,\Delta t}$ (real time) or
$e^{-H\,\Delta\tau}$ (imaginary time) through a Chebyshev expansion
[Tal-Ezer & Kosloff, J. Chem. Phys. 81, 3967 (1984)]:

- **Spectral window.** $H$ is mapped onto $[-1, 1]$ using a window
  $[E_\mathrm{min}, E_\mathrm{max}]$ that must contain its spectrum. By default
  the window is computed at every step from the analytic bound
  $E \in [\min(0,\Lambda) - |\Omega| - |\delta|,\ \max(0,\Lambda) + |\Omega| + |\delta|]$.
  A tighter window can be obtained with `spectral_bounds(..., exact=True)`;
  explicit windows that are provably too narrow raise a `RuntimeWarning`.
- **Adaptive truncation.** Bessel coefficients are added until they fall below
  `tol` (default `1e-14`) relative to the largest one.
- **Imaginary-time sub-stepping.** Large imaginary-time steps are split so that
  $|z| = (E_\mathrm{max} - E_\mathrm{min})\,\Delta\tau/2 \le 10$, avoiding loss of
  precision from cancellation between exponentially large terms.

## Installation

The project uses [pixi](https://pixi.sh) to manage the environment:

```bash
git clone https://github.com/serranolio/many-body-quantum-simulator.git
cd many-body-quantum-simulator
pixi install
pixi shell        # or prefix commands with `pixi run`
```

The package is installed in editable mode, so changes in `src/` take effect
immediately.

## Usage

```python
import numpy as np
from many_body_quantum_simulator import (
    Sz, evolve_state, get_ground_state, get_energy, spectral_bounds)

N = 200                                  # number of bosons
nonlinear, coupling, detuning = 0.5, 1.0, 0.0

# Ground state by imaginary-time evolution from a random initial state
rng = np.random.default_rng(0)
psi0 = rng.normal(size=N + 1) + 0j
psi0 /= np.linalg.norm(psi0)
ground = get_ground_state(psi0, nonlinear, coupling, detuning,
                          t_step=5.0, steps=200)
print("E0 =", get_energy(ground, nonlinear, coupling, detuning).real)

# Real-time dynamics; parameters are functions of time
trace = evolve_state(ground,
                     nonlinear=lambda t: nonlinear,
                     coupling=lambda t: coupling,
                     detuning=lambda t: 0.1 * np.sin(t),
                     t_step=0.05, steps=2000, snapshots=200)
sz = [(s.conj() @ Sz(s)).real for s in trace]

# Tight spectral window for constant parameters
e_min, e_max = spectral_bounds(N, nonlinear, coupling, detuning, exact=True)
```

Avoid starting imaginary-time evolution from a state with a tiny overlap with
the ground state (e.g. a uniform superposition for some parameters):
convergence is then slow and sensitive to round-off.

### Main functions

| Function | Description |
|---|---|
| `Sx(state)`, `Sy(state)`, `Sz(state)` | Apply the normalized spin operators |
| `get_H_matrix(N, nonlinear, coupling, detuning)` | Sparse (CSR) Hamiltonian; use `.toarray()` for dense |
| `get_energy(state, nonlinear, coupling, detuning)` | Energy expectation value of a normalized state |
| `spectral_bounds(N, nonlinear, coupling, detuning, exact=False)` | Interval containing the spectrum of $H$ |
| `evolve_state(state, nonlinear, coupling, detuning, t_step, steps, ...)` | Real- or imaginary-time evolution with time-dependent parameters; returns `snapshots` states |
| `get_ground_state(state, nonlinear, coupling, detuning, ...)` | Ground state by imaginary-time evolution |

## Tests

The tests compare the propagator against exact matrix exponentiation and
diagonalization:

```bash
pixi run test
```

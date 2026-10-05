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

### With pip

To use the package in an existing Python (>= 3.11) environment:

```bash
pip install git+https://github.com/serranolio/many-body-quantum-simulator.git
```

This installs the core dependencies (NumPy, SciPy, Numba).

### For development, with pixi

The project uses [pixi](https://pixi.sh) to manage reproducible environments:

```bash
git clone https://github.com/serranolio/many-body-quantum-simulator.git
cd many-body-quantum-simulator
pixi install
pixi shell        # or prefix commands with `pixi run`
```

The package is installed in editable mode, so changes in `src/` take effect
immediately. Two environments are defined:

| Environment | Contents | Activate with |
|---|---|---|
| `default` | core dependencies and pytest | `pixi shell` |
| `dev` | default plus Matplotlib, Jupyter, pandas and joblib for notebooks and analysis | `pixi shell -e dev` |

For example, `pixi run -e dev jupyter notebook` starts Jupyter in the `dev`
environment.

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
| `spin_coherent_state(N, theta, phi)` | Spin-coherent state pointing along $(\theta, \phi)$ |
| `husimi_q(state, theta, phi)` | Husimi Q function on the Bloch sphere |
| `wigner(state, theta, phi)` | Spin (Agarwal) Wigner function on the Bloch sphere |

## Phase-space representations

A state of $N$ bosons in two modes is a spin $j = N/2$ state, so it can be
visualized on the Bloch sphere. A point $(\theta, \phi)$ corresponds to the
direction $\langle \mathbf{S} \rangle = (\sin\theta\cos\phi,
\sin\theta\sin\phi, \cos\theta)$; $\theta = 0$ means all bosons in the right
mode.

- `husimi_q`: $Q(\theta, \phi) = \frac{2j+1}{4\pi}
  |\langle \theta, \phi | \psi \rangle|^2$, the overlap with spin-coherent
  states. Always non-negative; cost $O(N)$ per grid point.
- `wigner`: the spin Wigner function of Agarwal (1981). It can be negative, e.g.
  for cat states. Cost $O(N^2)$ per grid point: about 1 s for $N = 1000$ on a
  $100 \times 200$ grid.

Both take a pure state and 1-D grids of `theta` and `phi`, return an array of
shape `(len(theta), len(phi))`, and are normalized to 1 over the sphere
($d\Omega = \sin\theta\, d\theta\, d\phi$).

```python
import numpy as np
import many_body_quantum_simulator as mbqs

N = 100
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(-np.pi, np.pi, 200)

# Cat state: superposition of all particles in the right and in the left mode
cat = mbqs.spin_coherent_state(N, 0, 0) + mbqs.spin_coherent_state(N, np.pi, 0)
cat /= np.linalg.norm(cat)

Q = mbqs.husimi_q(cat, theta, phi)   # shape (100, 200), Q >= 0
W = mbqs.wigner(cat, theta, phi)     # negative near the equator

# e.g. plt.pcolormesh(phi, np.cos(theta), W)
```

## Tests

The tests compare the propagator against exact matrix exponentiation and
diagonalization:

```bash
pixi run test
```

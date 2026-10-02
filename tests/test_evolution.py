"""
Regression tests for the two-mode Hamiltonian and the Chebyshev propagator,
checked against exact diagonalization / matrix exponentiation.
"""

import numpy as np
import pytest
from scipy.linalg import expm, eigh

import many_body_quantum_simulator
from many_body_quantum_simulator import two_mode as mb

N = 40
NONLINEAR, COUPLING, DETUNING = 0.3, 0.5, 0.1


def fock_arrays(N):
    n_left = np.arange(N + 1)
    n_right = N - n_left
    al_ar = np.sqrt((n_right + 1) * n_left)
    ar_al = np.sqrt((n_left + 1) * n_right)
    return n_left, n_right, al_ar, ar_al


def dense_H(N, nonlinear, coupling, detuning):
    return mb.get_H_matrix(N, nonlinear, coupling, detuning).toarray()


def operator_matrix(op, N):
    basis = np.eye(N + 1, dtype=complex)
    return np.array([op(basis[:, k]) for k in range(N + 1)]).T


def random_state(N, seed=0):
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=N + 1) + 1j * rng.normal(size=N + 1)
    return psi / np.linalg.norm(psi)


def step(psi, t_step, nonlinear, coupling, detuning, imaginary_time=False,
         e_min=-1.5, e_max=1.5):
    n_left, n_right, al_ar, ar_al = fock_arrays(len(psi) - 1)
    return mb.evaluate_state(psi, t_step, e_min, e_max, len(psi) - 1,
                             n_left, n_right, al_ar, ar_al,
                             nonlinear, coupling, detuning,
                             imaginary_time=imaginary_time)


# --- Hamiltonian and spin operators -----------------------------------------

def test_hamiltonian_implementations_agree():
    psi = random_state(N)
    H = dense_H(N, NONLINEAR, COUPLING, DETUNING)
    n_left, n_right, al_ar, ar_al = fock_arrays(N)

    assert np.allclose(H, H.T)
    assert np.allclose(H @ psi, mb.apply_H_vectorized(psi, NONLINEAR,
                                                       COUPLING, DETUNING))
    assert np.allclose(H @ psi, mb.apply_H_static(psi, N, n_left, n_right,
                                                   al_ar, ar_al, NONLINEAR,
                                                   COUPLING, DETUNING))


def test_hamiltonian_matches_spin_operators():
    X = operator_matrix(mb.Sx, N)
    Z = operator_matrix(mb.Sz, N)
    H = dense_H(N, NONLINEAR, COUPLING, DETUNING)
    assert np.allclose(H, NONLINEAR * Z @ Z + COUPLING * X + DETUNING * Z)


def test_spin_algebra_is_right_handed():
    X, Y, Z = (operator_matrix(op, N) for op in (mb.Sx, mb.Sy, mb.Sz))
    assert np.allclose(X @ Y - Y @ X, 2j / N * Z)
    assert np.allclose(Y @ Z - Z @ Y, 2j / N * X)
    assert np.allclose(Z @ X - X @ Z, 2j / N * Y)


# --- Real-time evolution ----------------------------------------------------

@pytest.mark.parametrize("t_step", [0.1, 0.5, 5.0, 20.0, 40.0])
def test_real_time_step_matches_expm(t_step):
    psi = random_state(N)
    H = dense_H(N, NONLINEAR, COUPLING, DETUNING)
    exact = expm(-1j * H * t_step) @ psi
    assert np.allclose(step(psi, t_step, NONLINEAR, COUPLING, DETUNING),
                       exact, atol=1e-10)


def test_real_time_preserves_norm():
    psi = random_state(N)
    out = step(psi, 5.0, NONLINEAR, COUPLING, DETUNING)
    assert np.isclose(np.linalg.norm(out), 1.0)


def test_larmor_precession_direction():
    # H = w Sz rotates the spin counter-clockwise about z: +x -> +y.
    X, Y = operator_matrix(mb.Sx, N), operator_matrix(mb.Sy, N)
    _, v = eigh(X)
    psi = v[:, -1].astype(complex)  # spin-coherent state along +x
    t = 0.3 * N
    out = step(psi, t, 0.0, 0.0, 1.0)
    sx = (out.conj() @ X @ out).real
    sy = (out.conj() @ Y @ out).real
    assert sy > 0
    # Exact result for the normalized spin: <Sx> = cos(2t/N), <Sy> = sin(2t/N)
    assert np.isclose(sx, np.cos(2 * t / N), atol=1e-10)
    assert np.isclose(sy, np.sin(2 * t / N), atol=1e-10)


def test_evolve_state_time_dependent_protocol():
    nonlinear = lambda t: 0.3 + 0.01 * t
    coupling = lambda t: 0.5
    detuning = lambda t: 0.1 * np.sin(t)
    t_step, steps = 0.5, 40

    psi0 = random_state(N, seed=1)
    trace = mb.evolve_state(psi0, nonlinear, coupling, detuning,
                            t_step, steps, snapshots=steps)

    exact = psi0.copy()
    time = 0.0
    for i in range(steps):
        assert np.allclose(trace[i], exact, atol=1e-10)
        tm = time + t_step / 2
        H = dense_H(N, nonlinear(tm), coupling(tm), detuning(tm))
        exact = expm(-1j * H * t_step) @ exact
        time += t_step


# --- Imaginary-time evolution -----------------------------------------------

def test_ground_state_matches_eigh():
    H = dense_H(N, NONLINEAR, COUPLING, DETUNING)
    energies, vectors = eigh(H)
    psi0 = np.ones(N + 1, dtype=complex) / np.sqrt(N + 1)

    ground = mb.get_ground_state(psi0, NONLINEAR, COUPLING, DETUNING)

    assert np.isclose(mb.get_energy(ground, NONLINEAR, COUPLING,
                                    DETUNING).real, energies[0])
    assert np.isclose(abs(vectors[:, 0].conj() @ ground) ** 2, 1.0)


@pytest.mark.parametrize("imaginary_time", [False, True])
def test_asymmetric_spectral_window(imaginary_time):
    psi = random_state(N)
    H = dense_H(N, NONLINEAR, COUPLING, DETUNING)
    t_step = 3.0
    U = expm(-H * t_step) if imaginary_time else expm(-1j * H * t_step)
    exact = U @ psi
    exact /= np.linalg.norm(exact)
    out = step(psi, t_step, NONLINEAR, COUPLING, DETUNING,
               imaginary_time=imaginary_time, e_min=-0.8, e_max=1.2)
    assert np.allclose(out, exact, atol=1e-10)


def exact_imaginary_time(H, psi, tau):
    # Eigendecomposition reference, shifted by E0 so that nothing overflows.
    energies, vectors = eigh(H)
    out = vectors @ (np.exp(-(energies - energies[0]) * tau)
                     * (vectors.T @ psi))
    return out / np.linalg.norm(out)


@pytest.mark.parametrize("e_min, e_max", [(-0.7, 0.7), (-3.0, 3.0),
                                          (-10.0, 10.0)])
@pytest.mark.parametrize("t_step", [5.0, 50.0])
def test_imaginary_time_wide_window(e_min, e_max, t_step):
    # Large (e_max - e_min) * t_step loses precision to cancellation unless
    # the step is split into sub-steps.
    psi = random_state(N)
    H = dense_H(N, NONLINEAR, COUPLING, DETUNING)
    out = step(psi, t_step, NONLINEAR, COUPLING, DETUNING,
               imaginary_time=True, e_min=e_min, e_max=e_max)
    assert np.allclose(out, exact_imaginary_time(H, psi, t_step), atol=1e-10)


def test_ground_state_wide_window():
    H = dense_H(N, NONLINEAR, COUPLING, DETUNING)
    energies, vectors = eigh(H)
    psi0 = random_state(N)

    ground = mb.get_ground_state(psi0, NONLINEAR, COUPLING, DETUNING,
                                 steps=200, e_min=-3.0, e_max=3.0)

    assert np.isclose(abs(vectors[:, 0].conj() @ ground) ** 2, 1.0)
    assert np.allclose(H @ ground, energies[0] * ground, atol=1e-10)


# --- Sparse Hamiltonian and spectral window ---------------------------------

def test_get_H_matrix_is_sparse_tridiagonal():
    from scipy.sparse import issparse
    H = mb.get_H_matrix(N, NONLINEAR, COUPLING, DETUNING)
    assert issparse(H)
    assert H.shape == (N + 1, N + 1)
    assert H.nnz <= 3 * N + 1
    psi = random_state(N)
    assert np.allclose(H @ psi, mb.apply_H_vectorized(psi, NONLINEAR,
                                                       COUPLING, DETUNING))


PARAMS = [(0.3, 0.5, 0.1), (1.0, 1.5, 0.3), (2.0, 2.0, 0.5), (4.0, 3.0, 1.0),
          (-2.0, 0.7, -0.4), (0.0, -1.0, 0.0)]


@pytest.mark.parametrize("nonlinear, coupling, detuning", PARAMS)
def test_spectral_bounds_contain_spectrum(nonlinear, coupling, detuning):
    energies = eigh(dense_H(N, nonlinear, coupling, detuning),
                    eigvals_only=True)
    lo, hi = mb.spectral_bounds(N, nonlinear, coupling, detuning)
    assert lo <= energies[0] and energies[-1] <= hi

    lo, hi = mb.spectral_bounds(N, nonlinear, coupling, detuning, exact=True)
    assert lo <= energies[0] and energies[-1] <= hi
    assert np.isclose(lo, energies[0], atol=1e-5)
    assert np.isclose(hi, energies[-1], atol=1e-5)


@pytest.mark.parametrize("nonlinear, coupling, detuning", PARAMS[1:4])
@pytest.mark.parametrize("t_step", [0.1, 1.0, 10.0])
def test_automatic_window_out_of_default_range(nonlinear, coupling, detuning,
                                               t_step):
    # These spectra exceed the old default window [-1.5, 1.5].
    psi = random_state(N)
    H = dense_H(N, nonlinear, coupling, detuning)
    out = step(psi, t_step, nonlinear, coupling, detuning,
               e_min=None, e_max=None)
    assert np.allclose(out, expm(-1j * H * t_step) @ psi, atol=1e-10)


def test_narrow_window_warns():
    psi = random_state(N)
    with pytest.warns(RuntimeWarning, match="Spectral window"):
        step(psi, 1.0, 4.0, 3.0, 1.0, e_min=-1.5, e_max=1.5)


def test_tight_exact_window_does_not_warn():
    import warnings
    psi = random_state(N)
    params = (2.0, 2.0, 0.5)
    e_min, e_max = mb.spectral_bounds(N, *params, exact=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = step(psi, 1.0, *params, e_min=e_min, e_max=e_max)
    assert np.allclose(out, expm(-1j * dense_H(N, *params)) @ psi,
                       atol=1e-10)


def test_large_N_without_dense_matrix():
    big_N = 100_000
    params = (1.0, 1.5, 0.3)
    lo_exact, hi_exact = mb.spectral_bounds(big_N, *params, exact=True)
    lo, hi = mb.spectral_bounds(big_N, *params)
    assert lo <= lo_exact and hi_exact <= hi

    psi = random_state(big_N)
    out = step(psi, 1.0, *params, e_min=None, e_max=None)
    assert np.isclose(np.linalg.norm(out), 1.0)
    H = mb.get_H_matrix(big_N, *params)
    energy = (psi.conj() @ (H @ psi)).real
    assert np.isclose((out.conj() @ (H @ out)).real, energy)


# --- Package interface ------------------------------------------------------

def test_public_api_is_exported():
    for name in many_body_quantum_simulator.__all__:
        assert getattr(many_body_quantum_simulator, name) is getattr(mb, name)
    assert many_body_quantum_simulator.__version__ == "0.1.0"

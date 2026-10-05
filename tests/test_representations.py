"""
Tests for the Bloch-sphere phase-space representations: conventions against
the spin operators of the simulator, normalization, moments and identities
that only the exact Husimi Q and Wigner functions satisfy.
"""

import numpy as np
import pytest

import many_body_quantum_simulator as mbqs
from many_body_quantum_simulator.representations import (
    spin_coherent_state, husimi_q, wigner)


def sphere_quadrature(degree):
    # Gauss-Legendre in cos(theta) times a uniform phi grid: exact for
    # band-limited functions up to the given spherical-harmonic degree.
    x, weights = np.polynomial.legendre.leggauss(degree // 2 + 2)
    n_phi = degree + 2
    theta = np.arccos(x)
    phi = np.arange(n_phi) * 2 * np.pi / n_phi

    def integrate(values):
        return (weights[:, None] * values).sum() * 2 * np.pi / n_phi

    unit = np.stack([np.sin(theta)[:, None] * np.cos(phi),
                     np.sin(theta)[:, None] * np.sin(phi),
                     np.cos(theta)[:, None] * np.ones_like(phi)])
    return theta, phi, integrate, unit


def spin_expectations(state):
    return np.array([(state.conj() @ op(state)).real
                     for op in (mbqs.Sx, mbqs.Sy, mbqs.Sz)])


def random_state(N, seed=0):
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=N + 1) + 1j * rng.normal(size=N + 1)
    return psi / np.linalg.norm(psi)


@pytest.mark.parametrize("theta, phi", [(0.0, 0.0), (1.1, 2.3), (np.pi, 0.7),
                                        (np.pi / 2, -1.0)])
def test_spin_coherent_state_direction(theta, phi):
    state = spin_coherent_state(20, theta, phi)
    assert np.isclose(np.linalg.norm(state), 1.0)
    direction = [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                 np.cos(theta)]
    assert np.allclose(spin_expectations(state), direction, atol=1e-12)


def test_husimi_q_peaks_at_coherent_state():
    theta0, phi0 = 1.1, 2.3
    theta = np.linspace(0, np.pi, 221)
    phi = np.linspace(-np.pi, np.pi, 441)
    Q = husimi_q(spin_coherent_state(50, theta0, phi0), theta, phi)
    row, col = np.unravel_index(Q.argmax(), Q.shape)
    assert abs(theta[row] - theta0) < 0.02
    assert abs(phi[col] - phi0) < 0.02


@pytest.mark.parametrize("N", [0, 1, 7, 20])
def test_normalization(N):
    theta, phi, integrate, _ = sphere_quadrature(2 * N)
    state = random_state(N)
    assert np.isclose(integrate(husimi_q(state, theta, phi)), 1.0)
    assert np.isclose(integrate(wigner(state, theta, phi)), 1.0)


@pytest.mark.parametrize("N", [1, 6, 21])
def test_first_moments(N):
    # <J> = (j + 1) int n Q dOmega = sqrt(j (j + 1)) int n W dOmega
    theta, phi, integrate, unit = sphere_quadrature(2 * N + 1)
    state = random_state(N, seed=N)
    j = N / 2
    spin = j * spin_expectations(state)
    Q = husimi_q(state, theta, phi)
    W = wigner(state, theta, phi)
    assert np.allclose([(j + 1) * integrate(Q * n) for n in unit], spin)
    assert np.allclose([np.sqrt(j * (j + 1)) * integrate(W * n)
                        for n in unit], spin)


def test_wigner_spin_half():
    # Spin 1/2: W(n) = (1 + sqrt(3) n . r) / (4 pi), r the Bloch vector
    theta, phi, _, unit = sphere_quadrature(8)
    state = np.array([0.6, 0.8j])
    expected = (1 + np.sqrt(3) * np.tensordot(spin_expectations(state), unit,
                                              axes=1)) / (4 * np.pi)
    assert np.allclose(wigner(state, theta, phi), expected, atol=1e-14)


@pytest.mark.parametrize("N", [2, 9, 30])
def test_wigner_overlap_formula(N):
    # |<a|b>|^2 = 4 pi / (2j + 1) int W_a W_b dOmega, specific to the Wigner
    # function among the s-parametrized representations
    theta, phi, integrate, _ = sphere_quadrature(4 * N)
    a, b = random_state(N, seed=1), random_state(N, seed=2)
    overlap = 4 * np.pi / (N + 1) * integrate(wigner(a, theta, phi)
                                              * wigner(b, theta, phi))
    assert np.isclose(overlap, abs(np.vdot(a, b))**2, atol=1e-12)
    assert np.isclose(4 * np.pi / (N + 1) * integrate(wigner(a, theta, phi)**2),
                      1.0)


def test_cat_state_negativity():
    N = 20
    cat = spin_coherent_state(N, 0, 0) + spin_coherent_state(N, np.pi, 0)
    cat /= np.linalg.norm(cat)
    theta = np.linspace(0, np.pi, 61)
    phi = np.linspace(-np.pi, np.pi, 121)
    assert wigner(cat, theta, phi).min() < -0.1
    assert husimi_q(cat, theta, phi).min() >= 0


def test_large_N_is_finite_and_normalized():
    N = 1000
    state = spin_coherent_state(N, 1.0, 0.5)
    theta, phi, integrate, _ = sphere_quadrature(2 * N)
    Q = husimi_q(state, theta, phi)
    assert np.isfinite(Q).all()
    assert np.isclose(integrate(Q), 1.0)
    W = wigner(state, np.linspace(0, np.pi, 20), np.linspace(0, 1, 10))
    assert np.isfinite(W).all()

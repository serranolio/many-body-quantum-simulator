"""
Phase-space representations of two-mode states on the Bloch sphere.

A state of N bosons in two modes is a spin j = N/2 state, with
m = (n_right - n_left) / 2 and the Fock index i = n_left, so that
Sz = 2 Jz / N. A point (theta, phi) on the sphere corresponds to the
mean-field direction <S> = (sin theta cos phi, sin theta sin phi, cos theta);
theta = 0 means all particles in the right mode.

All representations are normalized to integrate to 1 over the sphere,
with the measure dOmega = sin(theta) dtheta dphi.
author: Federico Serrano
Washington State University
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.special import gammaln, xlogy


def _coherent_magnitudes(N, theta):
    # |<n_left = i | theta, phi>| for each theta (rows) and i (columns):
    # sqrt(binom(N, i)) cos(theta/2)^(N - i) sin(theta/2)^i, in logs so that
    # large N neither overflows nor underflows prematurely.
    i = np.arange(N + 1)
    log_binom = gammaln(N + 1) - gammaln(i + 1) - gammaln(N - i + 1)
    half = np.asarray(theta, dtype=float)[:, None] / 2
    log_mag = (0.5 * log_binom
               + xlogy(N - i, np.abs(np.cos(half)))
               + xlogy(i, np.abs(np.sin(half))))
    return np.exp(log_mag)


def spin_coherent_state(N, theta, phi):
    """
    Spin-coherent state of N bosons pointing along (theta, phi).

    Parameters
    ----------
    N : int
        Number of bosons.
    theta, phi : float
        Polar and azimuthal angles, with theta in [0, pi].

    Returns
    -------
    ndarray, shape (N + 1,)
        Normalized state in the Fock basis (index n_left), with
        <Sx, Sy, Sz> = (sin theta cos phi, sin theta sin phi, cos theta).
    """
    magnitudes = _coherent_magnitudes(N, [theta])[0]
    return magnitudes * np.exp(1j * np.arange(N + 1) * phi)


def husimi_q(state, theta, phi):
    """
    Husimi Q function on the Bloch sphere,
    Q(theta, phi) = (N + 1) / (4 pi) |<theta, phi | psi>|^2.

    Parameters
    ----------
    state : ndarray, shape (N + 1,)
        Pure state in the Fock basis.
    theta, phi : array_like, 1-D
        Grid of polar and azimuthal angles.

    Returns
    -------
    ndarray, shape (len(theta), len(phi))
        Non-negative Q function, normalized to 1 over the sphere.
    """
    state = np.asarray(state)
    N = len(state) - 1
    phases = np.exp(-1j * np.outer(np.arange(N + 1), np.asarray(phi)))
    overlaps = _coherent_magnitudes(N, theta) @ (state[:, None] * phases)
    return (N + 1) / (4 * np.pi) * np.abs(overlaps)**2


def _wigner_weights(N):
    # Wigner kernel at the north pole, diagonal in m:
    #   w(m) = sqrt(2j + 1) / (4 pi) sum_k sqrt(2k + 1) T_k0(m),
    # where T_k0(m) are the diagonal multipole operators. Up to normalization
    # they are the discrete Chebyshev (Gram) polynomials in m, orthonormal on
    # m = -j..j; they are obtained stably as eigenvectors of their Jacobi
    # matrix, whose eigenvalues are the nodes m.
    if N == 0:
        return np.array([1 / (4 * np.pi)])
    k = np.arange(1, N + 1)
    beta = k * np.sqrt(((N + 1)**2 - k**2) / (4.0 * (4 * k**2 - 1)))
    nodes, vectors = eigh_tridiagonal(np.zeros(N + 1), beta)
    # Orthonormal polynomials with positive leading coefficient: p_0 > 0.
    multipoles = vectors * np.sign(vectors[0])
    w = (np.sqrt(N + 1) / (4 * np.pi)
         * (np.sqrt(2 * np.arange(N + 1) + 1) @ multipoles))
    # nodes run over m = -j..j in ascending order, while the Fock index
    # i = n_left = j - m runs in descending m.
    assert np.allclose(nodes, np.arange(N + 1) - N / 2, atol=1e-8)
    return w[::-1]


def wigner(state, theta, phi):
    """
    Spin (Agarwal) Wigner function on the Bloch sphere.

    Computed as W(n) = sum_m w_m |<j, m| R(n)^dagger |psi>|^2, where w_m is
    the Wigner kernel at the north pole and R(n) rotates the north pole to
    n = (theta, phi). The cost is O(N^2) per grid point plus one O(N^2)
    diagonalization, and memory is O(N^2).

    Parameters
    ----------
    state : ndarray, shape (N + 1,)
        Pure state in the Fock basis.
    theta, phi : array_like, 1-D
        Grid of polar and azimuthal angles.

    Returns
    -------
    ndarray, shape (len(theta), len(phi))
        Real Wigner function, normalized to 1 over the sphere; it can be
        negative.
    """
    state = np.asarray(state, dtype=complex)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    N = len(state) - 1
    weights = _wigner_weights(N)

    # R(theta, phi) = exp(-i phi Jz) exp(-i theta Jy). With Jy = D Jx D^dagger,
    # D = diag(i^n_left), and Jx = V diag(lam) V^T (real tridiagonal),
    #   R^dagger psi = D V exp(i theta lam) V^T D^dagger exp(i phi Jz) psi.
    # The leftmost D does not change |.|^2 and is dropped.
    n_left = np.arange(N + 1)
    m = N / 2 - n_left
    off_diagonal = np.sqrt((n_left[1:]) * (N - n_left[1:] + 1)) / 2
    lam, V = eigh_tridiagonal(np.zeros(N + 1), off_diagonal)

    d_conj = np.exp(-1j * np.pi / 2 * n_left)
    rotated_z = (d_conj * state)[:, None] * np.exp(1j * np.outer(m, phi))
    X = V.T @ rotated_z

    W = np.empty((len(theta), len(phi)))
    for row, angle in enumerate(theta):
        amplitudes = V @ (np.exp(1j * angle * lam)[:, None] * X)
        W[row] = weights @ np.abs(amplitudes)**2
    return W

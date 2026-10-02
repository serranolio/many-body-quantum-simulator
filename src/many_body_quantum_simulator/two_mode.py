"""
Description: Quantum many-body representation of two-mode bosonic systems
             with conserved particle number (e.g. two-component BECs,
             bosonic Josephson junctions, Lipkin-Meshkov-Glick model).
author: Federico Serrano
Washington State University
"""

import warnings

import numpy as np
from scipy.linalg import eigvalsh_tridiagonal
from scipy.sparse import diags
from scipy.special import jv, iv
from numba import njit

@njit(fastmath=True)
def Sx_kernel(state, al_ar, ar_al, N):
    out = np.empty_like(state)
    n = len(state)
    for i in range(n):
        val = 0.0j
        if i > 0:
            val += al_ar[i] * state[i-1]
        if i < n-1:
            val += ar_al[i] * state[i+1]
        out[i] = val / N
    return out 

@njit(fastmath=True)
def Sy_kernel(state, al_ar, ar_al, N):
    out = np.empty_like(state)
    n = len(state)
    for i in range(n):
        val = 0.0j
        if i > 0:
            val -= al_ar[i] * state[i-1]
        if i < n-1:
            val += ar_al[i] * state[i+1]
        out[i] = val / (1j * N)
    return out

@njit(fastmath=True)
def Sz_kernel(state, n_left, n_right, N):
    out = np.empty_like(state)
    for i in range(len(state)):
        out[i] = (n_right[i] - n_left[i]) * state[i] / N
    return out 

def Sx(state):
    N = len(state) - 1
    n_left = np.arange(N + 1)
    n_right = N - n_left
    al_ar = np.sqrt((n_right + 1) * n_left)
    ar_al = np.sqrt((n_left + 1) * n_right)

    return Sx_kernel(state, al_ar, ar_al, N)

def Sy(state):
    N = len(state) - 1
    n_left = np.arange(N + 1)
    n_right = N - n_left
    al_ar = np.sqrt((n_right + 1) * n_left)
    ar_al = np.sqrt((n_left + 1) * n_right)

    return Sy_kernel(state, al_ar, ar_al, N)

def Sz(state):
    N = len(state) - 1
    n_left = np.arange(N + 1)
    n_right = N - n_left

    return Sz_kernel(state, n_left, n_right, N)

def apply_H_vectorized(state,
                       nonlinear_t,
                       coupling_t,
                       detuning_t):
    N = len(state) - 1                                                          
    n_left = np.arange(N + 1)                                                   
    n_right = N - n_left                                                        
    al_ar = np.sqrt((n_right + 1) * n_left)                                     
    ar_al = np.sqrt((n_left + 1) * n_right)
    #output = np.zeros_like(state)
    state_p1 = np.roll(state, -1)
    state_p1[-1] = 0.0
    state_m1 = np.roll(state, 1)
    state_m1[0] = 0.0
    # off-diagonal terms
    output = (nonlinear_t * (n_right - n_left)**2 / N**2 * state
              + coupling_t * (al_ar * state_m1 + ar_al * state_p1) / N
              + detuning_t * (n_right - n_left) / N * state)
    return output

def get_H_tridiagonal(N,
                      nonlinear,
                      coupling,
                      detuning):
    """
    Diagonal and off-diagonal (symmetric) entries of H in the Fock basis.
    """
    n_left = np.arange(N + 1)
    n_right = N - n_left
    sz = (n_right - n_left) / N
    diagonal = nonlinear * sz**2 + detuning * sz
    off_diagonal = coupling * np.sqrt((n_left[:-1] + 1) * n_right[:-1]) / N
    return diagonal, off_diagonal

def get_H_matrix(N,
                 nonlinear,
                 coupling,
                 detuning):
    """
    Sparse (CSR) Hamiltonian in the Fock basis. Use .toarray() for a dense
    matrix, e.g. for np.linalg.eigh at small N.
    """
    diagonal, off_diagonal = get_H_tridiagonal(N, nonlinear, coupling,
                                               detuning)
    return diags([off_diagonal, diagonal, off_diagonal], [-1, 0, 1],
                 format='csr')

def spectral_bounds(N,
                    nonlinear,
                    coupling,
                    detuning,
                    exact=False,
                    margin=1e-6):
    """
    Interval [e_min, e_max] containing the spectrum of H.

    By default uses the analytic bound from the normalized spin operators
    having eigenvalues in [-1, 1]: O(1) cost, valid for any N, but loose.
    With exact=True, computes the extreme eigenvalues of the tridiagonal H
    (O(N) memory), padded by margin times the spectral width.
    """
    if exact:
        diagonal, off_diagonal = get_H_tridiagonal(N, nonlinear, coupling,
                                                   detuning)
        e_min = eigvalsh_tridiagonal(diagonal, off_diagonal, select='i',
                                     select_range=(0, 0))[0]
        e_max = eigvalsh_tridiagonal(diagonal, off_diagonal, select='i',
                                     select_range=(N, N))[0]
        pad = margin * (e_max - e_min) + 1e-12
        return e_min - pad, e_max + pad

    spread = abs(coupling) + abs(detuning)
    e_min = min(0.0, nonlinear) - spread
    e_max = max(0.0, nonlinear) + spread
    if e_max - e_min < 1e-12:
        # H = 0: any non-degenerate window works.
        return -1.0, 1.0
    # The bound can be attained (e.g. H = coupling * Sx); pad for round-off.
    pad = 1e-12 * (e_max - e_min)
    return e_min - pad, e_max + pad

def check_spectral_window(N,
                          nonlinear,
                          coupling,
                          detuning,
                          e_min,
                          e_max):
    """
    Warn if [e_min, e_max] provably misses part of the spectrum.

    By Cauchy interlacing, the eigenvalues of every 2x2 diagonal block of H
    lie within the spectrum of H, so a window that excludes any of them is
    too narrow. Never warns for a valid window, but can miss some bad ones.
    """
    diagonal, off_diagonal = get_H_tridiagonal(N, nonlinear, coupling,
                                               detuning)
    lower, upper = diagonal.min(), diagonal.max()
    if N > 0:
        mean = (diagonal[:-1] + diagonal[1:]) / 2
        radius = np.hypot((diagonal[:-1] - diagonal[1:]) / 2, off_diagonal)
        lower = min(lower, (mean - radius).min())
        upper = max(upper, (mean + radius).max())
    slack = 1e-12 * max(1.0, abs(lower), abs(upper))
    if lower < e_min - slack or upper > e_max + slack:
        warnings.warn(f"Spectral window [{e_min:.4g}, {e_max:.4g}] does not "
                      f"contain the spectrum of H (at least "
                      f"[{lower:.4g}, {upper:.4g}]); the Chebyshev "
                      f"propagator will be inaccurate. Pass "
                      f"e_min=e_max=None for an automatic window.",
                      RuntimeWarning, stacklevel=3)

def get_energy(state,
               nonlinear,
               coupling,
               detuning):

    return (state.conj() @
            apply_H_vectorized(state, nonlinear, coupling, detuning))


@njit(fastmath=True)
def apply_H_static(state,
                   N,
                   n_left,
                   n_right,
                   al_ar,
                   ar_al,
                   nonlinear_t,
                   coupling_t,
                   detuning_t):
    n = len(state)
    output = np.zeros_like(state)

    # Calculate output values
    for i in range(n):
        value = 0.0
        if i > 0:
            value += coupling_t * al_ar[i] * state[i-1]
        if i < n-1:
            value += coupling_t * ar_al[i] * state[i+1]
        value = value / N
        Sz_value = (n_right[i] - n_left[i]) / N
        value += detuning_t * Sz_value * state[i]
        value += nonlinear_t * Sz_value * Sz_value * state[i]
        output[i] = value
    return output

@njit(fastmath=True)
def chebyshev_evolver_kernel(u0,
                             u1,
                             bessel_coeffs,
                             e_min,
                             e_max,
                             phase_shift,
                             imaginary_time,
                             N,
                             n_left,
                             n_right,
                             al_ar,
                             ar_al,
                             nonlinear_t,
                             coupling_t,
                             detuning_t):
    """
    Fast Chebyshev expansion for time evolution
    """

    n_terms = len(bessel_coeffs)
    state_t = bessel_coeffs[0] * u0 + 2 * bessel_coeffs[1] * u1
    
    u_nm1, u_n = u0.copy(), u1.copy()

    for n in range(2, n_terms):
        # Apply normalized Hamiltonian
        Hu_n = apply_H_static(u_n,
                              N,
                              n_left,
                              n_right,
                              al_ar,
                              ar_al,
                              nonlinear_t,
                              coupling_t,
                              detuning_t)
        Hu_n = (2.0*Hu_n - (e_max + e_min) * u_n) / (e_max - e_min)

        # Recursion relation
        if imaginary_time:
            u_np1 = 2.0 * Hu_n - u_nm1
        else:
            u_np1 = -2.0j * Hu_n + u_nm1

        state_t += 2.0 * bessel_coeffs[n] * u_np1

        u_nm1, u_n = u_n, u_np1

    # Apply phase shift
    state_t *= phase_shift

    # Normalize
    norm = np.sqrt(state_t.conj() @ state_t)
    return state_t / norm

#@njit(fastmath=True)
def evaluate_state(state_t0,
                   t_step,
                   e_min,
                   e_max,
                   N,
                   n_left,
                   n_right,
                   al_ar,
                   ar_al,
                   nonlinear_t,
                   coupling_t,
                   detuning_t,
                   imaginary_time=False,
                   tol=1e-14,
                   max_terms=10_000,
                   max_alpha_imag=10.0):
    # e_min/e_max = None: use the analytic bound on the spectrum of H.
    # An explicit window is used as given, but checked for being too narrow.
    if e_min is None or e_max is None:
        auto_min, auto_max = spectral_bounds(N, nonlinear_t, coupling_t,
                                             detuning_t)
        e_min = auto_min if e_min is None else e_min
        e_max = auto_max if e_max is None else e_max
    else:
        check_spectral_window(N, nonlinear_t, coupling_t, detuning_t,
                              e_min, e_max)

    # In imaginary time the I_n(z) ~ e^|z| terms cancel down to a much
    # smaller result, losing up to ~e^|z| in relative precision. Split
    # large steps into sub-steps with |z| <= max_alpha_imag.
    z = abs((e_max - e_min) * t_step / 2.0)
    n_sub = (int(np.ceil(z / max_alpha_imag))
             if imaginary_time and z > max_alpha_imag else 1)
    state_t = state_t0
    for _ in range(n_sub):
        state_t = chebyshev_step(state_t,
                                 t_step / n_sub,
                                 e_min,
                                 e_max,
                                 N,
                                 n_left,
                                 n_right,
                                 al_ar,
                                 ar_al,
                                 nonlinear_t,
                                 coupling_t,
                                 detuning_t,
                                 imaginary_time,
                                 tol,
                                 max_terms)
    return state_t

def chebyshev_step(state_t0,
                   t_step,
                   e_min,
                   e_max,
                   N,
                   n_left,
                   n_right,
                   al_ar,
                   ar_al,
                   nonlinear_t,
                   coupling_t,
                   detuning_t,
                   imaginary_time,
                   tol,
                   max_terms):
    # exp(-i z x) = J_0(z) + 2 sum_n (-i)^n J_n(z) T_n(x)  (real time)
    # exp(-z x)   = I_0(z) + 2 sum_n (-1)^n I_n(z) T_n(x)  (imaginary time)
    # with z = (e_max - e_min) * t_step / 2; the (-1)^n is absorbed by
    # evaluating I_n at -z.
    alpha = (e_max - e_min) * t_step / 2.0
    if imaginary_time:
        alpha = -alpha
        phase_shift = np.exp(-(e_max + e_min) * t_step / 2.0)
    else:
        phase_shift = np.exp(-1j * (e_max + e_min) * t_step / 2.0)

    # Choose Bessel function
    bessel_v = iv if imaginary_time else jv
    factor = 1.0 if imaginary_time else -1j

    # Initial recursion states
    u0 = state_t0.copy()
    Hu_0 = apply_H_static(u0,
                          N,
                          n_left,
                          n_right,
                          al_ar,
                          ar_al,
                          nonlinear_t,
                          coupling_t,
                          detuning_t)
    u1 = factor * (2.0*Hu_0 - (e_max + e_min) * u0) / (e_max - e_min)

    # Precompute Bessel coefficients: add terms until they fall below
    # tol relative to the largest one. Beyond n ~ |alpha| the coefficients
    # decay faster than exponentially, so check only past that point.
    n = np.arange(int(abs(alpha)) + 2)
    bessel_coeffs = bessel_v(n, alpha)
    scale = np.max(np.abs(bessel_coeffs))
    while abs(bessel_coeffs[-1]) > tol * scale:
        if len(bessel_coeffs) > max_terms:
            raise RuntimeError(f"Chebyshev series did not converge within "
                               f"{max_terms} terms (alpha={alpha:.3g})")
        bessel_coeffs = np.append(bessel_coeffs,
                                  bessel_v(len(bessel_coeffs), alpha))

    # Call Chebyshev kernel
    state_t = chebyshev_evolver_kernel(u0,
                                       u1,
                                       bessel_coeffs,
                                       e_min,
                                       e_max,
                                       phase_shift,
                                       imaginary_time,
                                       N,
                                       n_left,
                                       n_right,
                                       al_ar,
                                       ar_al,
                                       nonlinear_t,
                                       coupling_t,
                                       detuning_t)
    return state_t

#@njit(fastmath=True)
def evolve_state(state_t0,
                 nonlinear,
                 coupling,
                 detuning,
                 t_step,
                 steps,
                 e_min=None,
                 e_max=None,
                 imaginary_time=False,
                 snapshots=500,
                 tol=1e-14):
    
    N = len(state_t0) - 1
    n_left = np.arange(N + 1)
    n_right = N - np.arange(N + 1)
    al_ar = np.sqrt((n_right + 1) * n_left)
    ar_al = np.sqrt((n_left + 1) * n_right)

    # Initialize routine
    if snapshots >= steps:
        snapshots = steps
    time = 0.0
    state_t = state_t0.copy()
    state_time_trace = np.empty((snapshots, len(state_t0)),
                                dtype='complex')

    snap_idx = 0

    for i in range(steps):
        nonlinear_value = nonlinear(time + t_step/2)
        coupling_value = coupling(time + t_step/2)
        detuning_value = detuning(time + t_step/2)
        
        if i % (steps // snapshots) == 0 and snap_idx < snapshots:

            state_time_trace[snap_idx] = state_t
            snap_idx += 1
        
        state_t = evaluate_state(state_t,
                                 t_step,
                                 e_min,
                                 e_max,
                                 imaginary_time=imaginary_time,
                                 N=N,
                                 n_left=n_left,
                                 n_right=n_right,
                                 al_ar=al_ar,
                                 ar_al=ar_al,
                                 nonlinear_t=nonlinear_value,
                                 coupling_t=coupling_value,
                                 detuning_t=detuning_value,
                                 tol=tol)
        time += t_step

    return np.array(state_time_trace)

def get_ground_state(state_t0,
                     nonlinear,
                     coupling,
                     detuning,
                     t_step=50,
                     steps=1000,
                     e_min=None,
                     e_max=None,
                     tol=1e-14,
                     ):
    
    N = len(state_t0) - 1
    n_left = np.arange(N + 1)
    n_right = N - np.arange(N + 1)
    al_ar = np.sqrt((n_right + 1) * n_left)
    ar_al = np.sqrt((n_left + 1) * n_right)

    state_t = state_t0.copy()
    for i in range(steps):

        state_t = evaluate_state(state_t,
                                 t_step,
                                 e_min,
                                 e_max,
                                 imaginary_time=True,
                                 N=N,
                                 n_left=n_left,
                                 n_right=n_right,
                                 al_ar=al_ar,
                                 ar_al=ar_al,
                                 nonlinear_t=nonlinear,
                                 coupling_t=coupling,
                                 detuning_t=detuning,
                                 tol=tol)
    return state_t

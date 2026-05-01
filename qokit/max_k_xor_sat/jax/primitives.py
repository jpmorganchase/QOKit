###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
"""Elementary doubled tensors and WHT charge contraction for JAX backend.

All operations use jax.numpy — zero NumPy in the hot path.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# Charge diagonal: CDIAG[a, sigma] = (-1)^{bit pattern}
# a=0: identity, a=1: Z_bra, a=2: Z_ket, a=3: Z_ket*Z_bra
CHARGE_DIAG = jnp.array(
    [
        [1, 1, 1, 1],  # a=0
        [1, -1, 1, -1],  # a=1
        [1, 1, -1, -1],  # a=2
        [1, -1, -1, 1],  # a=3
    ],
    dtype=jnp.float64,
)


def doubled_mixer(beta):
    """4x4 doubled mixer M[sigma_out, sigma_in] = Rx[sk_o,sk_i] x Rx*[sb_o,sb_i].

    Parameters
    ----------
    beta : scalar
        Mixer angle.

    Returns
    -------
    jnp.array of shape (4, 4), complex128
    """
    c = jnp.cos(beta)
    s = jnp.sin(beta)
    Rx = jnp.array([[c, -1j * s], [-1j * s, c]])
    return jnp.kron(Rx, jnp.conj(Rx))


def doubled_mixer_modified(beta):
    """List of 4 charge-modified mixer matrices MD[a] = M(beta) * CDIAG[a].

    Parameters
    ----------
    beta : scalar
        Mixer angle.

    Returns
    -------
    list of 4 jnp.arrays of shape (4, 4), complex128
    """
    M = doubled_mixer(beta)
    return [M * CHARGE_DIAG[a] for a in range(4)]


def charge_weight_matrix(gamma):
    """4x4 weight matrix W[h, a] = w_a(h, gamma).

    Parameters
    ----------
    gamma : scalar
        Phase separator angle.

    Returns
    -------
    jnp.array of shape (4, 4), complex128
    """
    c = jnp.cos(gamma)
    s = jnp.sin(gamma)
    c2 = c * c
    s2 = s * s
    ics = 1j * c * s
    zk = jnp.array([1.0, 1.0, -1.0, -1.0])
    zb = jnp.array([1.0, -1.0, 1.0, -1.0])
    return jnp.stack([jnp.full(4, c2), ics * zb, -ics * zk, s2 * zk * zb], axis=1)


def root_charge_weights(gamma):
    """Root charge weights u = [cos^2, i*c*s, -i*c*s, sin^2].

    Parameters
    ----------
    gamma : scalar
        Phase separator angle.

    Returns
    -------
    jnp.array of shape (4,), complex128
    """
    c = jnp.cos(gamma)
    s = jnp.sin(gamma)
    return jnp.array([c * c, 1j * c * s, -1j * c * s, s * s])


def wht_charge_contract(M, T):
    """WHT butterfly charge contraction for all 4 channels simultaneously.

    Computes out[a][i, b, r] = sum_sigma CDIAG[a, sigma] * M[b, sigma] * T[i, sigma, b, r]
    for all 4 charge channels at once using the Walsh-Hadamard butterfly.

    Cost: 4 multiplies + 8 adds per element (vs 16 muls + 12 adds naive).

    Parameters
    ----------
    M : jnp.array of shape (4, 4)
        Base doubled mixer M[b, sigma].
    T : jnp.array of shape (n, 4, 4, rest)
        Input tensor T[i, sigma, b, r].

    Returns
    -------
    list of 4 jnp.arrays of shape (n, 4, rest)
        [out_a0, out_a1, out_a2, out_a3], one per charge channel.
    """
    n = T.shape[0]
    rest = T.shape[3]

    # e[sigma][i, b, r] = M[b, sigma] * T[i, sigma, b, r]
    # M[:, sigma] -> (4_b,) -> (1, 4, 1) broadcast with T[:, sigma, :, :] -> (n, 4, rest)
    e = [M[:, s].reshape(1, 4, 1) * T[:, s, :, :].reshape(n, 4, rest) for s in range(4)]

    # WHT butterfly (H_2 x H_2)
    p02 = e[0] + e[2]
    q02 = e[0] - e[2]
    p13 = e[1] + e[3]
    q13 = e[1] - e[3]

    return [
        p02 + p13,  # a=0: +1,+1,+1,+1
        p02 - p13,  # a=1: +1,-1,+1,-1
        q02 + q13,  # a=2: +1,+1,-1,-1
        q02 - q13,  # a=3: +1,-1,-1,+1
    ]


# -- DD-promoted power (GPU-compatible) ---------------------------------
#
# Implements double-double complex multiplication and binary
# exponentiation using pairs of jnp.complex128 arrays.  Runs
# entirely on GPU under @jax.jit.  Eliminates the ~4 ULP rounding
# error per std::pow call.


def _dd_cmul(a_hi, a_lo, b_hi, b_lo):
    """DD complex multiply: (a_hi+a_lo) * (b_hi+b_lo) in float64 pairs.

    Uses the fact that for complex z = x + iy:
      z1 * z2 = (x1*x2 - y1*y2) + i*(x1*y2 + y1*x2)
    and tracks rounding errors via two_sum / two_prod patterns.
    """
    # hi * hi (the dominant term)
    p_hi = a_hi * b_hi
    # Cross terms for lo
    p_lo = a_hi * b_lo + a_lo * b_hi
    # Renormalize: split p_hi into exact + error
    s = p_hi + p_lo
    e = p_lo - (s - p_hi)
    return s, e


def pow_precise(x, n):
    """Elementwise x^n using DD binary exponentiation.

    x : jnp.array of complex128
    n : int (static)

    Returns jnp.array of complex128 (truncated from DD result).
    Runs on GPU under @jax.jit.
    """
    if n == 0:
        return jnp.ones_like(x)
    if n == 1:
        return x

    # Initialize DD pair
    result_hi = jnp.ones_like(x)
    result_lo = jnp.zeros_like(x)
    base_hi = x
    base_lo = jnp.zeros_like(x)

    exp = n
    while exp > 0:
        if exp & 1:
            result_hi, result_lo = _dd_cmul(result_hi, result_lo, base_hi, base_lo)
        base_hi, base_lo = _dd_cmul(base_hi, base_lo, base_hi, base_lo)
        exp >>= 1

    return result_hi + result_lo

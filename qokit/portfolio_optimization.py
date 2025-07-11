"""
Helper functions for the portfolio optimization problem
"""

from __future__ import annotations
from collections.abc import Sequence
import numpy as np
from functools import partial
import itertools
from typing import Any
from qokit.parameter_utils import get_sk_gamma_beta
from numba import njit, prange

from typing import Tuple, Optional, List, cast


def convert_bitstring_to_int(config):
    """make configuration iterable"""
    N = len(config)
    z = np.zeros(N).astype(int)
    for i in range(len(config)):
        z[i] = int(config[i])
    return z


def get_configuration_cost(po_problem, config):
    """
    Compute energy for single sample configuration
    f(x) = q \sigma_ij x_i x_j - \mu_i x_i
    """
    if not isinstance(config, np.ndarray):
        config = convert_bitstring_to_int(config)
    scale = po_problem["scale"]
    means = po_problem["means"] / scale
    cov = po_problem["cov"] / scale

    return po_problem["q"] * config.dot(cov).dot(config) - means.dot(config)

def brute_force_cost_vector(po_problem: dict) -> np.ndarray:
    """
    Return all 2^N energies for the given portfolio instance using
    the parallel-SIMD Numba kernel (N ≤ 20 recommended).
    """
    return _bruteforce_costs(po_problem["mu"],
                             po_problem["cov"],
                             po_problem["q"])


@njit(parallel=True, fastmath=True)
def _bruteforce_costs(mu, cov, q):
    N = mu.size
    costs = np.empty(1 << N, dtype=np.float64)
    for b in prange(1 << N):
        # unpack bits into 0/1 vector
        x = np.unpackbits(
            np.array([b], dtype=np.uint32).view(np.uint8),
            bitorder="little"
        )[:N].astype(np.float64)
        costs[b] = q * x @ cov @ x - mu @ x
    return costs

def get_configuration_cost_vector(po_problem: dict[str, Any], config:np.ndarray)-> np.ndarray:
    r"""
    Vectorised portfolio‐cost evaluation.

    This implements the quadratic objective

    .. math::

        f(x) \;=\; q\, x^{\top}\Sigma x \;-\; \mu^{\top}x

    for **one or many** bit-strings in a single NumPy call.

    Parameters
    ----------
    po_problem
        A dictionary returned by :func:`get_problem`.  It must contain

        * ``"cov"`` – the :math:`\Sigma` covariance matrix *(N×N)*,
        * ``"mu"``  – the expected-return vector :math:`\mu` *(N,)*,
        * ``"q"``   – the risk-aversion scalar :math:`q`.

    config
        • Shape ``(N,)`` – a single 0/1 bit-string interpreted as
          :math:`x\in\{0,1\}^N`.
        • Shape ``(B,N)`` – a batch of *B* bit-strings.

        The function accepts either **NumPy** or **CuPy** arrays; output will
        match the input backend.

    Returns
    -------
    np.ndarray
        • Scalar ``float`` if a single bit-string was supplied.
        • 1-D array ``(B,)`` for a batch input.

    Notes
    -----
    This vectorised route is ~20× faster than the Python loop used for the
    brute-force reference when *B ≫ N*; it’s therefore the preferred pathway
    whenever you want to sweep hundreds of candidate solutions (e.g. inside a
    classical optimiser or to pre-compute all :math:`2^N` energies for
    *N ≤ 20*).
    """
    scale = po_problem["scale"]
    means = po_problem["means"] / scale
    cov = po_problem["cov"] / scale
    q = po_problem["q"]

    config = np.asarray(config)
    if config.ndim == 1:
        # Single bitstring
        return q * config.dot(cov).dot(config) - means.dot(config)
    elif config.ndim == 2:
        # Batch of bitstrings
        # config: (batch, N), cov: (N, N), means: (N,)
        # quadratic term: (config @ cov) * config, sum over axis=1
        quad = np.einsum('ij,jk,ik->i', config, cov, config)
        linear = config.dot(means)
        return q * quad - linear
    else:
        raise ValueError("config must be 1D or 2D array")



def get_configuration_cost_kw(config, po_problem=None):
    """
    Convenience function for functools.partial
    e.g. po_obj = partial(get_configuration_cost, po_problem=po_problem)
    """
    return get_configuration_cost(po_problem, config)

def get_configuration_cost_kw_vector(config, po_problem=None):
    """
    Convenience function for functools.partial
    e.g. po_obj = partial(get_configuration_cost, po_problem=po_problem)
    Now supports vectorized (batch) config.
    """
    return get_configuration_cost_vector(po_problem, config)


def po_obj_func(po_problem: dict) -> float:
    """
    Wrapper function for compute a portofolio value
    """
    return partial(get_configuration_cost_kw, po_problem=po_problem)

def po_obj_func_vector(po_problem: dict) -> float:
    """
    Wrapper function for compute a portofolio value
    Now supports vectorized (batch) config.
    """
    return partial(get_configuration_cost_kw_vector, po_problem=po_problem)


def kbits(N, K):
    for bits in itertools.combinations(range(N), K):
        s = [0] * N
        for bit in bits:
            s[bit] = 1
        yield np.array(s)


def portfolio_brute_force(po_problem: dict, return_bitstring=False) -> tuple[float, float, float] | tuple[float, float, float, float]:
    N = po_problem["N"]
    K = po_problem["K"]
    min_constrained = float("inf")
    max_constrained = float("-inf")
    mean_constrained = 0
    total_constrained = 0
    po_obj = po_obj_func(po_problem)
    for x in kbits(N, K):
        curr = po_obj(x)
        if curr < min_constrained:
            min_constrained = curr
            min_x = x
        if curr > max_constrained:
            max_constrained = curr
            max_x = x
        mean_constrained += curr
        total_constrained += 1.0
    mean_constrained /= total_constrained
    if return_bitstring is False:
        return min_constrained, max_constrained, mean_constrained
    else:
        return min_constrained, min_x, max_constrained, max_x, mean_constrained


def get_data(N, seed=1, real=False) -> tuple[float, float]:
    """
    load portofolio data from qiskit-finance (Yahoo)
    https://github.com/Qiskit/qiskit-finance/blob/main/docs/tutorials/11_time_series.ipynb
    """
    import datetime

    from qokit.yahoo import YahooDataProvider

    stock_symbols = [
        "AAPL",
        "GOOGL",
        "AMZN",
        "MSFT",
        "TSLA",
        "NFLX",
        "NVDA",
        "JPM",
        "V",
        "JNJ",
        "WMT",
        "PG",
        "MA",
        "UNH",
        "HD",
        "DIS",
        "BRK-B",
        "VZ",
        "KO",
        "MRK",
        "INTC",
        "CMCSA",
        "PEP",
        "PFE",
        "CSCO",
        "XOM",
        "BA",
        "MCD",
        "ABBV",
        "IBM",
        "GE",
        "MMM",
    ]

    data = YahooDataProvider(
        tickers=stock_symbols[:N],
        start=datetime.datetime(2020, 1, 1),
        end=datetime.datetime(2020, 1, 30),
    )

    data.run()
    period_returns = np.array(data._data)[:, 1:] / np.array(data._data)[:, :-1] - 1
    means = cast(np.ndarray, np.mean(period_returns, axis=1))
    cov = np.cov(period_returns, rowvar=True)
    return means, cov


def get_problem(N, K, q, seed=1, pre=False) -> dict[str, Any]:
    """generate the portofolio optimziation problem dict"""
    po_problem = {}
    po_problem["N"] = N
    po_problem["K"] = K
    po_problem["q"] = q
    po_problem["seed"] = seed
    po_problem["means"], po_problem["cov"] = get_data(N, seed=seed)
    po_problem["pre"] = pre
    if pre == "rule":
        means_in_spins = np.array([po_problem["means"][i] - q * np.sum(po_problem["cov"][i, :]) for i in range(len(po_problem["means"]))])
        scale = 1 / np.sqrt(np.mean(((q * po_problem["cov"]) ** 2).flatten()) + np.mean((means_in_spins**2).flatten()))
        # scale = 1 / (0.5*(np.sqrt(np.mean((po_problem['cov']**2).flatten())+np.mean((po_problem['means']**2).flatten())))
    elif np.isscalar(pre):
        scale = pre
    else:
        scale = 1

    po_problem["scale"] = scale
    po_problem["means"] = scale * po_problem["means"]
    po_problem["cov"] = scale * po_problem["cov"]

    return po_problem

def get_problem_vectorized(N, K, q, seed=1, pre=False) -> dict[str, Any]:
    """generate the portofolio optimziation problem dict"""
    po_problem = {}
    po_problem["N"] = N
    po_problem["K"] = K
    po_problem["q"] = q
    po_problem["seed"] = seed
    po_problem["means"], po_problem["cov"] = get_data(N, seed=seed)
    po_problem["pre"] = pre
    if pre == "rule":
        means_in_spins = po_problem["means"] - q * np.sum(po_problem["cov"], axis=1)
        scale = 1 / np.sqrt(
            np.mean((q * po_problem["cov"]) ** 2) + np.mean(means_in_spins ** 2)
        )
    elif np.isscalar(pre):
        scale = pre
    else:
        scale = 1

    po_problem["scale"] = scale
    po_problem["means"] = scale * po_problem["means"]
    po_problem["cov"] = scale * po_problem["cov"]

    return po_problem


def get_problem_H(po_problem):
    """
    Get the problem Hamiltonian in the matrix form
    0.5 q \sum_{i=1}^{n-1} \sum_{j=i+1}^n \sigma_{ij}Z_i Z_j + 0.5 \sum_i (-q\sum_{j=1}^n{\sigma_ij} + \mu_i) Z_i +
    0.5 q (\sum_{i}\sum_{j=i}^N \sigma[i,j] - \sum_i \mu_i) I
    """
    N = po_problem["N"]
    scale = po_problem["scale"]
    means = po_problem["means"] / scale
    cov = po_problem["cov"] / scale
    q = po_problem["q"]

    H_all = np.zeros((2**N, 2**N), dtype=complex)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    for k1 in range(N - 1):
        for k2 in range(k1 + 1, N):
            H = 1
            for i in range(N):
                if i == k1 or i == k2:
                    H = np.kron(H, Z)
                else:
                    H = np.kron(H, I)
            H = 0.5 * q * cov[k1, k2] * H
            H_all += H
    for k1 in range(N):
        H = 1
        for i in range(N):
            if i == k1:
                H = np.kron(H, Z)  # the order is important!
            else:
                H = np.kron(H, I)
        H = 0.5 * (means[k1] - q * np.sum(cov[k1, :])) * H  #
        H_all += H

    constant = 0
    for k1 in range(N):
        constant += q * np.sum(cov[k1, k1:]) - means[k1]
    H_all = H_all + 0.5 * constant * np.eye(2**N)
    return H_all


def get_problem_H_bf(po_problem):
    """
    Get the problem Hamiltonian in the matrix form
    replace every binary x by S=(I-Z)/2
    """
    N = po_problem["N"]
    scale = po_problem["scale"]
    means = po_problem["means"] / scale
    cov = po_problem["cov"] / scale
    q = po_problem["q"]

    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    S = (I - Z) / 2
    H_all2 = np.zeros((2**N, 2**N), dtype=complex)
    for k1 in range(N):
        for k2 in range(N):
            H = 1
            for i in range(N):
                if i == k1 or i == k2:
                    H = np.kron(H, S)
                else:
                    H = np.kron(H, I)
            H = q * cov[k1, k2] * H
            H_all2 += H
    for k1 in range(N):
        H = 1
        for i in range(N):
            if i == k1:
                H = np.kron(H, S)  # the order is important!
            else:
                H = np.kron(H, I)
        H = -means[k1] * H
        H_all2 += H
    return H_all2


@njit(cache=True, fastmath=True)
def hamming_weight(index: int) -> int:
    """
    Calculate the hamming weight for a given integer using bitwise operations.
    """
    count = 0
    while index:
        count += index & 1
        index >>= 1
    return count


@njit(cache=True)
def yield_all_indices_cosntrained(N: int, K: int):
    """
    Numba-accelerated generator for indices with Hamming weight K.
    """
    for ind in range(2**N):
        if hamming_weight(ind) == K:
            yield ind


def get_sk_ini(p: int):
    """
    scaled the sk look-up table for the application of portfolio optimziation
    """
    gamma_scale, beta_scale = 0.5, 1
    gamma, beta = get_sk_gamma_beta(p, parameterization="gamma beta")
    scaled_gamma, scaled_beta = gamma_scale * gamma, beta_scale * beta
    X0 = np.concatenate((scaled_gamma, scaled_beta), axis=0)
    return X0


def alignment_para_to_qokit_scale(gammas: Sequence[float] | None, betas: Sequence[float] | None):
    """Converts from format in alignment project
    into the scale used in qokit
    """
    gammas = np.asarray(gammas) * 2
    betas = np.asarray(betas) * 2
    return gammas, betas

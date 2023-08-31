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


def get_configuration_cost_kw(config, po_problem=None):
    """
    Convenience function for functools.partial
    e.g. po_obj = partial(get_configuration_cost, po_problem=po_problem)
    """
    return get_configuration_cost(po_problem, config)


def po_obj_func(po_problem: dict) -> float:
    """
    Wrapper function for compute a portofolio value
    """
    return partial(get_configuration_cost_kw, po_problem=po_problem)


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
    from qiskit_finance.data_providers import RandomDataProvider, YahooDataProvider

    tickers = []
    for i in range(N):
        tickers.append("t" + str(i))
    if real is False:
        data = RandomDataProvider(
            tickers=tickers,
            start=datetime.datetime(2016, 1, 1),
            end=datetime.datetime(2016, 1, 30),
            seed=seed,
        )
    else:
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
            # end=datetime.datetime(2021, 1, 1),
        )

    data.run()
    # use get_period_return_mean_vector & get_period_return_covariance_matrix to get return!
    # https://github.com/Qiskit/qiskit-finance/blob/main/docs/tutorials/01_portfolio_optimization.ipynb
    means = data.get_period_return_mean_vector()
    cov = data.get_period_return_covariance_matrix()
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
        scale = 1 / (np.sqrt(np.mean(((q * po_problem["cov"]) ** 2).flatten())) + np.sqrt(np.mean((means_in_spins**2).flatten())))
        # scale = 1 / (0.5*(np.sqrt(np.mean((po_problem['cov']**2).flatten()))+np.sqrt(np.mean((po_problem['means']**2).flatten()))))
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


def hamming_weight(index: str) -> int:
    """
    Calculate the hamming weight for a given bitstring
    e.g. 107 == 1101011 --> 5
    """
    binary = bin(index)[2:]
    return binary.count("1")


def yield_all_indices_cosntrained(N: int, K: int):
    """
    Helper function to avoid having to store all indices in memory
    """
    for ind in range(2**N):
        if hamming_weight(ind) == K:
            yield ind


def get_sk_ini(p: int):
    """
    scaled the sk look-up table for the application of portfolio optimziation
    """
    gamma_scale, beta_scale = -0.5, 1
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

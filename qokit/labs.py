###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from __future__ import annotations
import sys
from collections.abc import Sequence, Iterable
import numpy as np
from itertools import combinations
from operator import mul
from functools import reduce
from qokit.fur import TermsType

# approximate optimal merit factor and energy for small Ns
# from Table 1 of https://arxiv.org/abs/1512.02475
true_optimal_mf = {
    3: 4.500,
    4: 4.000,
    5: 6.250,
    6: 2.571,
    7: 8.167,
    8: 4.000,
    9: 3.375,
    10: 3.846,
    11: 12.100,
    12: 7.200,
    13: 14.083,
    14: 5.158,
    15: 7.500,
    16: 5.333,
    17: 4.516,
    18: 6.480,
    19: 6.224,
    20: 7.692,
    21: 8.481,
    22: 6.205,
    23: 5.628,
    24: 8.000,
    25: 8.681,
    26: 7.511,
    27: 9.851,
    28: 7.840,
    29: 6.782,
    30: 7.627,
    31: 7.172,
    32: 8.000,
    33: 8.508,
    34: 8.892,
    35: 8.390,
}

true_optimal_energy = {
    3: 1,
    4: 2,
    5: 2,
    6: 7,
    7: 3,
    8: 8,
    9: 12,
    10: 13,
    11: 5,
    12: 10,
    13: 6,
    14: 19,
    15: 15,
    16: 24,
    17: 32,
    18: 25,
    19: 29,
    20: 26,
    21: 26,
    22: 39,
    23: 47,
    24: 36,
    25: 36,
    26: 45,
    27: 37,
    28: 50,
    29: 62,
    30: 59,
    31: 67,
    32: 64,
    33: 64,
    34: 65,
    35: 73,
}


def get_terms_offset(N: int):
    """
    Retrun `terms` and `offset` for QAOA LABS problem

    Parameters
    ----------
    N : int
        Problem size (number of spins)

    Returns
    -------
    terms : Sequence of `(float, term)`, where `term` is a tuple of ints.
        Each term corresponds to a summand in the cost Hamiltonian
        and th float value is the coefficient of this term.
        e.g. if terms = [(0.5, (0,1)), (0.3, (0,1,2,3))]
        the Hamiltonian is 0.5*Z0Z1 + 0.3*Z0Z1Z2Z3

    offset : int
        energy offset required due to constant factors (identity terms)
        not included in the Hamiltonian

    """
    term_indices, offset = get_energy_term_indices(N)
    terms = [(len(t), t) for t in term_indices]
    return terms, offset


def get_energy_term_indices(N: int):
    """Return indices of Pauli Zs in the LABS problem definition

    Parameters
    ----------
    N : int
        Problem size (number of spins)

    Returns
    -------
    terms : list of tuples
        List of tuples, where each tuple defines a summand
        and contains indices of the Pauli Zs in the product
        e.g. if terms = [(0,1), (0,1,2,3), (1,2)]
        the Hamiltonian is Z0Z1 + Z0Z1Z2Z3 + Z1Z2

    offset : int
        energy offset required due to constant factors (identity terms)
        not included in the Hamiltonian
    """
    # return the indices of Pauli Zs in products
    all_terms = []
    offset = 0
    # sum_{k=1}^{N-1} (\sum_{i=1}^{N-k} s_i s_{i+k})^2
    for k in range(1, N):
        offset += N - k  # quadratic terms go to one
        for i, j in combinations(range(1, N - k + 1), 2):
            # -1 because we index from 1
            # Drop duplicate terms, e.g. Z1Z2Z2Z3 should be just Z1Z3
            if i + k == j:
                all_terms.append(tuple(sorted((i - 1, j + k - 1))))
            else:
                all_terms.append(tuple(sorted((i - 1, i + k - 1, j - 1, j + k - 1))))
    return set(all_terms), offset


def energy_vals(s: Sequence, N: int | None = None) -> float:
    """Compute LABS energy values from a string of spins

    Parameters
    ----------
    s : list-like
        sequence of spins for which to compute the merit factor
        set(s) \in {+1, -1}
    N : int
        Number of spins

    Returns
    -------
    energy_vals : float
        energy_vals of s
    """
    if N is None:
        N = len(s)
    E_s = 0
    for k in range(1, N):
        C_k = 0
        for i in range(1, N - k + 1):
            C_k += s[i - 1] * s[i + k - 1]
        E_s += C_k**2
    return E_s


def energy_vals_from_bitstring(x, N: int | None = None) -> float:
    """Convenience function
    Useful to get the LABS energy values for bitstrings which are {0,1}^N

    Parameters
    ----------
    s : list-like
        sequence of bits for which to compute the merit factor
        set(s) \in {0, 1}
    N : int
        Number of spins


    Returns
    -------
    energy_Vals : float
        energy_vals of s
    """
    return energy_vals(1 - 2 * x, N=N)


def energy_vals_general(s: Sequence, terms: Iterable | None = None, offset: float | None = None, check_parameters: bool = True) -> float:
    """Compute energy values from a string of spins
    Parameters
    ----------
    s : list-like
        sequence of spins for which to compute the merit factor
        set(s) \in {+1, -1}
    terms : list-like, default None
    offset : float, default None
        precomputed output of get_energy_term_indices
        terms, offset = get_energy_term_indices(N)
    check_parameters : bool, default True
        if set to False, no input validation is performed
    Returns
    -------
    energy_vals : float
        energy_vals of s
    """

    if check_parameters:
        assert isinstance(s, Iterable)
        assert set(s).issubset(set([-1, 1]))
    N = len(s)
    if terms is None or offset is None:
        terms, offset = get_energy_term_indices(N)
    E_s = offset
    for term in terms:
        E_s += (4 if len(term) == 4 else 2) * reduce(mul, [s[idx] for idx in term])
    return E_s


def energy_vals_from_bitstring_general(x, terms: Sequence | None = None, offset: float | None = None, check_parameters: bool = False) -> float:
    """Convenience func
    Useful to get the energy values for bitstrings which are {0,1}^N
    Parameters
    ----------
    s : list-like
        sequence of bits for which to compute the merit factor
        set(s) \in {0, 1}
    terms : list-like, default None
    offset : float, default None
        precomputed output of get_energy_term_indices
        terms, offset = get_energy_term_indices(N)
    check_parameters : bool, default True
        if set to False, no input validation is performed
    Returns
    -------
    energy_Vals : float
        energy_vals of s
    """
    return energy_vals_general(1 - 2 * x, terms=terms, offset=offset, check_parameters=check_parameters)  # type: ignore


def slow_merit_factor(s: Sequence, terms: Iterable | None = None, offset: float | None = None, check_parameters: bool = True) -> float:
    """Compute merit factor from a string of spins

    Parameters
    ----------
    s : list-like
        sequence of spins for which to compute the merit factor
        set(s) \in {-1, +1}
    terms : list-like, default None
    offset : float, default None
        precomputed output of get_energy_term_indices
        terms, offset = get_energy_term_indices(N)
    check_parameters : bool, default True
        if set to False, no input validation is performed


    Returns
    -------
    merit_factor : float
        merit factor of s
    """
    if check_parameters:
        assert isinstance(s, Iterable)
        assert set(s).issubset(set([-1, 1]))
    N = len(s)
    if terms is None or offset is None:
        terms, offset = get_energy_term_indices(N)
    E_s = offset
    for term in terms:
        E_s += (4 if len(term) == 4 else 2) * reduce(mul, [s[idx] for idx in term])
    return N**2 / (2 * E_s)


def merit_factor(s: Sequence, N: int | None = None) -> float:
    """Compute merit factor from a string of spins
    Faster implementation that does not rely on terms computation

    Parameters
    ----------
    s : list-like
        sequence of spins for which to compute the merit factor
        set(s) \in {-1, +1}
    N : int
        Number of spins

    Returns
    -------
    merit_factor : float
        merit factor of s
    """
    if N is None:
        N = len(s)
    E_s = energy_vals(s, N=N)
    return N**2 / (2 * E_s)


def negative_merit_factor_from_bitstring(x, N: int | None = None) -> float:
    """Convenience function
    Useful for e.g. QAOA since convention is to minimize, and we want a maximum merit factor

    Parameters
    ----------
    s : list-like
        sequence of spins for which to compute the merit factor
        set(s) \in {-1, +1}
    N : int
        Number of spins

    Returns
    -------
    merit_factor : float
        negative merit factor of s
    """
    return -merit_factor(1 - 2 * x, N=N)


def get_term_indices(N: int) -> list:
    """Return indices of Pauli Zs in the LABS problem definition

    Parameters
    ----------
    N : int
        Problem size (number of spins)

    Returns
    -------
    terms : list of tuples
        List of ordered tuples, where each tuple defines a summand
        and contains indices of the Pauli Zs in the product
        e.g. if terms = [(0,1), (0,1,2,3), (1,2)]
        the Hamiltonian is Z0Z1 + Z0Z1Z2Z3 + Z1Z2
    """
    return list(get_energy_term_indices(N)[0])


def get_terms(N: int) -> TermsType:
    """Return terms definition of the LABS problem

    Parameters
    ----------
    N : int
        Problem size (number of spins)

    Returns
    -------
    terms : TermsType
        List of tuples (number, tuple) where the
        tuple determines the location of Z operators
        and the number is a scaling factor for the product.

        e.g. if terms = [(2, (0,1)), (4, (0,1,2,3)), (2, (1,2))]
        the Hamiltonian is 2*Z0Z1 + 4*Z0Z1Z2Z3 + 2*Z1Z2
    """
    return [(len(x), x) for x in get_term_indices(N)]


def get_depth_optimized_terms(N: int) -> list:
    """Return indices of Pauli Zs in the LABS problem definition. The terms in the returned list are ordered to attempt to compress
    the circuit depth and increase parallelism.

    Parameters
    ----------
    N : int
        Problem size (number of spins)

    Returns
    -------
    terms : list of tuples
        List of ordered tuples, where each tuple defines a summand
        and contains indices of the Pauli Zs in the product
        e.g. if terms = [(0,1), (0,1,2,3), (1,2)]
        the Hamiltonian is Z0Z1 + Z0Z1Z2Z3 + Z1Z2
    """

    done = []  # contains gates that have already been applied
    layers = []

    # prioritize ZZZZ terms
    for pivot in range(N - 3, 0, -1):
        for t in range(1, int((N - pivot - 1) / 2) + 1):
            for k in range(t + 1, N - pivot - t + 1):
                interactions: list[tuple[int, int, int, int] | tuple[int, int]] = [(pivot, pivot + t, pivot + k, pivot + t + k)]
                if set(interactions[0]) in done:
                    continue
                idx_used = list(interactions[0])

                done.append(set(interactions[0]))
                # greedily apply ZZZZ terms to free qubits
                stack = sorted(list(set(range(1, N + 1)) - set(idx_used)))
                try:
                    while stack:
                        f = stack.pop()
                        for s in stack:
                            if f in idx_used:
                                break
                            for th in filter(lambda x: x < s, stack):
                                if s in idx_used:
                                    break
                                a = th - (f - s)
                                if a > 0 and not (a in idx_used) and set([a, th, s, f]) not in done:
                                    stack.remove(a)
                                    stack.remove(s)
                                    stack.remove(th)
                                    idx_used += [a, th, s, f]
                                    done.append(set([a, th, s, f]))
                                    interactions.append((a, th, s, f))
                except IndexError:
                    pass
                # greedily apply ZZ terms to free qubits
                try:
                    stack = sorted(list(set(range(1, N + 1)) - set(idx_used)))

                    while stack:
                        f = stack.pop()
                        for k in range(1, int((f - 1) / 2) + 1):
                            a = f - 2 * k
                            if a > 0 and not (a in idx_used) and set([a, f]) not in done:
                                stack.remove(a)
                                idx_used += [a, f]
                                done.append(set([a, f]))
                                interactions.append((a, f))
                except IndexError:
                    pass
                layers.append(set(j - 1 for j in i) for i in interactions)

    # add any missing ZZ terms not covered by the above. Typically isn't used at high N.
    for pivot in range(N - 2, 0, -1):
        for t in range(1, int((N - pivot) / 2) + 1):
            interactions = [(pivot, pivot + 2 * t)]
            if set(interactions[0]) in done:
                continue
            idx_used = list(interactions[0])
            done.append(set(interactions[0]))
            # greedily apply ZZ terms to free qubits
            try:
                stack = sorted(list(set(range(1, N + 1)) - set(idx_used)))

                while stack:
                    f = stack.pop()
                    for k in range(1, int((f - 1) / 2) + 1):
                        a = f - 2 * k
                        if a > 0 and not (a in idx_used) and set([a, f]) not in done:
                            stack.remove(a)
                            idx_used += [a, f]
                            done.append(set([a, f]))
                            interactions.append((a, f))
            except IndexError:
                pass
            layers.append(set(j - 1 for j in i) for i in interactions)

    # linearize list of cost operator layers
    terms = []
    for layer in layers:
        terms += [tuple(sorted(l)) for l in layer]
    return terms


def get_gate_optimized_terms_naive(N: int, number_of_gate_zones: int = 4):
    """
    Try to naively line up terms to encourage many CNOT cancellations
    """
    terms = []
    for i in range(1, N - 3 + 1):
        for t in range(1, int((N - i - 1) / 2) + 1):
            for k in range(t + 1, N - i - t + 1):
                terms.append((i - 1, i + t - 1, i + k - 1, i + t + k - 1))
    for i in range(1, N - 2 + 1):
        for t in range(1, int((N - i) / 2) + 1):
            terms.append((i - 1, i + 2 * t - 1))

    if number_of_gate_zones:
        k = 0
        while k < len(terms):
            num_zones_left = number_of_gate_zones - len(terms[k]) // 2
            j = k + 1
            swapped = 1
            while j < len(terms) and num_zones_left != 0:
                new_term = terms[j]
                if not set(terms[k]).intersection(set(new_term)):
                    terms.remove(new_term)
                    terms.insert(k + 1, new_term)
                    num_zones_left -= len(new_term) // 2
                    swapped += 1
                j += 1
            k += swapped

    return terms


def get_gate_optimized_terms_greedy(N: int, number_of_gate_zones: int = 4, seed: int | None = None):
    """
    Try to greedly cancel CNOTs for RZZZZ terms
    """

    four_body_by_k = []
    for k in range(1, N - 1):
        terms = []
        for i in range(0, N - k):
            for j in range(i + 1, N - k):
                if i + k < j:
                    terms.append((i, i + k, j, j + k))
        if terms:
            four_body_by_k.append(terms)

    two_body = []
    for i in range(0, N - 2):
        for k in range(1, int((N - i + 1) // 2)):
            two_body.append((i, i + 2 * k))

    circuit = []

    seed = seed if seed else np.random.randint(np.iinfo(np.int32).max)
    print(f"seed: {seed}")
    rng = np.random.default_rng(seed)

    # greedly align four body terms to cancel CNOTs
    for terms in four_body_by_k:
        i = rng.choice(len(terms))
        first_term = terms.pop(i)
        circuit.append(first_term)
        open_tops = [first_term[:2]]
        open_bottoms = [first_term[2:]]

        while terms:
            terms_with_scores = []
            for t in terms:
                sc = [0, 0, t]

                cancels_top = False
                cancels_bottom = False
                if t[:2] in open_tops:
                    sc[0] += 1
                    sc[1] += 1
                    cancels_top = True
                if t[2:] in open_bottoms:
                    sc[0] += 1
                    sc[1] += 1
                    cancels_bottom = True

                qubits_top = set(sum(map(lambda x: list(x), open_tops), []))
                if set(qubits_top).intersection(set(t[2:])):
                    sc[0] -= 1

                qubits_bottom = set(sum(map(lambda x: list(x), open_bottoms), []))
                if set(qubits_bottom).intersection(set(t[:2])):
                    sc[0] -= 1

                if not cancels_top:
                    if set(qubits_top).intersection(set(t[:2])):
                        sc[0] -= 1
                if not cancels_bottom:
                    if set(qubits_bottom).intersection(set(t[2:])):
                        sc[0] -= 1

                terms_with_scores.append(sc)

            score, num_cancels, new_term = tuple(sorted(terms_with_scores, key=lambda x: -x[0])[0])

            terms.remove(new_term)
            circuit.append(new_term)
            open_tops.append(new_term[:2])
            open_bottoms.append(new_term[2:])

    # squeeze two body inbetween aligned four-body terms
    for b in two_body:
        found = False
        for i, term in enumerate(circuit):
            if b == term[:2] or b == term[2:]:
                circuit.insert(i + 1, b)
                found = True
                break
        if not found:
            circuit.append(b)

    if number_of_gate_zones:
        # move terms back to accomodate gate zone parallization
        k = 0
        while k < len(circuit):
            num_zones_left = number_of_gate_zones - len(circuit[k]) // 2
            j = k + 1
            swapped = 1
            while j < len(circuit) and num_zones_left != 0:
                new_term = circuit[j]
                if not set(circuit[k]).intersection(set(new_term)):
                    circuit.remove(new_term)
                    circuit.insert(k + 1, new_term)
                    num_zones_left -= len(new_term) // 2
                    swapped += 1
                j += 1
            k += swapped

    return circuit

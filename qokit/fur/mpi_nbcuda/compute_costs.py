###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numba.cuda


@numba.cuda.jit
def zero_init_kernel(x):
    tid = numba.cuda.grid(1)

    if tid < len(x):
        x[tid] = 0


def zero_init(x):
    zero_init_kernel.forall(len(x))(x)


@numba.cuda.jit
def compute_costs_kernel(costs, coef: float, pos_mask: int, offset: int):
    tid = numba.cuda.grid(1)

    if tid < len(costs):
        parity = numba.cuda.popc((tid + offset) & pos_mask) & 1
        if parity:
            costs[tid] -= coef
        else:
            costs[tid] += coef


def compute_costs(rank: int, n_local_qubits: int, terms, out):
    offset = rank << n_local_qubits
    n = len(out)

    for coef, pos in terms:
        pos_mask = sum(2**x for x in pos)
        compute_costs_kernel.forall(n)(out, coef, pos_mask, offset)

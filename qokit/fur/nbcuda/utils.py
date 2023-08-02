###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
# pyright: reportGeneralTypeIssues=false
import math
import numba.cuda


@numba.cuda.jit
def norm_squared_kernel(sv):
    n = len(sv)
    tid = numba.cuda.grid(1)

    if tid < n:
        sv[tid] = abs(sv[tid]) ** 2


def norm_squared(sv):
    """
    compute norm squared of a statevector
    i.e. convert amplitudes to probabilities
    """
    norm_squared_kernel.forall(len(sv))(sv)


@numba.cuda.jit
def initialize_uniform_kernel(sv, scale):
    n = len(sv)
    tid = numba.cuda.grid(1)

    if tid < n:
        sv[tid] = scale / math.sqrt(n)


def initialize_uniform(sv, scale=1.0):
    """
    initialize a uniform superposition statevector on GPU
    """
    initialize_uniform_kernel.forall(len(sv))(sv, scale)


@numba.cuda.jit
def multiply_kernel(a, b):
    n = len(a)
    tid = numba.cuda.grid(1)

    if tid < n:
        a[tid] = a[tid] * b[tid]


def multiply(a, b):
    multiply_kernel.forall(len(a))(a, b)


@numba.cuda.jit
def copy_kernel(a, b):
    n = len(a)
    tid = numba.cuda.grid(1)

    if tid < n:
        a[tid] = b[tid]


def copy(a, b):
    copy_kernel.forall(len(a))(a, b)


@numba.cuda.reduce
def sum_reduce(a, b):
    return a + b


@numba.cuda.reduce
def real_max_reduce(a, b):
    return max(a.real, b.real)

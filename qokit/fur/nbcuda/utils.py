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
def initialize_uniform_kernel(sv):
    n = len(sv)
    tid = numba.cuda.grid(1)

    if tid < n:
        sv[tid] = 1.0 / math.sqrt(n)


def initialize_uniform(sv):
    """
    initialize a uniform superposition statevector on GPU
    """
    initialize_uniform_kernel.forall(len(sv))(sv)

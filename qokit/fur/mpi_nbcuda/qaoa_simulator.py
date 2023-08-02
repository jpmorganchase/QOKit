###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from __future__ import annotations
import typing
from collections.abc import Sequence
import numpy as np
import math
import numba.cuda

from qokit.fur.nbcuda.qaoa_simulator import DeviceArray
from ..lazy_import import MPI

from ..nbcuda.qaoa_simulator import QAOAFastSimulatorGPUBase, CostsType, TermsType, ParamType
from ..nbcuda.utils import norm_squared, initialize_uniform, multiply, sum_reduce, copy
from ..diagonal_precomputation import precompute_gpu
from .qaoa_fur import apply_qaoa_furx  # , apply_qaoa_furxy_complete, apply_qaoa_furxy_ring
from .compute_costs import compute_costs, zero_init


def mpi_available(allow_single_process=False):
    try:
        size = MPI.COMM_WORLD.Get_size()  # will raise on import if MPI not available
        return size > 1 or allow_single_process
    except Exception:
        return False


def dtype_to_mpi(t):
    if hasattr(MPI, "_typedict"):
        mpi_type = MPI._typedict[np.dtype(t).char]  # type: ignore
    elif hasattr(MPI, "__TypeDict__"):
        mpi_type = MPI.__TypeDict__[np.dtype(t).char]  # type: ignore
    else:
        raise ValueError("cannot convert type")
    return mpi_type


def _set_gpu_device_roundrobin():
    rank = MPI.COMM_WORLD.Get_rank()
    numba.cuda.select_device(rank % len(numba.cuda.gpus))


@numba.cuda.reduce
def sum_reduce(a, b):
    return a + b


@numba.cuda.reduce
def real_max_reduce(a, b):
    return max(a.real, b.real)


def expectation(device_sv, device_Z):
    multiply(device_sv, device_Z)
    return sum_reduce(device_sv)  # type: ignore


def get_costs(terms, N, rank=0):
    c = numba.cuda.device_array(2**N, dtype=np.float64)
    zero_init(c)
    compute_costs(rank, N, terms, c)
    return c


class QAOAFastSimulatorGPUMPIBase(QAOAFastSimulatorGPUBase):
    def __init__(self, n_qubits: int, costs: CostsType | None = None, terms: TermsType | None = None) -> None:
        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()
        _set_gpu_device_roundrobin()

        assert not self._size & (self._size - 1), f"MPI size ({self._size}) is not a power of 2"
        self.n_all_qubits = n_qubits
        self.n_all_states = 2**self.n_all_qubits
        assert self._size <= self.n_all_states, f"MPI size ({self._size}) is larger than the number of states ({self.n_all_states})"
        self.n_local_qubits = n_qubits - int(round(math.log2(self._size)))
        self.n_local_states = 2**self.n_local_qubits
        self._local_index_start = self._rank << self.n_local_qubits
        self._local_index_end = (self._rank + 1) << self.n_local_qubits

        # proceed with normal initialization
        super().__init__(self.n_local_qubits, costs, terms)

    def _diag_from_costs(self, costs: CostsType) -> DeviceArray:
        return self._get_local_slice(numba.cuda.to_device(costs))

    def _diag_from_terms(self, terms: TermsType) -> DeviceArray:
        out = numba.cuda.device_array(self.n_states, dtype="float32")  # type: ignore
        precompute_gpu(self._rank, self.n_local_qubits, terms, out)
        return out

    def _get_local_slice(self, a: typing.Any) -> typing.Any:
        if len(a) == self.n_local_states:
            return a
        elif len(a) == self.n_all_states:
            return a[self._local_index_start : self._local_index_end]
        else:
            raise ValueError(f"Length of the array must be either 2^{self.n_local_qubits} or 2^{self.n_all_qubits}, got {len(a)} instead")

    def _initialize(
        self,
        sv0: np.ndarray | None = None,
    ) -> None:
        if sv0 is None:
            initialize_uniform(self._sv_device, 1.0 / math.sqrt(self._size))
        else:
            numba.cuda.to_device(np.asarray(self._get_local_slice(sv0), dtype="complex"), to=self._sv_device)

    def _get_optimum_overlap(self, probs_host, costs_host, broadcast=True):
        """
        Get overlap from probabilites and costs using MPI
        """
        local_optimal_energy = np.min(costs_host)
        optimal_probs = probs_host[costs_host == local_optimal_energy]
        local_opt_overlap = sum(optimal_probs)
        opt_rank = np.array([local_opt_overlap, local_optimal_energy])
        global_opt = 0.0
        if self._rank == 0:
            opt_rank = np.array(opt_rank)
            p_rank = opt_rank[:, 0]
            e_rank = opt_rank[:, 1]
            optimal_energy = np.min(e_rank)
            optimal_probs = p_rank[e_rank == optimal_energy]
            global_opt = np.sum(optimal_probs)
            if broadcast:
                # send to all ranks using broadcast
                self._comm.Bcast(global_opt, root=0)
        else:
            if not broadcast:
                global_opt = 0.0
            else:
                global_opt = self._comm.Bcast(global_opt, root=0)
        return global_opt

    def _get_global_statevector_cpu(self):
        chunks = self._comm.allgather(self._sv_device.copy_to_host())
        return np.concatenate(chunks)

    def simulate_qaoa(
        self,
        gammas: ParamType,
        betas: ParamType,
        sv0: np.ndarray | None = None,
        **kwargs,
    ) -> DeviceArray:
        """
        simulator QAOA circuit using FUR
        """
        self._initialize(sv0=sv0)
        self._apply_qaoa(list(gammas), list(betas), **kwargs)
        return self._sv_device

    def get_statevector(self, result: DeviceArray, **kwargs) -> np.ndarray:
        mpi_gather = kwargs.get("mpi_gather", True)
        if mpi_gather:
            chunks = self._comm.allgather(result.copy_to_host())
            return np.concatenate(chunks)
        else:
            return result.copy_to_host()

    def get_probabilities(self, result: DeviceArray, **kwargs) -> np.ndarray:
        mpi_gather = kwargs.get("mpi_gather", True)
        preserve_state = kwargs.get("preserve_state", True)
        if preserve_state:
            result_orig = result
            result = numba.cuda.device_array_like(result_orig)
            copy(result, result_orig)
        norm_squared(result)
        if mpi_gather:
            chunks = self._comm.allgather(result.copy_to_host())
            return np.concatenate(chunks)
        else:
            return result.copy_to_host()

    def get_expectation(self, result: DeviceArray, costs: CostsType | None = None, **kwargs) -> float:
        if costs is None:
            costs = self._hc_diag
        else:
            costs = self._diag_from_costs(costs)
        preserve_state = kwargs.get("preserve_state", True)
        if preserve_state:
            result_orig = result
            result = numba.cuda.device_array_like(result_orig)
            copy(result, result_orig)
        norm_squared(result)
        multiply(result, costs)
        local_sum = sum_reduce(result).real  # type: ignore
        global_sum = self._comm.allreduce(local_sum, op=MPI.SUM)
        return global_sum

    def get_overlap(self, result: DeviceArray, costs: CostsType | None = None, indices: np.ndarray | Sequence[int] | None = None, **kwargs) -> float:
        """
        Get probability corresponding to indices or to ground state of costs

        Parameters
        ----------
        result : DeviceArray
        costs : CostsType, optional
            If None, use the ground state of the problem Hamiltonian
        indices : np.ndarray or typing.Sequence[int], optional
            Specify indices of statevector to sum over

        """
        kwargs.pop("mpi_gather", None)
        broadcast = kwargs.pop("mpi_broadcast_float", True)
        probs = self.get_probabilities(result, mpi_gather=False, **kwargs)
        if costs is None:
            costs_host = self._hc_diag.copy_to_host()
        else:
            costs_host = np.asarray(costs)

        return self._get_optimum_overlap(probs, costs_host, broadcast=broadcast)


class QAOAFURXSimulatorGPUMPI(QAOAFastSimulatorGPUMPIBase):
    def _apply_qaoa(self, gammas: Sequence[float], betas: Sequence[float], **kwargs):
        apply_qaoa_furx(self._sv_device, gammas, betas, self._hc_diag, self.n_local_qubits, self.n_all_qubits, self._comm)


# class QAOAFURXYRingSimulatorGPUMPI(QAOAFastSimulatorGPUMPIBase):
#     def _apply_qaoa(self, gammas: Sequence[float], betas: Sequence[float], n_trotters: int = 1):
#         apply_qaoa_furxy_ring(self._sv_device, gammas, betas, self._hc_diag_device, self.n_qubits, n_trotters=n_trotters)


# class QAOAFURXYCompleteSimulatorGPUMPI(QAOAFastSimulatorGPUMPIBase):
#     def _apply_qaoa(self, gammas: Sequence[float], betas: Sequence[float], n_trotters: int = 1):
#         apply_qaoa_furxy_complete(self._sv_device, gammas, betas, self._hc_diag_device, self.n_qubits, n_trotters=n_trotters)

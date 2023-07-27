# don't report type errors in this file, since custatevec is not typed
# pyright: reportGeneralTypeIssues=false
import typing
from mpi4py import MPI
import cupy as cp
import numpy as np
import cuquantum
from cuquantum import custatevec as cusv
import cuquantum.custatevec as cusv
from contextlib import contextmanager
import socket
import numba.cuda

from qokit.fur.mpi_nbcuda.qaoa_simulator import QAOAFastSimulatorGPUMPIBase
from qokit.fur.nbcuda.diagonal import apply_diagonal
import time
from cuquantum import cudaDataType

cqdt = cuquantum.cudaDataType
N_LOCAL_INDEX_BITS = 8

# -- Cuda Statevec helper functions
# Adapted from examples


def get_cusv_communicator(handle):
    name, _ = MPI.get_vendor()
    if name == "Open MPI":
        # use built-in OpenMPI communicator
        communicator_type = cusv.CommunicatorType.OPENMPI
        soname = ""
    elif name == "MPICH":
        # use built-in MPICH communicator
        communicator_type = cusv.CommunicatorType.MPICH
        # work around a Python limitation as discussed in NVIDIA/cuQuantum#31
        soname = "/opt/cray/pe/mpich/default/ofi/gnu/9.1/lib/libmpi.so"
    else:
        # use external communicator
        communicator_type = cusv.CommunicatorType.EXTERNAL
        # please compile mpicomm.c to generate the shared library and place its path here
        soname = ""
        if not soname:
            raise ValueError("please supply the soname to the shared library providing " "an external communicator for cuStateVec")

    communicator = cusv.communicator_create(handle, communicator_type, soname)
    return communicator


from dataclasses import dataclass


@dataclass
class CudaDistributedContext:
    """
    Class to store cuda distributed swap worker and p2p communication events
    """

    remote_events = None
    d_sub_svs_p2p = None
    handle = None
    swap_worker = None
    scheduler = None
    communicator = None
    comm = None
    local_event: cp.cuda.Event = None
    local_stream: cp.cuda.Stream = None
    sv_deviceptr = None


def distribute_device_handles(comm, deviceptr, eventptr):
    ipc_mem_handle = cp.cuda.runtime.ipcGetMemHandle(deviceptr)
    ipc_mem_handles = comm.allgather(ipc_mem_handle)
    # comm.Barrier()
    ipc_event_handle = cp.cuda.runtime.ipcGetEventHandle(eventptr)
    ipc_event_handles = comm.allgather(ipc_event_handle)
    return ipc_mem_handles, ipc_event_handles


def setup_p2p_comm(ctx: CudaDistributedContext, n_p2p_bits, sv_index):
    sub_sv_indices_p2p = []
    ctx.d_sub_svs_p2p = []
    ctx.remote_events = []
    # the number of index bits corresponding to sub state vectors accessible via GPUDirect P2P,
    # and it should be adjusted based on the number of GPUs/node, N, participating in the distributed
    # state vector (N=2^n_p2p_device_bits) that supports P2P data transfer
    n_sub_svs_p2p = 1 << n_p2p_bits
    # bind the device to the process
    # this is based on the assumption of the global rank placement that the
    # processes are mapped to nodes in contiguous chunks (see the comment below)
    num_devices = cp.cuda.runtime.getDeviceCount()
    assert num_devices > 0, "No available CUDA devices found"

    if n_p2p_bits > 0:
        # distribute device memory handles
        # under the hood the handle is stored as a Python bytes object
        ipc_mem_handles, ipc_event_handles = distribute_device_handles(ctx.comm, ctx.sv_deviceptr, ctx.local_event.ptr)
        if sv_index < 8:
            print(f"25 rank {sv_index} Handles distributed", flush=True)

        # get remote device pointers and events
        # this calculation assumes that the global rank placement is done in a round-robin fashion
        # across nodes, so for example if n_p2p_device_bits=2 there are 2^2=4 processes/node (and
        # 1 GPU/progress) and we expect the global MPI ranks to be assigned as
        #   0  1  2  3 -> node 0
        #   4  5  6  7 -> node 1
        #   8  9 10 11 -> node 2
        #             ...
        # if the rank placement scheme is different, you will need to calculate based on local MPI
        # rank/size, as CUDA IPC is only for intra-node, not inter-node, communication.
        p2p_sub_sv_index_begin = (sv_index // n_sub_svs_p2p) * n_sub_svs_p2p
        p2p_sub_sv_index_end = p2p_sub_sv_index_begin + n_sub_svs_p2p
        for p2p_sub_sv_index in range(p2p_sub_sv_index_begin, p2p_sub_sv_index_end):
            if sv_index == p2p_sub_sv_index:
                continue  # don't need local sub state vector pointer
            sub_sv_indices_p2p.append(p2p_sub_sv_index)

            dst_mem_handle = ipc_mem_handles[p2p_sub_sv_index]
            # default is to use cudaIpcMemLazyEnablePeerAccess
            d_sub_sv_p2p = cp.cuda.runtime.ipcOpenMemHandle(dst_mem_handle)
            ctx.d_sub_svs_p2p.append(d_sub_sv_p2p)

            event_p2p = cp.cuda.runtime.ipcOpenEventHandle(ipc_event_handles[p2p_sub_sv_index])
            ctx.remote_events.append(event_p2p)

        # set p2p sub state vectors
        assert len(ctx.d_sub_svs_p2p) == len(sub_sv_indices_p2p) == len(ctx.remote_events)
        if sv_index < 8:
            print(f"26 rank {sv_index} set sub svs", flush=True)
        cusv.sv_swap_worker_set_sub_svs_p2p(ctx.handle, ctx.swap_worker, ctx.d_sub_svs_p2p, sub_sv_indices_p2p, ctx.remote_events, len(ctx.d_sub_svs_p2p))


def destroy_cuda_distributed_context(ctx: CudaDistributedContext):
    cusv.dist_index_bit_swap_scheduler_destroy(ctx.handle, ctx.scheduler)
    cusv.sv_swap_worker_destroy(ctx.handle, ctx.swap_worker)
    cusv.communicator_destroy(ctx.handle, ctx.communicator)
    cusv.destroy(ctx.handle)
    # free IPC pointers and events
    for d_sub_sv in ctx.d_sub_svs_p2p:
        cp.cuda.runtime.ipcCloseMemHandle(d_sub_sv)
    for event in ctx.remote_events:
        cp.cuda.runtime.eventDestroy(event)


def setup_distributed_cusv_swap(rank, size, n_global_index_bits, n_local_index_bits, n_p2p_device_bits, sv_deviceptr):
    ctx = CudaDistributedContext()
    ctx.sv_deviceptr = sv_deviceptr

    # data type of the state vector, acceptable values are CUDA_C_32F and CUDA_C_64F.
    sv_data_type = cqdt.CUDA_C_64F
    sv_dtype = cp.complex128 if sv_data_type == cqdt.CUDA_C_64F else cp.complex64

    # transfer workspace size
    transfer_workspace_size = 1 << n_local_index_bits
    # allocate stream and event
    ctx.local_stream = cp.cuda.Stream()
    ctx.local_event = cp.cuda.Event(disable_timing=True, interprocess=True)
    # create cuStateVec handle
    ctx.handle = cusv.create()
    ctx.communicator = get_cusv_communicator(ctx.handle)
    ctx.comm = MPI.COMM_WORLD
    if rank < 2:
        print(f"20 rank {rank} ctx.communicator {ctx.communicator}", flush=True)

    # -- workspace and swap worker stuff
    # create sv segment swap worker
    ctx.swap_worker, extra_workspace_size, min_transfer_workspace_size = cusv.sv_swap_worker_create(
        ctx.handle, ctx.communicator, sv_deviceptr, rank, ctx.local_event.ptr, sv_data_type, ctx.local_stream.ptr
    )

    # set extra workspace
    d_extra_workspace = cp.cuda.alloc(extra_workspace_size)
    cusv.sv_swap_worker_set_extra_workspace(ctx.handle, ctx.swap_worker, d_extra_workspace.ptr, extra_workspace_size)

    if rank < 2:
        print(f"21 rank {rank} Created workspace", flush=True)
    # set transfer workspace
    # The size should be equal to or larger than min_transfer_workspace_size
    # Depending on the systems, larger transfer workspace can improve the performance
    transfer_workspace_size = max(min_transfer_workspace_size, transfer_workspace_size)
    d_transfer_workspace = cp.cuda.alloc(transfer_workspace_size)
    cusv.sv_swap_worker_set_transfer_workspace(ctx.handle, ctx.swap_worker, d_transfer_workspace.ptr, transfer_workspace_size)
    # --
    if rank < 2:
        print(f"22 rank {rank} Finished workspace", flush=True)
    setup_p2p_comm(ctx, n_p2p_device_bits, rank)
    # create distributed index bit swap scheduler
    if rank < 2:
        print(f"23 rank {rank} P2P comm set up", flush=True)
    ctx.scheduler = cusv.dist_index_bit_swap_scheduler_create(ctx.handle, n_global_index_bits, n_local_index_bits)

    # set the index bit swaps to the scheduler
    # n_swap_batches is obtained by the call.  This value specifies the number of loops

    return ctx


def run_distributed_index_bit_swaps(ctx, index_bit_swaps, rank):
    ctx.local_stream.synchronize()
    numba.cuda.synchronize()
    ctx.comm.barrier()
    start = time.perf_counter()
    mask_bit_string, mask_ordering = [], []
    assert len(mask_bit_string) == len(mask_ordering)
    n_swap_batches = cusv.dist_index_bit_swap_scheduler_set_index_bit_swaps(
        ctx.handle, ctx.scheduler, index_bit_swaps, len(index_bit_swaps), mask_bit_string, mask_ordering, len(mask_bit_string)
    )

    for swap_batch_index in range(n_swap_batches):
        # get parameters
        parameters = cusv.dist_index_bit_swap_scheduler_get_parameters(ctx.handle, ctx.scheduler, swap_batch_index, rank)

        # "rank == sub_sv_index" is assumed in the present sample.
        dst_rank = parameters.dst_sub_sv_index
        cusv.sv_swap_worker_set_parameters(ctx.handle, ctx.swap_worker, parameters, dst_rank)
        cusv.sv_swap_worker_execute(ctx.handle, ctx.swap_worker, 0, parameters.transfer_size)

    # synchronize all operations on device
    return
    ctx.local_stream.synchronize()
    numba.cuda.synchronize()
    ctx.comm.barrier()
    # barrier here for time measurement
    ctx.comm.barrier()
    elapsed = time.perf_counter() - start
    if rank == 0:
        # output benchmark result
        elm_size = 16 if sv_data_type == cqdt.CUDA_C_64F else 8
        fraction = 1.0 - 0.5 ** len(index_bit_swaps)
        transferred = 2**n_local_index_bits * fraction * elm_size
        bw = transferred / elapsed * 1e-9
        print(f"BW {bw} [GB/s]")

    # free all resources


# --

# -- Gate-based diagonal application


def apply_gate(handle, device_ptr, angle, targets, N):
    index_bits = N
    controls = np.array([], dtype=np.int32)
    targets = np.array(targets, dtype=np.int32)
    paulis = np.asarray([cusv.Pauli.Z] * len(targets), dtype=np.int32)

    cusv.apply_pauli_rotation(
        handle,
        device_ptr,
        cqdt.CUDA_C_64F,
        index_bits,
        angle,
        paulis.ctypes.data,
        targets.ctypes.data,
        len(targets),
        controls.ctypes.data,
        controls.ctypes.data,
        len(controls),
    )


def apply_diagonal_gates(device_ptr, gamma, terms, N):
    handle = cusv.create()
    for c, targets in terms:
        apply_gate(handle, device_ptr, gamma * c, targets, N)
    cusv.destroy(handle)


def _set_gpu_device_roundrobin():
    rank = MPI.COMM_WORLD.Get_rank()
    cp.cuda.runtime.setDevice(rank % cp.cuda.runtime.getDeviceCount())
    if rank < 8:
        print(f"Rank {rank} uses device {cp.cuda.runtime.getDevice()}", flush=True)
    return rank


class CuStateVecMPIQAOASimulator(QAOAFastSimulatorGPUMPIBase):
    def __init__(
        self,
        n_qubits: int,
        costs: typing.Optional[typing.Sequence[float]] = None,
        terms: typing.Optional[typing.Sequence[typing.Tuple[float, typing.Sequence[int]]]] = None,
    ) -> None:
        """
        Args:
            n_qubits (int): The number of qubits in the system.
            costs: The cost Hamiltonian diagonal.
            terms: The cost Hamiltonian terms.
        """
        # set GPU device before allocating statevector in super().__init__
        rank = _set_gpu_device_roundrobin()
        # invert qubit order in terms for cuquantum convention
        if terms is not None:
            terms = [(c, tuple(n_qubits - 1 - i for i in term)) for c, term in terms]
        super().__init__(n_qubits, costs, terms)
        self._terms = terms
        self._handle = cusv.create()
        n_local_bits, n_p2p_bits, n_global_bits = self._get_topology()
        if rank < 2:
            print("2 Topology:", n_local_bits, n_p2p_bits, n_global_bits, flush=True)
        n_global_index_bits = n_global_bits + n_p2p_bits

        self._nv_swap_context = setup_distributed_cusv_swap(
            self._rank,
            self._size,
            n_global_index_bits,
            n_local_bits,
            n_p2p_bits,
            self._sv_deviceptr,
        )
        if rank < 2:
            print("3 Set up the context.", flush=True)
        # nv_run_distributed_index_swap(
        # print('Rank', self._rank, ', hostname', socket.gethostname(),
        #      f'{self.n_all_qubits=}, {self.n_local_qubits=}')

    def _get_topology(self):
        """
        Returns:
            n_local_bits (int): number of bits in statevector that correspond to
                local sv
            n_p2p_bits (int): log2 number of gpus used per node
            n_global_bits (int): log2 number of nodes

            The three numbers add up to total sv index bits
            This function has to be called after super().__init__
        """
        num_devices = cp.cuda.runtime.getDeviceCount()
        n_p2p_bits = int(np.log2(num_devices))
        n_ranks = int(np.log2(self._size))
        return self.n_all_qubits - n_ranks, n_p2p_bits, n_ranks - n_p2p_bits

    def _apply_swap(self, deviceptr):
        # compute n_global_index_bits from the size
        # n_global_index_bits = log2(size)
        n_local_bits, n_p2p_bits, n_global_bits = self._get_topology()
        n_bits = sum([n_global_bits, n_local_bits, n_p2p_bits])

        # Iterate over largest K local index bits and K global qubits
        index_bit_swaps = []
        n_index_bit_swaps = n_global_bits + n_p2p_bits
        for idx in range(n_index_bit_swaps):
            index_bit_swaps.append((n_local_bits - 1 - idx, n_bits - idx - 1))
        run_distributed_index_bit_swaps(self._nv_swap_context, index_bit_swaps, self._rank)

    @property
    def _sv_deviceptr(self):
        return self._sv_device.__cuda_array_interface__["data"][0]

    def _apply_mixer(self, beta):
        controls = np.asarray([], dtype=np.int32)
        paulis = np.asarray([cusv.Pauli.X], dtype=np.int32)
        N = self.n_local_qubits
        dtype = cqdt.CUDA_C_64F

        # -- apply local gates
        for i in range(N):
            targets = np.asarray([i], dtype=np.int32)
            # apply Pauli operator
            cusv.apply_pauli_rotation(
                self._handle,
                self._sv_deviceptr,
                dtype,
                N,
                -beta,
                paulis.ctypes.data,
                targets.ctypes.data,
                len(targets),
                controls.ctypes.data,
                controls.ctypes.data,
                len(controls),
            )

        # -- apply global gates
        # print(f"Before swap {self._rank=}, {self._sv_device[:16].copy_to_host()}")
        start = time.time()
        self._apply_swap(self._sv_deviceptr)
        n_local_bits, n_p2p_bits, n_global_bits = self._get_topology()
        n_index_bit_swaps = n_global_bits + n_p2p_bits
        for i in range(n_local_bits - n_index_bit_swaps, n_local_bits):
            targets = np.asarray([i], dtype=np.int32)
            # apply Pauli operator
            cusv.apply_pauli_rotation(
                self._handle,
                self._sv_deviceptr,
                dtype,
                N,
                -beta,
                paulis.ctypes.data,
                targets.ctypes.data,
                len(targets),
                controls.ctypes.data,
                controls.ctypes.data,
                len(controls),
            )

        self._apply_swap(self._sv_deviceptr)

    def _apply_qaoa(self, gammas, betas):
        for gamma, beta in zip(gammas, betas):
            apply_diagonal(self._sv_device, gamma, self._hc_diag_device)
            self._apply_mixer(beta)

    def _apply_qaoa_gates(self, gammas, betas):
        for gamma, beta in zip(gammas, betas):
            apply_diagonal_gates(self._sv_deviceptr, gamma, self._terms, self.n_local_qubits)
            self._apply_mixer(beta)

    def __del__(self):
        if hasattr(self, "_handle"):
            cusv.destroy(self._handle)
        if hasattr(self, "_nv_swap_context"):
            destroy_cuda_distributed_context(self._nv_swap_context)

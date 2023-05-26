import typing
import numpy as np
import numba.cuda

from ..qaoa_simulator_base import QAOAFastSimulatorBase
from .qaoa_fur import apply_qaoa_furx, apply_qaoa_furxy_complete, apply_qaoa_furxy_ring
from .utils import norm_squared, initialize_uniform


class QAOAFastSimulatorGPUBase(QAOAFastSimulatorBase):
    def __init__(self, n_qubits: int, costs: typing.Sequence[float]) -> None:
        super().__init__(n_qubits, costs)
        self._sv_device = numba.cuda.device_array(self.n_states, "complex")
        self._hc_diag_device = numba.cuda.to_device(self.hc_diag)

    def _apply_qaoa(self, gammas: typing.Sequence[float], betas: typing.Sequence[float], **kwargs):
        raise NotImplementedError

    def _initialize(
        self,
        sv0: typing.Optional[np.ndarray] = None,
    ) -> None:
        if sv0 is None:
            initialize_uniform(self._sv_device)
        else:
            numba.cuda.to_device(np.asarray(sv0, dtype="complex"), to=self._sv_device)

    def simulate_qaoa(
        self,
        gammas: typing.Sequence[float],
        betas: typing.Sequence[float],
        sv0: typing.Optional[np.ndarray] = None,
        return_probabilities: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        simulator QAOA circuit using FUR
        If return_probabilities is True, probabilities vector instead of statevector will be returned
        simulator QAOA circuit using FUR
        @param gammas parameters for the phase separating layers
        @param betas parameters for the mixing layers
        @param sv0 (optional) initial statevector, default is uniform superposition state
        @param return_probabilities (optional) return probabilities vector instead of statevector, default is False
        @return statevector or vector of probabilities
        """
        self._initialize(sv0=sv0)
        self._apply_qaoa(gammas, betas, **kwargs)
        if return_probabilities:
            norm_squared(self._sv_device)
            return self._sv_device.copy_to_host().real
        else:
            return self._sv_device.copy_to_host()


class QAOAFURXSimulatorGPU(QAOAFastSimulatorGPUBase):
    def _apply_qaoa(self, gammas: typing.Sequence[float], betas: typing.Sequence[float]):
        apply_qaoa_furx(self._sv_device, gammas, betas, self._hc_diag_device, self.n_qubits)


class QAOAFURXYRingSimulatorGPU(QAOAFastSimulatorGPUBase):
    def _apply_qaoa(self, gammas: typing.Sequence[float], betas: typing.Sequence[float], n_trotters: int = 1):
        apply_qaoa_furxy_ring(self._sv_device, gammas, betas, self._hc_diag_device, self.n_qubits, n_trotters=n_trotters)


class QAOAFURXYCompleteSimulatorGPU(QAOAFastSimulatorGPUBase):
    def _apply_qaoa(self, gammas: typing.Sequence[float], betas: typing.Sequence[float], n_trotters: int = 1):
        apply_qaoa_furxy_complete(self._sv_device, gammas, betas, self._hc_diag_device, self.n_qubits, n_trotters=n_trotters)

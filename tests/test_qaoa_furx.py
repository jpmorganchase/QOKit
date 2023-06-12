import pytest
import numpy as np
import numba.cuda
import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.providers.aer import AerSimulator

from qokit.fur import QAOAFURXSimulatorC, QAOAFURXSimulator, QAOAFURXSimulatorGPU
from qokit.fur.c.gates import furx as furx_c
from qokit.fur.python.gates import furx
from qokit.fur.nbcuda.gates import furx as furx_gpu


class _TestFURXBase(object):
    furx: staticmethod
    simulator_class: type
    get_sv = staticmethod(lambda sv: sv)

    def test_rx(self):
        for N in [4, 5]:
            beta = np.random.uniform(0, np.pi)
            q = 2

            sv0 = np.zeros(2**N)
            sv0[0] = 1

            sv_fur = self.get_sv(self.furx(sv0, beta, q))

            qc = QuantumCircuit(N)

            qc.append(
                qiskit.circuit.library.RXGate(beta),
                [q],
            )

            qc.save_state()
            backend = AerSimulator(method="statevector")
            sv_qiskit = execute(qc, backend).result().get_statevector()

            assert sv_qiskit.equiv(sv_fur)

    def test_rx_all(self):
        for N in [4, 5]:
            beta = np.random.uniform(0, np.pi)
            sim = self.simulator_class(N, np.zeros(2**N))

            sv0 = np.zeros(2**N)
            sv0[0] = 1

            sv_fur = self.get_sv(sim.simulate_qaoa([0], [beta], sv0=sv0))

            qc = QuantumCircuit(N)

            for q in range(N):
                qc.append(
                    qiskit.circuit.library.RXGate(beta),
                    [q],
                )

            qc.save_state()
            backend = AerSimulator(method="statevector")
            sv_qiskit = execute(qc, backend).result().get_statevector()

            assert sv_qiskit.equiv(sv_fur)


class TestFURXC(_TestFURXBase):
    furx = staticmethod(furx_c)
    simulator_class = QAOAFURXSimulatorC
    get_sv = staticmethod(lambda sv: sv.get_complex())


class TestFURXPython(_TestFURXBase):
    furx = staticmethod(furx)
    simulator_class = QAOAFURXSimulator


@pytest.mark.skipif(not numba.cuda.is_available(), reason="no GPU devices found")
class TestFURXGPU(_TestFURXBase):
    furx = staticmethod(furx_gpu)
    simulator_class = QAOAFURXSimulatorGPU

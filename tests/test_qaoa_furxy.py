import pytest
import numpy as np
import numba.cuda
import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.providers.aer import AerSimulator

from qokit.fur import (
    QAOAFURXYRingSimulatorC,
    QAOAFURXYCompleteSimulatorC,
    QAOAFURXYRingSimulator,
    QAOAFURXYCompleteSimulator,
    QAOAFURXYRingSimulatorGPU,
    QAOAFURXYCompleteSimulatorGPU,
)
from qokit.fur.c.gates import furxy as furxy_c
from qokit.fur.python.gates import furxy as furxy
from qokit.fur.nbcuda.gates import furxy as furxy_gpu


class _TestFURXYBase(object):
    furxy: staticmethod
    ring_simulator_class: type
    complete_simulator_class: type
    get_sv = staticmethod(lambda sv: sv)

    def test_rxy(self):
        for N in [4, 5]:
            beta = np.random.uniform(0, np.pi)
            q1 = 0
            q2 = 2

            sv0 = np.zeros(2**N)
            sv0[1] = 1

            sv_fur = self.get_sv(self.furxy(sv0, beta, q1, q2))

            qc = QuantumCircuit(N)
            qc.rx(np.pi, 0)

            qc.append(
                qiskit.circuit.library.XXPlusYYGate(beta),
                [q1, q2],
            )

            qc.save_state()
            backend = AerSimulator(method="statevector")
            sv_qiskit = execute(qc, backend).result().get_statevector()

            assert sv_qiskit.equiv(sv_fur)

    def test_rxy_ring(self):
        for N in [4, 5]:
            beta = np.random.uniform(0, np.pi)
            sim = self.ring_simulator_class(N, np.zeros(2**N))

            sv0 = np.zeros(2**N)
            sv0[1] = 1

            sv_fur = self.get_sv(sim.simulate_qaoa([0], [beta], sv0=sv0))

            qc = QuantumCircuit(N)
            qc.rx(np.pi, 0)

            for i in range(2):
                for q1 in range(i, N - 1, 2):
                    qc.append(
                        qiskit.circuit.library.XXPlusYYGate(beta),
                        [q1, q1 + 1],
                    )
            qc.append(
                qiskit.circuit.library.XXPlusYYGate(beta),
                [0, N - 1],
            )

            qc.save_state()
            backend = AerSimulator(method="statevector")
            sv_qiskit = execute(qc, backend).result().get_statevector()

            assert sv_qiskit.equiv(sv_fur)

    def test_rxy_complete(self):
        for N in [4, 5]:
            beta = np.random.uniform(0, np.pi)
            sim = self.complete_simulator_class(N, np.zeros(2**N))

            sv0 = np.zeros(2**N)
            sv0[1] = 1

            sv_fur = self.get_sv(sim.simulate_qaoa([0], [beta], sv0=sv0))

            qc = QuantumCircuit(N)
            qc.rx(np.pi, 0)

            for q1 in range(N - 1):
                for q2 in range(q1 + 1, N):
                    qc.append(
                        qiskit.circuit.library.XXPlusYYGate(beta),
                        [q1, q2],
                    )

            qc.save_state()
            backend = AerSimulator(method="statevector")
            sv_qiskit = execute(qc, backend).result().get_statevector()

            assert sv_qiskit.equiv(sv_fur)

    def test_rxy_ring_trotter(self):
        for N in [4, 5]:
            beta = np.random.uniform(0, np.pi)
            n_trotters = 2

            sim = self.ring_simulator_class(N, np.zeros(2**N))

            sv0 = np.zeros(2**N)
            sv0[1] = 1

            sv_fur = self.get_sv(sim.simulate_qaoa([0], [beta], sv0=sv0, n_trotters=n_trotters))

            qc = QuantumCircuit(N)
            qc.rx(np.pi, 0)

            for _ in range(n_trotters):
                for i in range(2):
                    for q1 in range(i, N - 1, 2):
                        qc.append(
                            qiskit.circuit.library.XXPlusYYGate(beta / n_trotters),
                            [q1, q1 + 1],
                        )
                qc.append(
                    qiskit.circuit.library.XXPlusYYGate(beta / n_trotters),
                    [0, N - 1],
                )

            qc.save_state()
            backend = AerSimulator(method="statevector")
            sv_qiskit = execute(qc, backend).result().get_statevector()

            assert sv_qiskit.equiv(sv_fur)

    def test_rxy_complete_trotter(self):
        for N in [4, 5]:
            beta = np.random.uniform(0, np.pi)
            n_trotters = 2

            sim = self.complete_simulator_class(N, np.zeros(2**N))

            sv0 = np.zeros(2**N)
            sv0[1] = 1

            sv_fur = self.get_sv(sim.simulate_qaoa([0], [beta], sv0=sv0, n_trotters=n_trotters))

            qc = QuantumCircuit(N)
            qc.rx(np.pi, 0)

            for _ in range(n_trotters):
                for q1 in range(N - 1):
                    for q2 in range(q1 + 1, N):
                        qc.append(
                            qiskit.circuit.library.XXPlusYYGate(beta / n_trotters),
                            [q1, q2],
                        )

            qc.save_state()
            backend = AerSimulator(method="statevector")
            sv_qiskit = execute(qc, backend).result().get_statevector()

            assert sv_qiskit.equiv(sv_fur)


class TestFURXYC(_TestFURXYBase):
    furxy = staticmethod(furxy_c)
    ring_simulator_class = QAOAFURXYRingSimulatorC
    complete_simulator_class = QAOAFURXYCompleteSimulatorC
    get_sv = staticmethod(lambda sv: sv.get_complex())


class TestFURXYPython(_TestFURXYBase):
    furxy = staticmethod(furxy)
    ring_simulator_class = QAOAFURXYRingSimulator
    complete_simulator_class = QAOAFURXYCompleteSimulator


@pytest.mark.skipif(not numba.cuda.is_available(), reason="no GPU devices found")
class TestFURXYGPU(_TestFURXYBase):
    furxy = staticmethod(furxy_gpu)
    ring_simulator_class = QAOAFURXYRingSimulatorGPU
    complete_simulator_class = QAOAFURXYCompleteSimulatorGPU

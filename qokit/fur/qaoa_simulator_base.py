###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import typing


class QAOAFastSimulatorBase(object):
    def __init__(self, n_qubits: int, costs: typing.Sequence[float]) -> None:
        """
        n_qubits: total number of qubits in the circuit
        costs: array containing values of the cost function at each computational basis state
        """
        self.n_qubits = n_qubits
        self.n_states = 2**n_qubits
        self.hc_diag = costs
        assert len(costs) == self.n_states

import numba.cuda
from .qaoa_simulator_base import QAOAFastSimulatorBase, ParamType, CostsType, TermsType
from .c.qaoa_simulator import QAOAFURXSimulatorC, QAOAFURXYRingSimulatorC, QAOAFURXYCompleteSimulatorC
from .python.qaoa_simulator import QAOAFURXSimulator, QAOAFURXYRingSimulator, QAOAFURXYCompleteSimulator
from .nbcuda.qaoa_simulator import QAOAFURXSimulatorGPU, QAOAFURXYRingSimulatorGPU, QAOAFURXYCompleteSimulatorGPU
from .mpi_nbcuda.qaoa_simulator import QAOAFURXSimulatorGPUMPI
from .mpi_nbcuda.qaoa_simulator import mpi_available
from .c import is_available as c_available

# from .mpi_custatevec import CuStateVecMPIQAOASimulator

SIMULATORS = {
    "x": {
        "c": QAOAFURXSimulatorC,
        "python": QAOAFURXSimulator,
        "gpu": QAOAFURXSimulatorGPU,
        "gpumpi": QAOAFURXSimulatorGPUMPI,
    },
    "xyring": {
        "c": QAOAFURXYRingSimulatorC,
        "python": QAOAFURXYRingSimulator,
        "gpu": QAOAFURXYRingSimulatorGPU,
    },
    "xycomplete": {
        "c": QAOAFURXYCompleteSimulatorC,
        "python": QAOAFURXYCompleteSimulator,
        "gpu": QAOAFURXYCompleteSimulatorGPU,
    },
}


def get_available_simulator_names(type: str = "x") -> list:
    """
    Return names of available simulators

    Parameters
    ----------
        type: type of QAOA mixer to simulate

    Returns
    -------
        List of available simulators
    """
    family = SIMULATORS.get(type, None)
    if family is None:
        raise ValueError(f"Unknown simulator type: {type}")
    precedence = ["gpumpi", "gpu", "c", "python"]
    checks = [mpi_available, numba.cuda.is_available, c_available]
    available = []
    for i in range(len(checks)):
        if precedence[i] not in family:
            continue
        if checks[i]():
            available.append(precedence[i])
    available.append(precedence[-1])
    return available


def get_available_simulators(type: str = "x") -> list:
    """
    Return (uninitialized) classes of available simulators

    Parameters
    ----------
        type: type of QAOA mixer to simulate

    Returns
    -------
        List of available simulators
    """
    available_names = get_available_simulator_names(type=type)
    return [SIMULATORS[type][s] for s in available_names]


def choose_simulator(name="auto", **kwargs):
    if name != "auto":
        return SIMULATORS["x"][name]

    return get_available_simulators("x")[0]


def choose_simulator_xyring(name="auto", **kwargs):
    if name != "auto":
        return SIMULATORS["xyring"][name]
    return get_available_simulators("xyring")[0]


def choose_simulator_xycomplete(name="auto", **kwargs):
    if name != "auto":
        return SIMULATORS["xycomplete"][name]

    return get_available_simulators("xycomplete")[0]


__all__ = [
    "QAOAFastSimulatorBase",
    "QAOAFURXSimulatorC",
    "QAOAFURXYRingSimulatorC",
    "QAOAFURXYCompleteSimulatorC",
    "QAOAFURXSimulator",
    "QAOAFURXYRingSimulator",
    "QAOAFURXYCompleteSimulator",
    "QAOAFURXSimulatorGPU",
    "QAOAFURXYRingSimulatorGPU",
    "QAOAFURXYCompleteSimulatorGPU",
    "QAOAFURXSimulatorGPUMPI",
    "mpi_available",
    "choose_simulator",
]

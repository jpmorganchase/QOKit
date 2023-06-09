###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from docplex.mp.model_reader import ModelReader
import os
import sys
from time import time, process_time
import logging
import pandas as pd

sys.path.append("../qokit/")

from classical_methods.utils import BestBoundAborter


def run_one_instance(path: str, TTS: bool = True, n: int = 3):
    mdl = ModelReader.read(path, ignore_names=True)
    mdl.context.solver.log_output = True
    if TTS:
        df = pd.read_csv("../qokit/assets/classical_solvers/solutions.csv").set_index("N")
        LABS_solution = df.loc[n]["E"]
        mdl.add_progress_listener(BestBoundAborter(max_best_bound=LABS_solution))
    time_bis = time()
    raw_time_start = process_time()
    msol = mdl.solve()
    raw_time_diff = process_time() - raw_time_start
    time_diff = time() - time_bis
    return msol, raw_time_diff, time_diff


def run_LABS(n_range, TTS: bool = True, enable_logs: bool = True):
    if TTS:
        prefix = "TTS_"
    else:
        prefix = ""
    log_path_dir = "../logging/"
    if not os.path.isdir(log_path_dir):
        os.makedirs(log_path_dir)
    logging.basicConfig(filename=log_path_dir + f"light_time_{prefix}LABS.log", level=logging.WARNING)
    logging.warning(("n", prefix + "process_time", prefix + "clock_time"))
    for n in n_range:
        if enable_logs:
            stdoutOrigin = sys.stdout
            sys.stdout = open(log_path_dir + f"time_{prefix}LABS_n{n}.log", "w")
        path = f"../qokit/assets/lp/LABS_n{n}_cplex.lp"
        msol, raw_time_diff, time_diff = run_one_instance(path, TTS=TTS, n=n)
        logging.warning(msg=(n, raw_time_diff, time_diff))
        if enable_logs:
            sys.stdout.close()
            sys.stdout = stdoutOrigin


if __name__ == "__main__":
    """
    python3 run_cplex.py TTS n_min n_max

    Example: python3 run_cplex.py TTS 10 20
    """
    if len(sys.argv) < 4:
        raise ValueError("python3 run_cplex TTS n_min n_max")
    TTS = sys.argv[1] == "TTS"
    n_min = int(sys.argv[2])
    n_max = int(sys.argv[3])
    run_LABS(range(n_min, n_max), TTS=TTS)

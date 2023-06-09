###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import os
import sys
from gurobipy import GRB, Env, read
from gurobipy import GurobiError
from time import time, process_time
import pandas as pd
import logging

Heuristics = 0
Threads = 8
Cuts = 0
Presolve = -1


def run_one_instance(path: str, TTS: bool = True, n: int = 3, log_to_write: str = [], env=env):
    if log_to_write:
        stdoutOrigin = sys.stdout
        sys.stdout = open(log_to_write, "w")
    try:
        m = read(path, env=env)
    except GurobiError:
        print(path)
    m.setParam(GRB.Param.Cuts, Cuts)
    m.setParam(GRB.Param.Presolve, Presolve)
    m.setParam(GRB.Param.Heuristics, Heuristics)
    m.setParam(GRB.Param.Threads, Threads)
    if TTS:
        df = pd.read_csv("../qokit/assets/classical_solvers/solutions.csv").set_index("N")
        LABS_solution = df.loc[n]["E"]
        m.setParam(GRB.Param.BestObjStop, LABS_solution)
    time_bis = time()
    raw_time_start = process_time()
    msol = m.optimize()
    raw_time_diff = process_time() - raw_time_start
    time_diff = time() - time_bis
    msol = int(m.getAttr("NodeCount"))
    if log_to_write:
        sys.stdout.close()
        sys.stdout = stdoutOrigin
    return msol, raw_time_diff, time_diff


def run_LABS(n_range, TTS: bool = True, env=env):
    if TTS:
        prefix = "TTS_"
    else:
        prefix = ""
    log_path_dir = "../logs/"
    if not os.path.isdir(log_path_dir):
        os.makedirs(log_path_dir)
    main_log_file = log_path_dir + f"gurobi_time_{prefix}LABS_Threads_{Threads}_Cuts_{Cuts}_Heuristics_{Heuristics}.log"
    logging.basicConfig(filename=main_log_file, encoding="utf-8", level=logging.WARNING)
    logging.warning(("n", prefix + "process_time", prefix + "clock_time", "node_count"))

    for n in n_range:
        log_name = f"{prefix}LABS_n_{n}_Threads_{Threads}_Cuts_{Cuts}_Heuristics_{Heuristics}"
        log_to_write = log_path_dir + log_name + ".log"
        path_lp = f"../qokit/assets/lp/LABS_n{n}_cplex.lp"
        msol, raw_time_diff, time_diff = run_one_instance(path_lp, TTS=TTS, n=n, log_to_write=log_to_write)
        logging.warning(msg=(n, raw_time_diff, time_diff, msol))


if __name__ == "__main__":
    """
    python3 run_gurobi.py TTS n_min n_max

    Example: python3 run_gurobi.py TTS 10 20
    """
    if len(sys.argv) < 4:
        raise ValueError("python3 run_cplex TTS n_min n_max")
    TTS = sys.argv[1] == "TTS"
    n_min = int(sys.argv[2])
    n_max = int(sys.argv[3])
    env = Env("")  # TO COMPLETE WITH LICENSE PARAMTERS
    run_LABS(range(n_min, n_max), TTS=TTS)

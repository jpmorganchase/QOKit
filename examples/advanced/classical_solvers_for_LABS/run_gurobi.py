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
import secrets

Heuristics = 0
Threads = 8
Cuts = 0
Presolve = -1
MAXINT = 2000000000


def run_one_instance(path: str, env: Env, TTS: bool = True, n: int = 3, log_to_write: str = "", seed=None):
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
    if not seed is None:
        m.setParam(GRB.Param.Seed, seed)
    if TTS:
        df = pd.read_csv("LABS_optimal_merit_factors.csv").set_index("N")
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


def run_LABS(n_range, env: Env, TTS: bool = True, nb_runs=1):
    if TTS:
        prefix = "TTS_"
    else:
        prefix = ""
    log_path_dir = "logs_gurobi/"
    if not os.path.isdir(log_path_dir):
        os.makedirs(log_path_dir)
    main_log_file = log_path_dir + f"gurobi_time_{prefix}LABS_Threads_{Threads}_Cuts_{Cuts}_Heuristics_{Heuristics}.csv"
    with open(main_log_file, "w") as f:
        msg = f"n,{prefix}process_time,{prefix}clock_time,node_count,runid\n"
        f.write(msg)

    for n in n_range:
        for runid in range(nb_runs):
            log_name = f"{prefix}LABS_n_{n}_Threads_{Threads}_Cuts_{Cuts}_Heuristics_{Heuristics}_runid{runid}"
            log_to_write = log_path_dir + log_name + ".log"
            path_lp = f"lp/LABS_n{n}_cplex.lp"
            # Set seed and get generate seed for gurobi
            secrets.SystemRandom().seed(a=runid)
            seed_gurobi = secrets.SystemRandom().randint(0, MAXINT - 1)
            msol, raw_time_diff, time_diff = run_one_instance(path_lp, env, TTS=TTS, n=n, log_to_write=log_to_write, seed=seed_gurobi)
            with open(main_log_file, "a") as f:
                msg = f"{n},{raw_time_diff},{time_diff},{msol},{runid}\n"
                f.write(msg)


if __name__ == "__main__":
    """
    python3 run_gurobi.py TTS n_min n_max nb_runs

    TTS is a flag controlling whether Gurobi should be stopped when optimal solution
    Alternatively, if TTO is passed, Gurobi is run until the gap is closed

    n_min and n_max describe the sizes of the LABS instances. Corresponding LPs must exist in lp/ folder

    nb_runs in the number of times to run Gurobi with different initializations

    Example: python3 run_gurobi.py TTS 10 20 10
    """
    if len(sys.argv) < 5:
        raise ValueError("python3 run_cplex TTS n_min n_max nb_runs")
    TTS = sys.argv[1] == "TTS"
    n_min = int(sys.argv[2])
    n_max = int(sys.argv[3])
    nb_runs = int(sys.argv[4])
    env = Env("")  # TO COMPLETE WITH LICENSE PARAMTERS
    run_LABS(range(n_min, n_max), TTS=TTS, env=env, nb_runs=nb_runs)

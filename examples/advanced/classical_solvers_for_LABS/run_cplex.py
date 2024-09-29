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
from docplex.mp.progress import ProgressListener, ProgressClock
import secrets


# copied from qokit.classical_methods.utils
class BestBoundAborter(ProgressListener):
    """
    Custom aborter to stop when finding a feasible solution matching the bound.
    see: https://ibmdecisionoptimization.github.io/docplex-doc/mp/_modules/docplex/mp/progress.html#ProgressClock
    https://dataplatform.cloud.ibm.com/exchange/public/entry/view/6e2bffa5869dacbae6500c7037ecd36f
    """

    def __init__(self, max_best_bound=0):
        super(BestBoundAborter, self).__init__(ProgressClock.BestBound)
        self.max_best_bound = max_best_bound
        self.last_obj = None

    def notify_start(self):
        super(BestBoundAborter, self).notify_start()
        self.last_obj = None

    def notify_progress(self, pdata):
        super(BestBoundAborter, self).notify_progress(pdata)
        if pdata.has_incumbent:
            self.last_obj = pdata.current_objective
            if self.last_obj <= self.max_best_bound:
                print(f"_____ FOUND Feasible solution {self.last_obj} smaller than stopping condition {self.max_best_bound}")
                self.abort()


# In theory [0, BIGINT]
MAXINT = 2000000000


def run_one_instance(path: str, TTS: bool = True, n: int = 3, seed=None):
    mdl = ModelReader.read(path, ignore_names=True)
    mdl.context.solver.log_output = True
    if not seed is None:
        mdl.context.cplex_parameters.randomseed = seed
    if TTS:
        df = pd.read_csv("LABS_optimal_merit_factors.csv").set_index("N")
        LABS_solution = df.loc[n]["E"]
        mdl.add_progress_listener(BestBoundAborter(max_best_bound=LABS_solution))
    time_bis = time()
    raw_time_start = process_time()
    msol = mdl.solve()
    raw_time_diff = process_time() - raw_time_start
    time_diff = time() - time_bis
    return msol, raw_time_diff, time_diff


def run_LABS(n_range, TTS: bool = True, enable_logs: bool = True, nb_runs=1):
    if TTS:
        prefix = "TTS_"
    else:
        prefix = ""
    log_path_dir = "logs_cplex/"
    if not os.path.isdir(log_path_dir):
        os.makedirs(log_path_dir)
    main_log_file = log_path_dir + f"cplex_time_{prefix}LABS.csv"
    with open(main_log_file, "w") as f:
        msg = f"n,{prefix}process_time,{prefix}clock_time,runid\n"
        f.write(msg)
    for n in n_range:
        for runid in range(nb_runs):
            if enable_logs:
                stdoutOrigin = sys.stdout
                sys.stdout = open(log_path_dir + f"time_{prefix}LABS_n{n}_runid{runid}.log", "w")
            path = f"lp/LABS_n{n}_cplex.lp"
            # Set seed and get generate seed for CPLEX
            secrets.SystemRandom().seed(a=runid)
            seed_cplex = secrets.SystemRandom().randint(0, MAXINT - 1)
            msol, raw_time_diff, time_diff = run_one_instance(path, TTS=TTS, n=n, seed=seed_cplex)
            with open(main_log_file, "a") as f:
                msg = f"{n},{raw_time_diff},{time_diff},{runid}\n"
                f.write(msg)
            if enable_logs:
                sys.stdout.close()
                sys.stdout = stdoutOrigin


if __name__ == "__main__":
    """
    python3 run_cplex.py TTS n_min n_max nb_runs

    TTS is a flag controlling whether CPLEX should be stopped when optimal solution
    Alternatively, if TTO is passed, CPLEX is run until the gap is closed

    n_min and n_max describe the sizes of the LABS instances. Corresponding LPs must exist in lp/ folder

    nb_runs in the number of times to run CPLEX with different initializations

    Examples:
        python run_cplex.py TTS 10 20 10
        python run_cplex.py TTO 10 20 10
    """
    if len(sys.argv) < 5:
        raise ValueError("python3 run_cplex TTS n_min n_max nb_runs")
    TTS = sys.argv[1] == "TTS"
    n_min = int(sys.argv[2])
    n_max = int(sys.argv[3])
    nb_runs = int(sys.argv[4])
    run_LABS(range(n_min, n_max), TTS=TTS, nb_runs=nb_runs)

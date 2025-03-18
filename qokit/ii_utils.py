import os
import numpy as np
import scipy
from qokit.qaoa_objective_labs import get_qaoa_labs_objective
import pandas as pd
import sys
import matplotlib.pyplot as plt
import time
from typing import Optional, Tuple
from joblib import Parallel, delayed
import nlopt


def initialize_gpu(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def func_with_gpu_id(func):
    def wrapper(*args, gpu_id, **kwargs):
        initialize_gpu(gpu_id)
        return func(*args, **kwargs)

    return wrapper


def parallel_execution(func, inputs, num_gpus):
    func_for_gpu = func_with_gpu_id(func)
    jobs = [delayed(func_for_gpu)(*given_input, gpu_id=i % num_gpus) for i, given_input in enumerate(inputs)]
    results = Parallel(n_jobs=num_gpus, backend="loky")(jobs)
    return results


def stack_params(param1, param2):
    return np.hstack([param1, param2])


def to_basis(gamma, beta, num_coeffs, basis):
    """Convert gamma,beta angles in standard parameterizing QAOA to a basis of functions

    Parameters
    ----------
    gamma : list-like
    beta : list-like
    num_coeffs : int
    basis : string
        QAOA parameters in standard basis
    Returns
    -------
    u, v : np.array
        QAOA parameters in given basis
    """
    # assert len(gamma) == len(beta)
    try:
        p = len(gamma)
    except:
        p = 1
    fit_interval = np.linspace(-1, 1, p)

    if basis == "fourier":
        u = 2 * scipy.fft.dst(gamma, type=4, norm="forward")  # difference of 2 due to normalization of dst
        v = 2 * scipy.fft.dct(beta, type=4, norm="forward")  # difference of 2 due to normalization of dct
    elif basis == "chebyshev":
        u = np.polynomial.chebyshev.chebfit(fit_interval, gamma, deg=num_coeffs - 1)  # offset of 1 due to fitting convention
        v = np.polynomial.chebyshev.chebfit(fit_interval, beta, deg=num_coeffs - 1)
    elif basis == "hermite":
        u = np.polynomial.hermite.hermfit(fit_interval, gamma, deg=num_coeffs - 1)
        v = np.polynomial.hermite.hermfit(fit_interval, beta, deg=num_coeffs - 1)
    elif basis == "legendre":
        u = np.polynomial.legendre.legfit(fit_interval, gamma, deg=num_coeffs - 1)
        v = np.polynomial.legendre.legfit(fit_interval, beta, deg=num_coeffs - 1)
    elif basis == "laguerre":
        u = np.polynomial.laguerre.lagfit(fit_interval, gamma, deg=num_coeffs - 1)
        v = np.polynomial.laguerre.lagfit(fit_interval, beta, deg=num_coeffs - 1)

    return u, v


def from_basis(u, v, p, basis):
    """Convert u,v in a given basis of functions
    to gamma, beta angles of QAOA schedule

    Parameters
    ----------
    u : list-like
    v : list-like
    p : int
    basis : string

    Returns
    -------
    gamma, beta : np.array
        QAOA angles parameters in standard parameterization
    """
    assert len(u) == len(v)
    fit_interval = np.linspace(-1, 1, p)

    if basis == "fourier":
        if p < len(u):
            raise Exception("p must exceed the length of u and v ")

        u_padded = np.zeros(p)
        v_padded = np.zeros(p)
        u_padded[: len(u)] = u
        v_padded[: len(v)] = v
        u_padded[len(u) :] = 0
        v_padded[len(v) :] = 0

        gamma = 0.5 * scipy.fft.idst(u_padded, type=4, norm="forward")  # difference of 1/2 due to normalization of idst
        beta = 0.5 * scipy.fft.idct(v_padded, type=4, norm="forward")  # difference of 1/2 due to normalization of idct
    elif basis == "chebyshev":
        gamma = np.polynomial.chebyshev.chebval(fit_interval, u)
        beta = np.polynomial.chebyshev.chebval(fit_interval, v)
    elif basis == "hermite":
        gamma = np.polynomial.hermite.hermval(fit_interval, u)
        beta = np.polynomial.hermite.hermval(fit_interval, v)
    elif basis == "legendre":
        gamma = np.polynomial.legendre.legval(fit_interval, u)
        beta = np.polynomial.legendre.legval(fit_interval, v)
    elif basis == "laguerre":
        gamma = np.polynomial.laguerre.lagval(fit_interval, u)
        beta = np.polynomial.laguerre.lagval(fit_interval, v)

    return gamma, beta


def interpolation_basis(gamma, beta, p, num_coeffs, basis):
    """Interpolate gamma, beta to the p value using the
    given basis
    ----------
    gamma : list-like
    beta : list-like
    p : int
    num_coeffs : int
    basis : string
    Returns
    -------
    gamma, beta : np.array
        QAOA parameters in standard parameterization
    """
    u, v = to_basis(gamma, beta, num_coeffs, basis)
    gamma, beta = from_basis(u, v, p, basis)
    return gamma, beta


def optimize_bobyqa(func, params, rhobeg=None, tol=None, maxiter=3000):
    def wrapped_func(x, grad):
        if grad.size > 0:
            grad[:] = np.zeros_like(x)
        return func(x)

    opt = nlopt.opt(nlopt.LN_BOBYQA, len(params))

    if rhobeg is not None:
        opt.set_initial_step(rhobeg)
    if tol is not None:
        opt.set_ftol_rel(tol)
    if maxiter is not None:
        opt.set_maxeval(maxiter)

    opt.set_min_objective(wrapped_func)

    lower_bounds = np.full(len(params), -10)
    upper_bounds = np.full(len(params), 10)

    opt.set_lower_bounds(lower_bounds)
    opt.set_upper_bounds(upper_bounds)

    optimized_params = opt.optimize(params)
    minf = opt.last_optimum_value()
    nfev = opt.get_numevals()
    result_code = opt.last_optimize_result()

    if result_code == 1:
        message = "Optimization terminated successfully."
        success = "True"
    if result_code == 3:
        message = "Optimization stopped because ftol was reached."
        success = "True"
    elif result_code == 5:
        message = "Maximum number of function evaluations has been exceeded."
        success = "False"

    res = {
        "message": message,
        "success": success,
        "status": result_code,
        "fun": minf,
        "x": optimized_params,
        "nfev": nfev,
        "maxcv": 0.0,
    }
    return res


def approx_ratio(merit, gs_energy, max_energy=0):
    approx_ratio = (max_energy - merit) / (max_energy - gs_energy)
    # if max_energy = 0 then we get aprox_ratio = merit/gs_energy
    # which is positive as long as both merit and gs_energy are negative
    return approx_ratio


def compute_merit(approx_ratio, gs_energy, max_energy=0):
    merit = max_energy - approx_ratio * (max_energy - gs_energy)
    return merit


def fine_tune_coeffs(f_overlap, f, gs_energy, N, p, num_coeffs, gamma, beta, basis, optimizer="bobyqa", rhobeg=None, maxiter=1000, max_energy=0):
    if rhobeg is None:
        rhobeg = 0.01 / N
    print(f"Optimizing with optimizer={optimizer} for N={N}, p={p}, num_coeffs = {num_coeffs} in {basis} basis")
    u, v = to_basis(gamma, beta, num_coeffs, basis)
    initial_pt = stack_params(u[:num_coeffs], v[:num_coeffs])

    def func(ins):
        u[:num_coeffs] = ins[:num_coeffs]
        v[:num_coeffs] = ins[num_coeffs:]
        gamma, beta = from_basis(u, v, p, basis)
        merit = f(stack_params(gamma, beta))  # this value of obj is negative
        minus_approx_ratio = -1.0 * approx_ratio(merit, gs_energy, max_energy)
        return minus_approx_ratio

    if optimizer == "bobyqa":
        res = optimize_bobyqa(func, initial_pt, rhobeg=rhobeg, tol=1e-6, maxiter=maxiter)
    elif optimizer == "cobyla":
        res = scipy.optimize.minimize(func, initial_pt, method="COBYLA", options={"rhobeg": rhobeg, "tol": 1e-6, "maxiter": maxiter})
    elif optimizer == "diff_evo":
        # Here maxiter are the number of generations so there number must be set by total function evals
        # The maximum number of function evaluations (with no polishing) is: (generations + 1) * popsize * 2 * num_coeffs
        # default popsize is 15
        # Note that bound is not strict since it does take into account polishing
        popsize = 15
        generations = maxiter // (popsize * 2 * num_coeffs) - 1
        bounds = [(-0.5, 0.5)] * (2 * num_coeffs)
        res = scipy.optimize.differential_evolution(func, bounds=bounds, maxiter=generations, popsize=popsize)
    elif optimizer == "dual_anneal":
        # Here maxfun are the number of evaluations so there number must be set by total function evals
        # Note that bound is not strict since it does take into account that
        # optimization cannot be stopped during a local search
        bounds = [(-0.5, 0.5)] * (2 * num_coeffs)
        res = scipy.optimize.dual_annealing(func, bounds=bounds, maxfun=maxiter)

    print(res)
    res_gamma, res_beta = from_basis(u, v, p, basis)
    return res, res_gamma, res_beta


def fine_tune_result(f_overlap, f, gs_energy, N, p, num_coeffs, gamma, beta, basis, optimizer="bobyqa", rhobeg=None, maxiter=1000, max_energy=0):
    res, res_gamma, res_beta = fine_tune_coeffs(
        f_overlap, f, gs_energy, N, p, num_coeffs, gamma, beta, basis, optimizer=optimizer, rhobeg=rhobeg, maxiter=maxiter, max_energy=max_energy
    )
    u, v = res["x"][:num_coeffs], res["x"][num_coeffs:]

    minus_approx_ratio = res["fun"]
    approx_ratio = -1.0 * minus_approx_ratio
    merit = compute_merit(approx_ratio, gs_energy, max_energy)
    overlap = 1 - f_overlap(stack_params(res_gamma, res_beta))

    nfev = res["nfev"]
    result = {
        "N": N,
        "p": p,
        "num_coeffs": num_coeffs,
        "gamma": res_gamma,
        "beta": res_beta,
        "basis": basis,
        "u": u,
        "v": v,
        "optimizer": optimizer,
        "approx_ratio": approx_ratio,
        "merit": merit,
        "overlap": overlap,
        "evaluations": nfev,
    }
    return result


def save_result(result, filename):
    df = pd.DataFrame([result])
    np.set_printoptions(threshold=sys.maxsize)  # outputs with length > 1000 are also printed in full
    df.to_csv(filename, mode="a", header=False, index=False)


def convert_to_np_array(column):
    return column.apply(lambda x: np.fromstring(x[1:-1], dtype=np.float64, sep=" "))


def df_data_to_np_array(df):  # converts string data to numpy array data
    df_new = df.copy()
    target_columns = ["gamma", "beta", "u", "v"]
    for col in df_new.columns:
        if col in target_columns:
            try:
                df_new[col] = convert_to_np_array(df_new[col])
            except Exception as e:
                print(f"Error converting column {col}: {e}")
    return df_new


def visualize_schedule(file, save=False, p=None):
    # filename = f'II_N_{N}_pmax_{pmax}_step_{step}_optimizer_{optimizer}.csv' # modify if you want to change directory

    df = pd.read_csv(file)
    df = df_data_to_np_array(df)
    if p == None:
        row = df[(df["p"] == df["p"].max())]
    else:
        row = df[(df["p"] == p)]

    N = row.iloc[0]["N"]
    p = row.iloc[0]["p"]
    gamma = row.iloc[0]["gamma"]
    beta = row.iloc[0]["beta"]

    basis = row.iloc[0]["basis"]
    u = row.iloc[0]["u"]
    v = row.iloc[0]["v"]
    print(f"Coefficients in {basis} basis are u={u}, v={v}")

    approx_ratio = row.iloc[0]["approx_ratio"]
    overlap = row.iloc[0]["overlap"]
    print(f"Overlap: {overlap}")
    optimizer = row.iloc[0]["optimizer"]

    # Calculate the total cumulative nfev
    total_nfev = df["evaluations"].sum()
    print(f"Total cumulative nfev: {total_nfev}")

    step_arr = np.arange(0, p)
    plt.plot(step_arr, gamma, label="Gamma")
    plt.plot(step_arr, -beta / 2, label="Beta/2")
    plt.xlabel("Step")
    plt.ylabel("Angle")
    plt.legend()
    plt.title(f"II Schedule {basis} N={N} p={p} approx. {approx_ratio:.2f} opt. {optimizer}")
    if save:
        plt.savefig(f"II Schedule {basis} N={N} p={p} approx. {approx_ratio:.2f} opt. {optimizer}.pdf", bbox_inches="tight")
    plt.show()


def visualize_improvement(file, save=False):
    df = pd.read_csv(file)
    row = df[(df["p"] == df["p"].max())]

    N = row.iloc[0]["N"]
    p = row.iloc[0]["p"]
    p_arr = df["p"]

    approx_ratio = df["approx_ratio"]
    overlap = df["overlap"]
    optimizer = row.iloc[0]["optimizer"]

    # Calculate the total cumulative nfev
    total_nfev = df["evaluations"].sum()
    print(f"Total cumulative nfev: {total_nfev}")

    plt.plot(p_arr, approx_ratio, label="Approx Ratio")
    plt.plot(p_arr, overlap, label="Overlap")
    plt.xlabel("p")
    plt.ylabel("Objective")
    plt.legend()
    plt.title(f"II Schedule Approx Ratio and Overlap for N={N} p={p} opt. {optimizer}")
    if save:
        plt.savefig(f"II Schedule Approx Ratio and Overlap for N={N} p={p} opt. {optimizer}.pdf", bbox_inches="tight")
    plt.show()


def initialize_csv(filename: str):
    columns = ["N", "p", "num_coeffs", "gamma", "beta", "basis", "u", "v", "optimizer", "approx_ratio", "merit", "overlap", "evaluations"]
    df = pd.DataFrame(columns=columns)
    df.to_csv(filename, index=False)


def generate_p_values(p0: int, pmax: int, step: int) -> np.ndarray:
    if step > p0:
        return np.concatenate(([p0], np.arange(step, pmax + 1, step, dtype=int)))
    return np.arange(p0, pmax + 1, step, dtype=int)


def generate_p_values_without_step(pmax: int) -> np.ndarray:
    p_values = []
    p0 = 10
    current_p = p0

    # # Increase by 1 from 5 to 10
    # while current_p < 10 and current_p <= pmax:
    #     p_values.append(current_p)
    #     current_p += 1

    # Increase by 2 until 20
    while current_p < 20 and current_p <= pmax:
        p_values.append(current_p)
        current_p += 2

    # Increase by 5 until 100
    while current_p < 200 and current_p <= pmax:
        p_values.append(current_p)
        current_p += 5

    # Increase by 10 thereafter
    while current_p <= pmax:
        p_values.append(current_p)
        current_p += 10

    return np.array(p_values, dtype=int)


def log_walltime(start_time: float):
    walltime_seconds = time.time() - start_time
    walltime_hours = int(walltime_seconds // 3600)
    walltime_minutes = (walltime_seconds % 3600) / 60
    print(f"Walltime: {walltime_hours} hrs and {walltime_minutes:.2f} minutes")

import math
import json
import pickle
from pathlib import Path
from typing import Tuple
import numpy as np
import networkx as nx

import qokit
from qokit.parameter_utils import get_fixed_gamma_beta
from qokit.maxcut import get_maxcut_terms, get_adjacency_matrix

import cvxpy as cp


class WSSolver:
    def __init__(self, graph: nx.Graph, graph_degree: int = 3, graph_seed: int = 1):
        self.G = graph
        self.graph_degree = graph_degree
        self.graph_seed = graph_seed
        self.best_cut = None
        self.p = None
        self.gamma = None
        self.beta = None

        self.nodes = list(graph.nodes())
        self.n_v = len(self.nodes)

    ########################### Objective and gradient ###########################
    ##############################################################################
    def get_p0_cut(self, theta=None):
        if theta is None:
            theta = self.theta
        f_value = 0
        for i, j in self.G.edges():
            f_value += 0.5 * (1 - np.cos(theta[i]) * np.cos(theta[j]))

        return f_value

    def get_p0_std_quantities(self, theta: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Weighted Max-Cut warm-start statistics on a NetworkX graph.

        Args
        ----
        G     : undirected nx.Graph with edge attribute 'weight' (default 1.0)
        theta : (n,) array of angles in radians, aligned with list(G.nodes())

        Returns
        -------
        exp_H : float        = <H>
        var_H : float        = <H^2> - <H>^2
        gE    : (n,) ndarray = d<H>/d theta
        gV    : (n,) ndarray = dVar(H)/d theta
        """

        G = self.G
        nodes = self.nodes
        n = self.n_v
        # Build edge arrays

        i_list, j_list, w_list = [], [], []
        for u, v, d in G.edges(data=True):
            i_list.append(u)
            j_list.append(v)
            w_list.append(float(d.get("weight", 1.0)))

        i = np.asarray(i_list, dtype=int)
        j = np.asarray(j_list, dtype=int)
        w = np.asarray(w_list, dtype=float)

        w2 = w * w
        c = np.cos(theta)  # c_i
        s = np.sin(theta)  # s_i
        ci, cj = c[i], c[j]

        # Aggregates:
        # S_k = sum_{nbr v} w_kv * c_v
        # Q_k = sum_{nbr v} w_kv^2 * c_v^2
        S = np.zeros(n, dtype=float)
        Q = np.zeros(n, dtype=float)
        np.add.at(S, i, w * cj)
        np.add.at(S, j, w * ci)
        np.add.at(Q, i, w2 * (cj * cj))
        np.add.at(Q, j, w2 * (ci * ci))

        # <H>
        exp_H = 0.5 * np.sum(w * (1.0 - ci * cj))

        # Var(H) = edge term + vertex term
        term_edges = 0.25 * np.sum(w2 * (1.0 - (ci * ci) * (cj * cj)))
        term_vertices = 0.25 * np.sum((1.0 - c * c) * (S * S - Q))
        var_H = term_edges + term_vertices

        # Grad <H>
        gE = 0.5 * s * S

        # Grad Var(H)
        extra = np.zeros(n, dtype=float)
        np.add.at(extra, i, (1.0 - cj * cj) * (-w * S[j] + w2 * c[i]))
        np.add.at(extra, j, (1.0 - ci * ci) * (-w * S[i] + w2 * c[j]))
        gV = 0.5 * s * (c * (S * S) + extra)

        return float(exp_H), float(var_H), gE, gV

    def get_p0_std_quantities_batch(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized version: theta is (batch_size, n)
        Returns:
            exp_H: (batch_size,)
            var_H: (batch_size,)
            gE:    (batch_size, n)
            gV:    (batch_size, n)
        """
        G = self.G
        n = self.n_v

        # Build edge arrays (same as before)
        i_list, j_list, w_list = [], [], []
        for u, v, d in G.edges(data=True):
            i_list.append(u)
            j_list.append(v)
            w_list.append(float(d.get("weight", 1.0)))
        i = np.asarray(i_list, dtype=int)
        j = np.asarray(j_list, dtype=int)
        w = np.asarray(w_list, dtype=float)
        w2 = w * w

        batch_size = theta.shape[0]
        c = np.cos(theta)  # (batch_size, n)
        s = np.sin(theta)  # (batch_size, n)
        ci = c[:, i]  # (batch_size, num_edges)
        cj = c[:, j]  # (batch_size, num_edges)

        # Aggregates
        S = np.zeros((batch_size, n), dtype=float)
        Q = np.zeros((batch_size, n), dtype=float)
        # For each batch, scatter-add
        np.add.at(S, (slice(None), i), w * cj)
        np.add.at(S, (slice(None), j), w * ci)
        np.add.at(Q, (slice(None), i), w2 * (cj * cj))
        np.add.at(Q, (slice(None), j), w2 * (ci * ci))

        # <H>
        exp_H = 0.5 * np.sum(w * (1.0 - ci * cj), axis=1)  # (batch_size,)

        # Var(H)
        term_edges = 0.25 * np.sum(w2 * (1.0 - (ci * ci) * (cj * cj)), axis=1)
        term_vertices = 0.25 * np.sum((1.0 - c * c) * (S * S - Q), axis=1)
        var_H = term_edges + term_vertices  # (batch_size,)

        # Grad <H>
        gE = 0.5 * s * S  # (batch_size, n)

        # Grad Var(H)
        extra = np.zeros((batch_size, n), dtype=float)
        # S[:, j] and S[:, i] are (batch_size, num_edges)
        np.add.at(extra, (slice(None), i), (1.0 - cj * cj) * (-w * S[:, j] + w2 * c[:, i]))
        np.add.at(extra, (slice(None), j), (1.0 - ci * ci) * (-w * S[:, i] + w2 * c[:, j]))
        gV = 0.5 * s * (c * (S * S) + extra)  # (batch_size, n)

        return exp_H, var_H, gE, gV

    def p0_theta_objective(self, theta: np.ndarray, lamd=None) -> Tuple[float, np.ndarray, dict]:
        # TODO: simplify
        """
        Compute the normalized risk-adjusted objective
            J(theta) =  <H>  + lam *  2*sqrt(Var)
        and its gradient with respect to theta.

        Returns
        -------
        J      : float
        gradJ  : (n,) ndarray
        info   : dict with exp_H, var_H, mu_bar, sig_bar, gE, gV
        """
        G = self.G
        if lamd is None:
            try:
                lamd = self.lamd
            except:
                lamd = 0
        exp_H, var_H, gE, gV = self.get_p0_std_quantities(theta)

        # Normalized mean and std-like measure
        eps = 1e-12  # Guard to avoid division by zero
        mu_bar = exp_H
        sig_bar = np.sqrt(var_H + eps)

        # Objective
        J = mu_bar + lamd * np.sum(np.sin(theta) ** 2)

        # Gradients
        grad_mu_bar = gE
        grad_sig_bar = gV / (2 * np.sqrt(var_H + eps))

        gradJ = grad_mu_bar + lamd * np.sum(np.sin(2 * theta))

        info = dict(exp_H=exp_H, var_H=var_H, mu_bar=mu_bar, sig_bar=sig_bar, gE=gE, gV=gV)
        return float(J)

    def p0_theta_objective_batch(self, theta: np.ndarray, lamd=None) -> Tuple[np.ndarray, dict]:
        """
        Vectorized: theta is (batch_size, n)
        Returns:
            J: (batch_size,)
            info: dict of arrays, each (batch_size, ...) as appropriate
        """
        if lamd is None:
            try:
                lamd = self.lamd
            except:
                lamd = 0

        exp_H, var_H, gE, gV = self.get_p0_std_quantities_batch(theta)
        eps = 1e-12
        mu_bar = exp_H
        sig_bar = np.sqrt(var_H + eps)

        # Objective: J = mu_bar + lamd * sum_i sin^2(theta_i)
        J = mu_bar + lamd * np.sum(np.sin(theta) ** 2, axis=1)  # (batch_size,)

        return J

    def p0_theta_grad(self, theta: np.ndarray, lamd=None) -> Tuple[float, np.ndarray, dict]:
        # TODO: simplify
        """
        Compute the normalized risk-adjusted objective
            J(theta) =  <H>  + lam *  2*sqrt(Var)
        and its gradient with respect to theta.

        Returns
        -------
        J      : float
        gradJ  : (n,) ndarray
        info   : dict with exp_H, var_H, mu_bar, sig_bar, gE, gV
        """
        G = self.G
        if lamd is None:
            try:
                lamd = self.lamd
            except:
                lamd = 0

        exp_H, var_H, gE, gV = self.get_p0_std_quantities(theta)

        # Normalized mean and std-like measure
        eps = 1e-12  # Guard to avoid division by zero
        mu_bar = exp_H
        sig_bar = np.sqrt(var_H + eps)

        # Objective
        J = mu_bar + lamd * np.sum(np.sin(theta) ** 2)

        # Gradients
        grad_mu_bar = gE
        grad_sig_bar = gV / (2 * np.sqrt(var_H + eps))

        gradJ = grad_mu_bar + lamd * np.sin(2 * theta)

        info = dict(exp_H=exp_H, var_H=var_H, mu_bar=mu_bar, sig_bar=sig_bar, gE=gE, gV=gV)
        return gradJ

    def p0_theta_grad_batch(self, theta: np.ndarray, lamd=None) -> np.ndarray:
        """
        Vectorized: theta is (batch_size, n)
        Returns:
            gradJ: (batch_size, n)
        """
        if lamd is None:
            try:
                lamd = self.lamd
            except:
                lamd = 0

        exp_H, var_H, gE, gV = self.get_p0_std_quantities_batch(theta)
        eps = 1e-12

        grad_mu_bar = gE  # (batch_size, n)
        gradJ = grad_mu_bar + lamd * np.sin(2 * theta)  # (batch_size, n)

        return gradJ

    def bm_gradient(self, theta, lamd=None):
        """Compute the gradient at each node for the objective
        f(theta) = sum_{(i,j) in E} w_ij * (1 - cos(theta_i - theta_j))
        Derivative with respect to theta_i is:
            grad_i = sum_{j in N(i)} w_ij * sin(theta_i - theta_j)
        """
        G = self.G
        if lamd is None:
            try:
                lamd = self.lamd
            except:
                lamd = 0
        grad = np.zeros(self.n_v)
        # Map node index -> theta index
        nodes = list(G.nodes())
        # node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        for i in nodes:
            # i_idx = node_to_idx[i]
            i_idx = i
            for j in G.neighbors(i):
                # j_idx = node_to_idx[j]
                j_idx = j
                # get edge weight; default to 1 if not present
                w_ij = G[i][j].get("weight", 1.0)
                grad[i_idx] += w_ij * np.sin(theta[i_idx] - theta[j_idx])
                grad[i_idx] += lamd * w_ij * np.sin(theta[i_idx]) * np.cos(theta[j_idx]) / 2
        return grad

    def bm_objective(self, theta, lamd=None):
        """Compute the objective function value:
            f(theta) = sum_{(i,j) in E} w_ij*(1 - cos(theta_i - theta_j))
        Note: This objective is proportional to the SDP objective (ignoring constant factors).
        """
        G = self.G
        if lamd is None:
            try:
                lamd = self.lamd
            except:
                lamd = 0
        nodes = self.nodes
        val = 0.0
        seen = set()
        for i in nodes:
            for j in G.neighbors(i):
                # Avoid double counting edges for undirected graph
                if (j, i) in seen:
                    continue
                w_ij = G[i][j].get("weight", 1.0)
                diff = theta[i] - theta[j]
                val += w_ij * (1 - math.cos(diff)) / 2
                val += lamd * w_ij * (1 - np.cos(theta[i]) * np.cos(theta[j])) / 2
                seen.add((i, j))
        return val

    def bm_gradient_batch(self, theta, lamd=None):
        """
        Vectorized: theta is (batch_size, n)
        Returns: grad (batch_size, n)
        """
        if lamd is None:
            try:
                lamd = self.lamd
            except:
                lamd = 0
        batch_size, n = theta.shape

        # Weighted adjacency matrix
        W = np.zeros((n, n))
        seen = set()
        for i in self.nodes:
            for j in self.G.neighbors(i):
                # Avoid double counting edges for undirected graph
                if (j, i) in seen:
                    continue
                W[i][j] = self.G[i][j].get("weight", 1.0)

        S = np.sin(theta)  # (batch_size, n)
        C = np.cos(theta)  # (batch_size, n)

        grad1 = S * (C @ W.T) - C * (S @ W.T)  # (batch_size, n)

        # Second term: lamd * w_ij * sin(theta_i) * cos(theta_j) / 2
        grad2 = lamd * S * (C @ W.T) / 2  # (batch_size, n)

        grad = grad1 + grad2
        return grad

    def bm_objective_batch(self, theta, lamd=None):
        """
        Vectorized: theta is (batch_size, n)
        Returns: val (batch_size,)
        """
        if lamd is None:
            try:
                lamd = self.lamd
            except:
                lamd = 0

        i_list, j_list, w_list = [], [], []
        for u, v, d in self.G.edges(data=True):
            i_list.append(u)
            j_list.append(v)
            w_list.append(float(d.get("weight", 1.0)))
        i = np.asarray(i_list, dtype=int)
        j = np.asarray(j_list, dtype=int)
        w = np.asarray(w_list, dtype=float)

        batch_size, n = theta.shape

        diff = theta[:, i] - theta[:, j]  # (batch_size, num_edges)
        term1 = w * (1 - np.cos(diff)) / 2  # (num_edges,) broadcasted
        term2 = lamd * w * (1 - np.cos(theta[:, i]) * np.cos(theta[:, j])) / 2

        val = np.sum(term1 + term2, axis=1)  # (batch_size,)
        return val

    def abid_objective(self, theta, local_bitstring, lamd=None):
        # given, a bitstring to start with
        # l = b^T E(x) + lamda \sum_i sin^2(theta_i)
        # l = \sum_i b_i sin^2(\theta_i/2) + lamda \sum_i sin^2(theta_i)
        if lamd is None:
            try:
                lamd = self.lamd
            except:
                lamd = 0
        epsilon = 1e-12
        y = local_bitstring * np.log(np.sin(theta / 2) ** 2 + epsilon) + (1 - local_bitstring) * np.log(np.cos(theta / 2) ** 2 + epsilon)
        term1 = np.sum(y) / self.n_v
        term2 = lamd / self.n_v * np.sum(np.sin(theta) ** 2)

        val = term1 + term2
        return val

    def abid_gradient(self, theta, local_bitstring, lamd=None):
        if lamd is None:
            try:
                lamd = self.lamd
            except:
                lamd = 0
        epsilon = 1e-12
        grad_term1 = (-local_bitstring / np.tan(theta / 2 + epsilon) + (1 - local_bitstring) * np.tan(theta / 2 + epsilon)) / -self.n_v
        # Term 2 gradient
        grad_term2 = lamd / self.n_v * np.sin(2 * theta)
        grad = grad_term1 + grad_term2

        return grad

    def abid_objective_batch(self, theta, local_bitstring, lamd=None):
        if lamd is None:
            try:
                lamd = self.lamd
            except:
                lamd = 0
        epsilon = 1e-12
        # y: (batch_size, n)
        y = local_bitstring * np.log(np.sin(theta / 2) ** 2 + epsilon) + (1 - local_bitstring) * np.log(np.cos(theta / 2) ** 2 + epsilon)
        # term1: (batch_size,)
        term1 = np.sum(y, axis=1) / self.n_v
        # term2: (batch_size,)
        term2 = lamd / self.n_v * np.sum(np.sin(theta) ** 2, axis=1)
        val = term1 + term2
        return val  # shape: (batch_size,)

    def abid_gradient_batch(self, theta, local_bitstring, lamd=None):
        if lamd is None:
            try:
                lamd = self.lamd
            except:
                lamd = 0

        epsilon = 1e-12
        # grad_term1: (batch_size, n)
        grad_term1 = (-local_bitstring / np.tan(theta / 2 + epsilon) + (1 - local_bitstring) * np.tan(theta / 2 + epsilon)) / -self.n_v
        # grad_term2: (batch_size, n)
        grad_term2 = lamd / self.n_v * np.sin(2 * theta)
        grad = grad_term1 + grad_term2
        return grad  # shape: (batch_size, n)

    ########################### GW warm start ###########################
    #####################################################################
    def get_GW_projection(self):
        adj_matrix = get_adjacency_matrix(self.G)

        # Define the semidefinite variable
        X = cp.Variable((self.n_v, self.n_v), symmetric=True)

        # Define the constraints
        constraints = [X >> 0]  # X is positive semidefinite
        constraints += [cp.diag(X) == 1]  # Diagonal elements are 1

        # Define the objective function
        objective = cp.Maximize(cp.sum(cp.multiply(adj_matrix, 0.25 * (1 - X))))

        # Solve the semidefinite program
        prob = cp.Problem(objective, constraints)
        prob.solve()

        # Extract the solution
        X_opt = X.value

        eigvals, eigvecs = np.linalg.eigh(X_opt)
        eigvals_clipped = np.clip(eigvals, 0, None)
        reconstructed_V = eigvecs @ np.diag(np.sqrt(eigvals_clipped))

        return reconstructed_V

    ########################### QAOA Simulation ###########################
    #######################################################################
    def get_qaoa_para(self, p: int = 11):
        if p <= 11:
            gamma, beta = get_fixed_gamma_beta(self.graph_degree, p)
        else:
            assert np.isclose(self.graph_degree, 3)
            with open("../maxcut_mps/data_max_cut_qaoa_d2.pkl", "rb") as file:
                data = pickle.load(file)
            gamma = np.asarray(data[data["p"] == p]["gammas"].values[0]) * 4
            beta = np.asarray(data[data["p"] == p]["betas"].values[0])
        return gamma, beta

    def run_ws_qaoa(
        self,
        p: int = 11,
        gamma: np.ndarray = None,
        beta: np.ndarray = None,
        theta: np.ndarray = None,
        result_folder: str = "result_ws_p0_std",
        check_saved_result: bool = True,
    ):
        ##################################################
        if check_saved_result:
            n_v = self.n_v
            if n_v <= 30:
                if self.objective == "GW":
                    result_name = f"GW_N{self.n_v}_d{self.graph_degree}_seed{self.graph_seed}_p{p}.json"
                elif self.objective in ["BM", "p0_std", "p0_theta", "p0_std_global", "p0_theta_global", "BM_global", "abid", "abid_global"]:
                    if self.scale in ["sqrtn", "sqrte"]:
                        result_name = f"N{self.n_v}_d{self.graph_degree}_seed{self.graph_seed}_{self.scale}lamd{self.lamd}_p{p}_multistart{self.trials}.json"
                    else:
                        result_name = f"N{self.n_v}_d{self.graph_degree}_seed{self.graph_seed}_lamd{self.lamd}_p{p}_multistart{self.trials}.json"
            else:
                if self.objective == "GW":
                    result_name = f"GW_N{self.n_v}_d{self.graph_degree}_seed{self.graph_seed}.json"
                elif self.objective in ["BM", "p0_std", "p0_theta", "p0_std_global", "p0_theta_global", "BM_global", "abid", "abid_global"]:
                    if self.scale in ["sqrtn", "sqrte"]:
                        result_name = f"N{self.n_v}_d{self.graph_degree}_seed{self.graph_seed}_{self.scale}lamd{self.lamd}_multistart{self.trials}.json"
                    else:
                        result_name = f"N{self.n_v}_d{self.graph_degree}_seed{self.graph_seed}_lamd{self.lamd}_multistart{self.trials}.json"

            result_path = f"{result_folder}/N{self.n_v}_d{self.graph_degree}/seed{self.graph_seed}/{result_name}"
            if Path(result_path).exists():
                print(f"Found ws results at {result_path}, skipping")
                with open(result_path, "rb") as json_file:
                    loaded_ws_data = json.load(json_file)
                    self.p = loaded_ws_data["p"]
                    self.gamma = loaded_ws_data["gamma"]
                    self.beta = loaded_ws_data["beta"]
                    self.ws_qaoa_energy = loaded_ws_data["ws_energy"]
                    return loaded_ws_data["ws_energy"]
        ##################################################
        assert self.n_v <= 30
        if theta is None:
            theta = self.theta
        if (gamma is None) and (beta is None):
            gamma, beta = self.get_qaoa_para(p)
        simclass = qokit.fur.choose_simulator_xz(name="auto")
        terms = get_maxcut_terms(self.G)
        sim = simclass(self.n_v, terms=terms)
        cost = sim.get_cost_diagonal()
        best_cut = np.max(cost)
        _result = sim.simulate_ws_qaoa(list(np.asarray(gamma)), list(np.asarray(beta)), np.asarray(theta))
        c_energy = sim.get_expectation(_result)
        self.best_cut = best_cut
        if p is None:
            self.p = len(gamma)
        else:
            self.p = p
        self.gamma = gamma
        self.beta = beta
        self.ws_qaoa_energy = c_energy
        return c_energy

    def run_standard_qaoa(self, p: int = 11, result_folder: str = "result_qaoa", check_saved_result: bool = True):
        assert self.n_v <= 30
        ##################################################
        if check_saved_result:
            n_v = self.n_v
            result_name = f"QAOA_N{self.n_v}_d{self.graph_degree}_seed{self.graph_seed}_p{p}.json"
            result_path = f"{result_folder}/N{self.n_v}_d{self.graph_degree}/seed{self.graph_seed}/{result_name}"
            if Path(result_path).exists():
                print(f"Found ws results at {result_path}, skipping")
                with open(result_path, "rb") as json_file:
                    loaded_qaoa_data = json.load(json_file)
                    self.best_cut = loaded_qaoa_data["best_cut"]
                    self.p = loaded_qaoa_data["p"]
                    self.gamma = loaded_qaoa_data["gamma"]
                    self.beta = loaded_qaoa_data["beta"]
                    self.qaoa_energy = loaded_qaoa_data["qaoa_energy"]
                    return loaded_qaoa_data["qaoa_energy"]

        ##################################################
        gamma, beta = self.get_qaoa_para(p)

        simclass = qokit.fur.choose_simulator(name="auto")
        terms = get_maxcut_terms(self.G)
        sim = simclass(self.n_v, terms=terms)
        cost = sim.get_cost_diagonal()
        best_cut = np.max(cost)
        _result = sim.simulate_qaoa(list(np.asarray(gamma)), list(np.asarray(beta)))
        c_energy = sim.get_expectation(_result)
        self.best_cut = best_cut
        self.p = p
        self.gamma = gamma
        self.beta = beta
        self.qaoa_energy = c_energy
        return c_energy


class WSSolverQUBO:
    def __init__(self, Q, **kwargs):
        self.Q = Q
        self.n_v = len(Q)

    def run_standard_qaoa(self, gamma: np.ndarray = None, beta: np.ndarray = None, result_folder: str = "result_qaoa", check_saved_result: bool = True):
        assert self.n_v <= 30

        simclass = qokit.fur.choose_simulator(name="c")
        terms = get_terms_from_QUBO(self.Q)
        sim = simclass(self.n_v, terms=terms)
        cost = sim.get_cost_diagonal()
        best_cut = np.max(cost)
        _result = sim.simulate_qaoa(list(np.asarray(gamma)), list(np.asarray(beta)))
        c_energy = sim.get_expectation(_result)
        self.best_cut = best_cut
        # self.p = p
        self.gamma = gamma
        self.beta = beta
        self.qaoa_energy = c_energy
        return c_energy


def maxcut_qubo_from_G(G):
    """
    Build a minimization QUBO x^T Q x equivalent to maximizing the Max-Cut weight
    on a (possibly weighted) undirected graph with adjacency matrix A.

    Parameters
    ----------
    A : (n, n) array_like
        Symmetric adjacency (weights >= 0). Diagonal should be zero.

    Returns
    -------
    Q : (n, n) np.ndarray
        Symmetric QUBO matrix such that minimizing x^T Q x is equivalent to
        maximizing the cut value. The linear term has been absorbed into the diagonal.
    """
    # Off-diagonal Q entries: Q_ij = w_ij so that 2*Q_ij = 2*w_ij in x^T Q x
    A = get_adjacency_matrix(G)
    Q = A.copy()

    # Linear term coefficients: c_i = -sum_j w_ij  (negative weighted degree)
    c = -A.sum(axis=1)

    # Absorb linear term into the diagonal: Q' = Q + diag(c) (since x_i^2 = x_i)
    np.fill_diagonal(Q, np.diag(Q) + c)

    Q = -Q  # sign adjustment
    return Q


def qubo_to_z_hamiltonian(Q):
    """
    Map QUBO x^T Q x to H = const*I + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j
    using x_i = (1 - Z_i)/2. Returns (const, h, J) with J symmetric, diag(J)=0.
    """

    Q = 0.5 * (Q + Q.T)  # symmetrize
    n = Q.shape[0]
    trQ = np.trace(Q)
    const = 0.25 * (Q.sum() + trQ)  # shift; ignorable for optimization
    h = -0.5 * Q.sum(axis=1)  # local fields
    J = 0.5 * (Q - np.diag(np.diag(Q)))  # pair couplings, zero diagonal
    return float(const), h, J


def get_terms_from_QUBO(Q):
    n = len(Q)
    constant, h, J = qubo_to_z_hamiltonian(Q)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            if not np.isclose(J[i, j], 0):
                terms += [(J[i, j], (i, j))]

    for i in range(n):
        if not np.isclose(h[i], 0):
            terms += [(h[i], (i,))]
    terms.append((+constant, tuple()))
    return terms

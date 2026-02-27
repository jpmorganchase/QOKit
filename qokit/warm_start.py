import math
import numpy as np
import networkx as nx
import scipy.optimize

import qokit
from qokit.parameter_utils import get_fixed_gamma_beta
from qokit.maxcut import get_maxcut_terms, get_adjacency_matrix


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

    def get_p0_std_quantities(
        self, theta: np.ndarray
    ):
        """
        Weighted Max-Cut warm-start statistics on a NetworkX graph.

        Parameters
        ----------
        theta : (n,) array of angles in radians, indexed by node label

        Returns
        -------
        exp_H : float        = <H>
        var_H : float        = <H^2> - <H>^2
        gE    : (n,) ndarray = d<H>/d theta
        gV    : (n,) ndarray = dVar(H)/d theta
        """

        G = self.G
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

    def rws_objective(self, theta: np.ndarray, lamd=None):
        """
        Compute the regularized objective
            J(theta) = <H> + lam * sum_i sin^2(theta_i)

        Parameters
        ----------
        theta : (n,) or (batch_size, n) ndarray

        Returns
        -------
        J : float (single) or (batch_size,) ndarray (batch)
        """
        if lamd is None:
            try:
                lamd = self.lamd
            except AttributeError:
                lamd = 0

        if theta.ndim == 1:
            exp_H = self.get_p0_cut(theta)
            J = exp_H + lamd * np.sum(np.sin(theta) ** 2)
            return float(J)

        # batch: theta is (batch_size, n)
        i_list, j_list, w_list = [], [], []
        for u, v, d in self.G.edges(data=True):
            i_list.append(u)
            j_list.append(v)
            w_list.append(float(d.get("weight", 1.0)))
        i = np.asarray(i_list, dtype=int)
        j = np.asarray(j_list, dtype=int)
        w = np.asarray(w_list, dtype=float)

        c = np.cos(theta)
        ci, cj = c[:, i], c[:, j]
        exp_H = 0.5 * np.sum(w * (1.0 - ci * cj), axis=1)

        J = exp_H + lamd * np.sum(np.sin(theta) ** 2, axis=1)
        return J

    def rws_grad(self, theta: np.ndarray, lamd=None) -> np.ndarray:
        """
        Compute the gradient of the regularized objective
            J(theta) = <H> + lam * sum_i sin^2(theta_i)

        Parameters
        ----------
        theta : (n,) or (batch_size, n) ndarray

        Returns
        -------
        gradJ : (n,) or (batch_size, n) ndarray
        """
        if lamd is None:
            try:
                lamd = self.lamd
            except AttributeError:
                lamd = 0

        n = self.n_v
        i_list, j_list, w_list = [], [], []
        for u, v, d in self.G.edges(data=True):
            i_list.append(u)
            j_list.append(v)
            w_list.append(float(d.get("weight", 1.0)))
        i = np.asarray(i_list, dtype=int)
        j = np.asarray(j_list, dtype=int)
        w = np.asarray(w_list, dtype=float)

        if theta.ndim == 1:
            c = np.cos(theta)
            s = np.sin(theta)
            S = np.zeros(n, dtype=float)
            np.add.at(S, i, w * c[j])
            np.add.at(S, j, w * c[i])
            gE = 0.5 * s * S
            return gE + lamd * np.sin(2 * theta)

        # batch: theta is (batch_size, n)
        batch_size = theta.shape[0]
        c = np.cos(theta)
        s = np.sin(theta)
        S = np.zeros((batch_size, n), dtype=float)
        np.add.at(S, (slice(None), i), w * c[:, j])
        np.add.at(S, (slice(None), j), w * c[:, i])
        gE = 0.5 * s * S
        return gE + lamd * np.sin(2 * theta)

    def bm_gradient(self, theta, lamd=None):
        """Compute the gradient for the BM objective.

        Parameters
        ----------
        theta : (n,) or (batch_size, n) ndarray

        Returns
        -------
        grad : (n,) or (batch_size, n) ndarray
        """
        if lamd is None:
            try:
                lamd = self.lamd
            except AttributeError:
                lamd = 0

        if theta.ndim == 1:
            grad = np.zeros(self.n_v)
            for i in list(self.G.nodes()):
                for j in self.G.neighbors(i):
                    w_ij = self.G[i][j].get("weight", 1.0)
                    grad[i] += w_ij * np.sin(theta[i] - theta[j])
                    grad[i] += lamd * w_ij * np.sin(theta[i]) * np.cos(theta[j]) / 2
            return grad

        # batch: theta is (batch_size, n)
        batch_size, n = theta.shape
        W = nx.to_numpy_array(self.G, nodelist=range(n))

        S = np.sin(theta)
        C = np.cos(theta)
        grad1 = S * (C @ W.T) - C * (S @ W.T)
        grad2 = lamd * S * (C @ W.T) / 2
        return grad1 + grad2

    def bm_objective(self, theta, lamd=None):
        """Compute the BM objective function value.

        Parameters
        ----------
        theta : (n,) or (batch_size, n) ndarray

        Returns
        -------
        val : float (single) or (batch_size,) ndarray (batch)
        """
        if lamd is None:
            try:
                lamd = self.lamd
            except AttributeError:
                lamd = 0

        if theta.ndim == 1:
            val = 0.0
            for u, v, d in self.G.edges(data=True):
                w_ij = d.get("weight", 1.0)
                diff = theta[u] - theta[v]
                val += w_ij * (1 - math.cos(diff)) / 2
                val += lamd * w_ij * (1 - np.cos(theta[u]) * np.cos(theta[v])) / 2
            return val

        # batch: theta is (batch_size, n)
        i_list, j_list, w_list = [], [], []
        for u, v, d in self.G.edges(data=True):
            i_list.append(u)
            j_list.append(v)
            w_list.append(float(d.get("weight", 1.0)))
        i = np.asarray(i_list, dtype=int)
        j = np.asarray(j_list, dtype=int)
        w = np.asarray(w_list, dtype=float)

        diff = theta[:, i] - theta[:, j]
        term1 = w * (1 - np.cos(diff)) / 2
        term2 = lamd * w * (1 - np.cos(theta[:, i]) * np.cos(theta[:, j])) / 2
        return np.sum(term1 + term2, axis=1)

    ########################### Optimizer ###########################
    #################################################################
    def theta_sdg(
        self, objective='rws', iterations=300, learning_rate=0.01, seed=0, lamd=None,
        ini_theta=None, threshold=1e-8, batch_size=None
    ):
        n = self.n_v
        if lamd is None:
            lamd = self.lamd
        np.random.seed(seed)

        is_batch = batch_size is not None

        if objective == 'BM':
            grad_func = self.bm_gradient
            obj_func = self.bm_objective
        elif objective == 'rws':
            grad_func = self.rws_grad
            obj_func = self.rws_objective
        else:
            raise ValueError(f"Unsupported objective: {objective}")

        if is_batch:
            if ini_theta is None:
                theta = np.random.uniform(0, 2 * np.pi, size=(batch_size, n))
            else:
                theta = ini_theta  # shape: (batch_size, n)

            converged = np.zeros(batch_size, dtype=bool)
        else:
            if ini_theta is None:
                theta = np.random.uniform(0, 2 * np.pi, size=n)
            else:
                theta = ini_theta

            converged = False

        for it in range(iterations):
            grad_theta = grad_func(theta)

            if is_batch:
                update_mask = ~converged
                theta[update_mask] += learning_rate * grad_theta[update_mask]
                theta[update_mask] = np.mod(theta[update_mask], 2 * np.pi)

                grad_norm_sq = np.linalg.norm(grad_theta, axis=1) ** 2 / n
                converged = converged | (grad_norm_sq < threshold)
                if np.all(converged):
                    break
            else:
                theta += learning_rate * grad_theta
                theta = np.mod(theta, 2 * np.pi)

                grad_norm_sq = np.linalg.norm(grad_theta) ** 2 / n
                if grad_norm_sq < threshold:
                    converged = True
                    break

        ws_obj = obj_func(theta)
        return ws_obj, theta, converged

    def theta_adam(
        self, objective='rws', iterations=300, learning_rate=0.01, seed=0, lamd=None, ini_theta=None,
        beta1=0.9, beta2=0.999, threshold=1e-8, batch_size=None
    ):
        n = self.n_v
        if lamd is None:
            lamd = self.lamd
        np.random.seed(seed)

        is_batch = batch_size is not None

        if objective == 'BM':
            grad_func = self.bm_gradient
            obj_func = self.bm_objective
        elif objective == 'rws':
            grad_func = self.rws_grad
            obj_func = self.rws_objective
        else:
            raise ValueError(f"Unsupported objective: {objective}")

        if is_batch:
            if ini_theta is None:
                theta = np.random.uniform(0, 2 * np.pi, size=(batch_size, n))
            else:
                theta = ini_theta  # shape: (batch_size, n)
                batch_size = ini_theta.shape[0]

            m = np.zeros((batch_size, n))
            v = np.zeros((batch_size, n))
            converged = np.zeros(batch_size, dtype=bool)
        else:
            if ini_theta is None:
                theta = np.random.uniform(0, 2 * np.pi, size=n)
            else:
                theta = ini_theta

            m = np.zeros(n)
            v = np.zeros(n)
            converged = False

        for it in range(1, iterations + 1):
            grad_theta = grad_func(theta)

            m = beta1 * m + (1 - beta1) * grad_theta
            v = beta2 * v + (1 - beta2) * (grad_theta ** 2)

            m_hat = m / (1 - beta1 ** it)
            v_hat = v / (1 - beta2 ** it)

            if is_batch:
                grad_norm_sq = np.linalg.norm(grad_theta, axis=1) ** 2 / n
                converged = converged | (grad_norm_sq < threshold)
                if np.all(converged):
                    break
                update_mask = ~converged
                theta[update_mask] += learning_rate * m_hat[update_mask] / (np.sqrt(v_hat[update_mask]) + 1e-8)
                theta[update_mask] = np.mod(theta[update_mask], 2 * np.pi)
            else:
                grad_norm_sq = np.linalg.norm(grad_theta) ** 2 / n
                if grad_norm_sq < threshold:
                    converged = True
                    break
                theta += learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
                theta = np.mod(theta, 2 * np.pi)

        ws_obj = obj_func(theta)
        return ws_obj, theta, converged

    def optimize_theta(
        self,
        objective: str = 'rws',
        optimizer: str = 'ADAM',
        global_alpha: bool = False,
        lamd: float = 0,
        trials: int = 50,
        learning_rate: float = 0.1,
        iterations: int = 10**6,
        threshold: float = 1e-8,
        ini_theta=None,
    ):
        assert objective in ['rws', 'BM']
        assert optimizer in ['ADAM', 'SGD']

        G = self.G
        self.lamd = lamd
        if objective == 'BM':
            assert np.isclose(self.lamd, 0)

        if optimizer == 'ADAM':
            ws_obj, init_rot_list, converged = self.theta_adam(
                objective=objective, iterations=int(iterations), learning_rate=learning_rate,
                seed=42, batch_size=trials, lamd=lamd, ini_theta=ini_theta, beta1=0.9, beta2=0.999, threshold=threshold
            )
        elif optimizer == 'SGD':
            ws_obj, init_rot_list, converged = self.theta_sdg(
                objective=objective, iterations=int(iterations), learning_rate=learning_rate,
                seed=42, batch_size=trials, lamd=lamd
            )
        max_idx = np.argmax(ws_obj)
        best_obj = ws_obj[max_idx]
        best_rot = init_rot_list[max_idx]
        

        if global_alpha:
            def f_obj(alpha):
                f_value = 0
                for (i, j) in G.edges():
                    f_value += np.cos(alpha + best_rot[i]) * np.cos(alpha + best_rot[j])
                return f_value
            opt_res = scipy.optimize.minimize_scalar(f_obj)
            opt_alpha = opt_res.x
            best_rot = best_rot + opt_alpha

        # Compute p0 energy
        f_value = 0
        for (i, j) in G.edges():
            f_value += np.cos(best_rot[i]) * np.cos(best_rot[j])
        p0_energy = self.n_v * self.graph_degree / 4 - 0.5 * f_value

        self.theta = best_rot
        self.p0_energy = p0_energy
        return best_rot, p0_energy

    def get_ws_qaoa_para(self, p: int = 1, graph_degree: int = None, lamd: float = None):
        if graph_degree is None:
            graph_degree = self.graph_degree
        if lamd is None:
            if not hasattr(self, 'lamd'):
                raise ValueError("lamd must be specified either as an argument or by calling optimize_theta first")
            lamd = self.lamd
        if graph_degree == 3:
            if lamd == 0.6:
                if p == 1:
                    gamma = [0.99454978]
                    beta = [0.43320568]
                elif p == 2:
                    gamma = [0.99959445, 2.15094905]
                    beta = [0.46375345, 0.25644381]
                elif p == 3:
                    gamma = [0.55505739, 1.49498448, 2.30554454]
                    beta = [0.61608703, 0.32868682, 0.17954946]
                elif p == 4:
                    gamma = [0.41408643, 0.9740519,  1.92943177, 2.21302296]
                    beta = [0.80261765, 0.49482749, 0.24675206, 0.15168564]
                elif p == 5:
                    gamma = [0.31752051, 0.79044374, 1.36835544, 2.06237682, 2.15305022]
                    beta = [0.78580456, 0.61792615, 0.33801126, 0.19951751, 0.1207062]
                elif p == 6:
                    gamma = [0.27934874, 0.68453166, 1.15664332, 1.57022904, 2.10503058, 2.21426356]
                    beta = [0.78112069, 0.61774672, 0.42712325, 0.26989133, 0.16503647, 0.10071336]
                else:
                    raise ValueError(f"parameters for d={graph_degree}, lamd={lamd}, p={p} are not available")
            else:
                raise ValueError(f"parameters for d={graph_degree}, lamd={lamd} are not available")
        elif graph_degree == 4:
            if lamd == 0.7:
                if p == 1:
                    gamma = [0.99454978]
                    beta = [0.43320568]
                elif p == 2:
                    gamma = [0.99959445, 2.15094905]
                    beta = [0.46375345, 0.25644381]
                elif p == 3:
                    gamma = [0.55505739, 1.49498448, 2.30554454]
                    beta = [0.61608703, 0.32868682, 0.17954946]
                elif p == 4:
                    gamma = [0.41408643, 0.9740519,  1.92943177, 2.21302296]
                    beta = [0.80261765, 0.49482749, 0.24675206, 0.15168564]
                else:
                    raise ValueError(f"parameters for d={graph_degree}, lamd={lamd}, p={p} are not available")
            else:
                raise ValueError(f"parameters for d={graph_degree}, lamd={lamd} are not available")
        elif graph_degree == 5:
            if lamd == 0.7:
                if p == 1:
                    gamma = [0.98989608]
                    beta = [0.43731751]
                elif p == 2:
                    gamma = [0.98697006, 2.09575721]
                    beta = [0.48603188, 0.2606322]
                elif p == 3:
                    gamma = [0.59759232, 1.49985135, 2.12198031]
                    beta = [0.54704863, 0.32533531, 0.1977249]
                else:
                    raise ValueError(f"parameters for d={graph_degree}, lamd={lamd}, p={p} are not available")
            else:
                raise ValueError(f"parameters for d={graph_degree}, lamd={lamd} are not available")
        else:
            raise ValueError(f"parameters for d={graph_degree}, lamd={lamd} are not available")
        
        return gamma, beta
    
    ########################### QAOA Simulation ###########################
    #######################################################################
    def get_qaoa_para(self, p: int = 11):
        if p <= 11:
            gamma, beta = get_fixed_gamma_beta(self.graph_degree, p)
        else:
            import pickle
            
            assert np.isclose(self.graph_degree, 3)
            with open("../assets/maxcut_datasets/data_max_cut_qaoa_d2.pkl", "rb") as file:
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
        lamd: float = None,
    ):
        assert self.n_v <= 30
        if theta is None:
            theta = self.theta
        if (gamma is None) and (beta is None):
            gamma, beta = self.get_ws_qaoa_para(p, lamd=lamd)
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

    def run_standard_qaoa(self, p: int = 11):
        assert self.n_v <= 30
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

    def run_standard_qaoa(self, gamma: np.ndarray = None, beta: np.ndarray = None):
        assert self.n_v <= 30

        simclass = qokit.fur.choose_simulator(name="auto")
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
    
    def product_state_stats_and_grads_from_Q(self, theta, include_const=True):
        """
        For |psi(theta)> = ⊗_i (cos(theta_i/2)|0> + sin(theta_i/2)|1>),
        return mean, var, grad_mean (d/dtheta), grad_var (d/dtheta).
        """
        
        theta = np.asarray(theta, dtype=float)
        const, h, J = qubo_to_z_hamiltonian(self.Q)

        z  = np.cos(theta)          # <Z_i>
        s  = np.sin(theta)
        s2 = 1.0 - z**2             # = sin^2(theta)

        a = h + J @ z               # a = h + J z

        # Mean and its gradient
        mean = h @ z + 0.5 * z @ (J @ z)
        if include_const:
            mean += const
        grad_mean = -a * s          # element-wise

        # Variance and its gradient
        # Var = sum_i a_i^2 s_i^2 + sum_{i<j} J_ij^2 s_i^2 s_j^2
        var_linear = np.sum((a**2) * s2)
        var_pair   = 0.5 * np.sum((J**2) * (s2[:, None] * s2[None, :]))  # i<j
        variance   = var_linear + var_pair

        # Gradient of variance:
        # b = a * s^2
        b = a * s2
        term1 = -(J @ b)                       # shape (n,)
        term2 = (a**2) * z                     # shape (n,)
        term3 = z * ((J**2) @ s2)              # shape (n,)
        grad_var = 2.0 * s * (term1 + term2 + term3)

        return float(mean), float(variance), grad_mean, grad_var

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



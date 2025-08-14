import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Union, Iterable, Optional

"""
目的：
QAOAのパラメータである gamma と beta の値を変化させたときに、
目的関数（MaxCut問題におけるコスト関数）の期待値がどのように変わるかを計算し、
その結果をヒートマップとして表示．
To generate a heatmap showing the relationship between the QAOA parameters (gamma and beta) 
and the expectation value of the MaxCut cost function.
"""

# ---- Utility: Parameter Range ----
def _parse_param_range(parameter_range: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    パラメータの範囲を定義した辞書を解析し、ガンマとベータのNumPy配列に変換する。
    Parses the parameter range dictionary into a numpy array for gamma and beta values.

    input: parameter_range
    output: (gamma_values, beta_values)

    input shape
    - {'gamma': (min, max, num), 'beta': (min, max, num)}
    - {'gamma': {'min':..., 'max':..., 'num':...}, 'beta': {...}}
    """
    def _to_grid(spec):
        if isinstance(spec, (tuple, list)) and len(spec) == 3:
            gmin, gmax, gnum = spec
        elif isinstance(spec, dict):
            gmin, gmax, gnum = spec['min'], spec['max'], spec['num']
        else:
            raise ValueError("parameter_range['gamma'/'beta'] must be in the format (min,max,num) or {'min','max','num'}.")
        return np.linspace(float(gmin), float(gmax), int(gnum))

    gamma_values = _to_grid(parameter_range['gamma'])
    beta_values  = _to_grid(parameter_range['beta'])
    return gamma_values, beta_values

# ---- Base Quantum Gates: Matrix Generation ----
I = np.array([[1,0],[0,1]], dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)

def _kron_n(ops: Iterable[np.ndarray]) -> np.ndarray:
    """
    演算子のリストのクロネッカー積を計算する。
    Computes the Kronecker product of a list of operators.
    """
    out = np.array([[1]], dtype=complex)
    for op in ops:
        out = np.kron(out, op)
    return out

def _single_qubit_unitary_on(n: int, target: int, U: np.ndarray) -> np.ndarray:
    """
    n量子ビット系で、target量子ビットに1量子ビットユニタリUを適用する全体のユニタリ行列を返す。
    Returns the full unitary matrix for applying a single-qubit unitary U on a target qubit in an n-qubit system.
    """
    ops = []
    for q in range(n):
        ops.append(U if q == target else I)
    return _kron_n(ops)

def _exp_iA(A: np.ndarray, theta: float) -> np.ndarray:
    """
    `exp(-i * theta * A)`を固有値分解を使って計算する。
    Calculates exp(-i * theta * A) using eigendecomposition.
    """
    vals, vecs = np.linalg.eigh(A)
    phase = np.exp(-1j * theta * vals)
    return (vecs @ np.diag(phase) @ np.linalg.inv(vecs))

def _rzz_unitary(theta: float) -> np.ndarray:
    """
    RZZ(theta) = exp(-i * theta/2 * (Z ⊗ Z)) の4x4行列を返す（2量子ビットゲート）。
    Returns the 4x4 matrix for RZZ(theta) = exp(-i * theta/2 * (Z ⊗ Z)) (2-qubit gate).
    """
    # Eigenvalues of Z⊗Z are {+1, -1, -1, +1}
    diag = np.diag(np.exp(-1j * (theta/2) * np.array([1, -1, -1, 1], dtype=float)))
    return diag

def _two_qubit_unitary_on(n: int, i: int, j: int, U2: np.ndarray) -> np.ndarray:
    """
    n量子ビット系で、iとjの量子ビットに2量子ビットユニタリU2を適用する全体のユニタリ行列を返す。
    Returns the full unitary matrix for applying a 2-qubit unitary U2 on qubits (i,j) in an n-qubit system.
    For small-scale systems: constructs the full 2^n x 2^n matrix (recommended for n <= 12).
    Bit order is [q0, q1, ..., q_{n-1}], with the tensor product applied from left to right.
    """
    size = 2**n
    U_full = np.zeros((size, size), dtype=complex)
    for basis_in in range(size):
        # 2進数表現に変換
        b = [(basis_in >> (n-1-k)) & 1 for k in range(n)]
        # 2つのターゲット量子ビットのローカルな基底インデックス
        loc = (b[i] << 1) | b[j]
        for loc_out in range(4):
            amp = U2[loc_out, loc]
            if abs(amp) == 0:
                continue
            b_out = b.copy()
            b_out[i] = (loc_out >> 1) & 1
            b_out[j] = (loc_out >> 0) & 1
            # 整数インデックスに戻す
            out_idx = 0
            for k in range(n):
                out_idx = (out_idx << 1) | b_out[k]
            U_full[out_idx, basis_in] += amp
    return U_full

# ---- QAOA Core Functions ----
def _plus_state(n: int) -> np.ndarray:
    """
    全ての量子ビットを重ね合わせ状態 |+\rangle に初期化．
    Returns the |+>^{\otimes n} state.
    """
    state = np.ones((2**n,), dtype=complex) / (2**(n/2))
    return state

def _apply_mixer_layer(n: int, state: np.ndarray, beta: float) -> np.ndarray:
    """
    ミキサーハミルトニアンに対応するユニタリ操作を適用．
    各量子ビットに Rx ゲートを作用させることで実装．
    Applies the mixer layer U_B(β) = ∏_i exp(-i β X_i)
    """
    U = np.eye(2**n, dtype=complex)
    # Applying each single-qubit unitary sequentially to reduce the number of matrix multiplications
    for q in range(n):
        U_i = _single_qubit_unitary_on(n, q, _exp_iA(X, beta))
        U = U_i @ U
    return U @ state

def _apply_cost_layer(graph: nx.Graph, state: np.ndarray, gamma: float) -> np.ndarray:
    """
    コストハミルトニアンに対応するユニタリ操作を適用．
    グラフのエッジ（辺）ごとに,2つの量子ビットに作用するRZZゲートとして実装.
    Applies the cost layer U_C(γ) = exp(-i γ C) for the MaxCut problem, where C = Σ_(i,j∈E) (1 - Z_i Z_j)/2.
    The global phase can be ignored, so RZZ(-γ) is applied to each edge.
    """
    n = graph.number_of_nodes()
    # Fix the node order by relabeling to 0..n-1
    g = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    U = np.eye(2**n, dtype=complex)
    for (u, v) in g.edges():
        if u == v:
            continue
        U2 = _rzz_unitary(-gamma)  # Refer to the comment above
        U_uv = _two_qubit_unitary_on(n, u, v, U2)
        U = U_uv @ U
    return U @ state

def qaoa_state_for_params(graph: nx.Graph, gammas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """
    上記の2つのレイヤーを交互に繰り返し,最終的な量子状態を生成.
    Returns the final state of a p-layer QAOA circuit, |ψ(γ,β)⟩.
    The lengths of gammas and betas, p, must be equal.
    """
    assert len(gammas) == len(betas), "The lengths of gammas and betas, p, must be equal."
    n = graph.number_of_nodes()
    state = _plus_state(n)
    p = len(gammas)
    for layer in range(p):
        state = _apply_cost_layer(graph, state, gammas[layer])
        state = _apply_mixer_layer(n, state, betas[layer])
    return state

def maxcut_cost_expectation(graph: nx.Graph, state: np.ndarray) -> float:
    """
    生成された量子状態から、MaxCut問題 の目的関数（カットされるエッジの数の期待値）を計算.
    最終状態の各基底状態（ビット列）が出現する確率を求め,
    それぞれのビット列に対応するコストを重み付け平均することで算出.
    Calculates the expectation value ⟨C⟩ for the MaxCut cost operator.
    C = Σ_(i,j∈E) (1 - Z_i Z_j)/2
    """
    n = graph.number_of_nodes()
    g = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    rho_diag = np.abs(state)**2  # Probabilities of measuring each basis state
    # Sums the expectation value of C over all bitstrings (recommended for n <= 14).
    cost = 0.0
    for bitstring in range(2**n):
        # z_i ∈ {+1,-1}
        z = np.array([1 if ((bitstring >> (n-1-k)) & 1)==0 else -1 for k in range(n)], dtype=int)
        # C(b) = Σ_{(i,j)∈E} (1 - z_i z_j)/2
        c_b = 0
        for (u,v) in g.edges():
            c_b += (1 - z[u]*z[v]) / 2.0
        cost += rho_diag[bitstring] * c_b
    return float(np.real_if_close(cost))

# ---- Proxy QAOA ----
class ProxyQAOA:
    """
    A simple proxy for calculating a "graph-independent approximation."
    For example, 'triangle' uses the 3-vertex complete graph K3 as its internal model.
    In this context, it is "independent of the target graph" and returns the expectation
    value calculated on a small, representative internal graph.
    """
    def __init__(self, kind: str = "triangle"):
        kind = kind.lower()
        if kind == "triangle":
            self.name = "triangle"
            self.model_graph = nx.complete_graph(3)
        elif kind in ("edge", "k2"):
            self.name = "edge"
            self.model_graph = nx.complete_graph(2)
        elif kind in ("square", "cycle4", "c4"):
            self.name = "square"
            self.model_graph = nx.cycle_graph(4)
        else:
            raise ValueError(f"Unknown proxy kind: {kind}")

    def expectation_value(self, gammas: np.ndarray, betas: np.ndarray) -> float:
        state = qaoa_state_for_params(self.model_graph, gammas, betas)
        return maxcut_cost_expectation(self.model_graph, state)

# ---- Heatmap Data Collection ----
def collect_heatmap_data(
    graph_or_proxy: Union[nx.Graph, ProxyQAOA],
    initial_gammas: np.ndarray,
    initial_betas: np.ndarray,
    parameter_range: Dict,
    gamma_index: int = 0,
    beta_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    指定されたγとβベクトルの1つの要素をスキャンし、⟨C⟩のヒートマップ行列を返す。
    Scans one component of the specified γ and β vectors and returns the ⟨C⟩ heatmap matrix.
    Returns: (data_matrix, gamma_values, beta_values)
      - data_matrix.shape == (len(gamma_values), len(beta_values))
      - data_matrix[i, j] corresponds to gamma=gamma_values[i] and beta=beta_values[j]
    """
    gammas_base = np.array(initial_gammas, dtype=float).copy()
    betas_base  = np.array(initial_betas, dtype=float).copy()
    assert len(gammas_base) == len(betas_base), "The lengths p of initial_gammas and initial_betas must be equal."

    gamma_values, beta_values = _parse_param_range(parameter_range)
    data = np.zeros((len(gamma_values), len(beta_values)), dtype=float)

    # Check if it's a real QAOA (nx.Graph) or a proxy (ProxyQAOA)
    real_qaoa = isinstance(graph_or_proxy, nx.Graph)
    proxy_qaoa = isinstance(graph_or_proxy, ProxyQAOA)
    if not (real_qaoa or proxy_qaoa):
        raise TypeError("graph_or_proxy must be an nx.Graph or ProxyQAOA instance.")

    for i, gv in enumerate(gamma_values):
        for j, bv in enumerate(beta_values):
            gammas = gammas_base.copy()
            betas  = betas_base.copy()
            gammas[gamma_index] = gv
            betas[beta_index]   = bv

            if real_qaoa:
                state = qaoa_state_for_params(graph_or_proxy, gammas, betas)
                val = maxcut_cost_expectation(graph_or_proxy, state)
            else:
                val = graph_or_proxy.expectation_value(gammas, betas)

            data[i, j] = val

    return data, gamma_values, beta_values

# ---- Heatmap Plotting (replacement for grips.plot_utils.plot_heat_map) ----
def plot_heat_map(
    data: np.ndarray,
    gamma_values: np.ndarray,
    beta_values: np.ndarray,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    colorbar_label: str = "Expectation ⟨C⟩"
):
    """
    ヒートマップをプロットする。data[i, j]はgamma=gamma_values[i]とbeta=beta_values[j]に対応する。
    Plots a heatmap, assuming data[i, j] corresponds to gamma=gamma_values[i] and beta=beta_values[j].
    """
    plt.figure(figsize=(6,5))
    extent = [beta_values[0], beta_values[-1], gamma_values[0], gamma_values[-1]]
    plt.imshow(data, origin='lower', aspect='auto', extent=extent)
    plt.xlabel(xlabel if xlabel else "β")
    plt.ylabel(ylabel if ylabel else "γ")
    cbar = plt.colorbar()
    cbar.set_label(colorbar_label)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()

# ---- Demo: Plotting both Real and Proxy QAOA ----
if __name__ == "__main__":
    # QAOAの層の深さを指定 (整数p)
    # Specify the depth of the QAOA layers (any integer p)
    p_layers = 3 

    # 表示したい層のインデックスを指定（0から始まる）
    # Specify the index of the layer to plot (0-indexed)
    layer_to_plot = 1 
    # ------------------------------------

    # 指定された層の深さ p に応じて initial_gammas と initial_betas を動的に生成
    # 0.5から1.5の範囲でp個の値を等間隔で生成
    # Dynamically generate initial_gammas and initial_betas based on the specified depth p
    initial_gammas = np.linspace(0.5, 1.5, p_layers)
    initial_betas  = np.linspace(0.5, 1.5, p_layers)

    if layer_to_plot >= len(initial_gammas):
        print(f"Error: layer_to_plot (index {layer_to_plot}) is out of bounds for a p={len(initial_gammas)} QAOA.")
    else:
        param_range = {
            "gamma": {"min": -2.0, "max": 2.0, "num": 81},
            "beta":  {"min": -2.0, "max": 2.0, "num": 81},
        }

        # Example 1: Real QAOA (small-scale graph; calculations become heavy for large n)
        """
        6頂点を持つランダムグラフ（Erdős Rényiグラフ）に対して
        QAOAを実行した結果のヒートマップ
        """
        G = nx.erdos_renyi_graph(n=6, p=0.5, seed=42)  # Change as desired
        data_real, gvals, bvals = collect_heatmap_data(
            G, initial_gammas, initial_betas, param_range, gamma_index=layer_to_plot, beta_index=layer_to_plot
        )
        plot_heat_map(
            data_real, gvals, bvals,
            xlabel=f"β (layer {layer_to_plot})", ylabel=f"γ (layer {layer_to_plot})",
            title=f"Real QAOA (Erdős–Rényi n=6, p=0.5, Layer {layer_to_plot})"
        )

        # Example 2: Proxy QAOA (triangle proxy)
        """
        3頂点からなる完全グラフ（三角形）に対して,
        QAOAを実行した結果のヒートマップ
        """
        proxy = ProxyQAOA(kind="triangle")
        data_proxy, gvals_p, bvals_p = collect_heatmap_data(
            proxy, initial_gammas, initial_betas, param_range, gamma_index=layer_to_plot, beta_index=layer_to_plot
        )
        plot_heat_map(
            data_proxy, gvals_p, bvals_p,
            xlabel=f"β (layer {layer_to_plot})", ylabel=f"γ (layer {layer_to_plot})",
            title=f"Proxy QAOA (triangle, Layer {layer_to_plot})"
        )
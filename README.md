# Accelerating QAOA Performance in QOkit

This project outlines our performance optimization efforts targeting the Quantum Approximate Optimization Algorithm (QAOA) within the QOkit library. Our primary objective was to significantly enhance simulation efficiency on CPU-based Qiskit simulators, with a particular focus on computational scalability for real-world applications.

---

## Project Summary

The Quantum Approximate Optimization Algorithm (QAOA) is a key hybrid quantum-classical algorithm for solving combinatorial optimization problems. In this Phase 2 submission, we addressed critical performance bottlenecks in QOkit's QAOA implementation. Our dual optimization strategy—leveraging NumPy vectorization for energy precomputation and Numba JIT acceleration for quantum circuit simulation—yielded substantial speedups. Furthermore, we integrated advanced techniques like Domain Wall Encoding (DWE) for problem formulation and a multiprocessing strategy for classical optimization, resulting in a highly scalable and efficient framework. This work paves the way for tackling larger, more complex quantum optimization problems on classical hardware.

---

## 1. Bottleneck Identification & Rationale

During our initial analysis, we identified two significant performance bottlenecks:

- **Bottleneck #1: `precompute_energy` Function**  
  This function suffered from poor scaling due to its loop-heavy, non-vectorized implementation. As the number of assets (N) increased, execution times surged dramatically—jumping from 34 seconds at N=22 to 217 seconds at N=23. This spike was directly attributable to inefficient loop-based calculations that scaled poorly with N.

- **Bottleneck #2: QAOA Simulation in XY Mixer**  
  Deep profiling revealed that after optimizing `precompute_energy`, the majority of the remaining time was spent within the XY mixer simulation component. Specifically, the functions `apply_qaoa_furxy_ring()`, `furxy_ring()`, and `furxy()` (the latter invoked over 500 times) were critical performance drags. These routines involved the sequential application of XY gates on exponentially growing statevectors, creating a major performance bottleneck at higher problem sizes.

---

## 2. Technical Enhancement Plan

To address these critical performance issues, we implemented a multi-faceted enhancement plan:

### 2.1. Core Performance Enhancements

- **Vectorized Objective Function:**  
  We completely rewrote the energy precomputation routine, replacing inefficient loops with highly optimized NumPy vectorized operations. This allowed for batch matrix computations, drastically reducing time spent in energy precomputation and improving scalability.

- **JIT Compilation Using Numba:**  
  We applied Numba's Just-In-Time (JIT) compilation to the `furxy` and `furxy_ring` functions, decorating them with `@njit(parallel=True)`. This enabled multithreaded execution, significantly accelerating the simulation of XY gate sequences, which are core to the mixer Hamiltonian.

- **Selective Profiling & Retention:**  
  Through careful profiling, functions such as `get_problem()` and `get_initial_value()` were confirmed to be efficient and scale well, thus requiring no modifications.

These enhancements were chosen due to their strong compatibility with Python-based development and their potential to reduce compute time without introducing complex external dependencies.

### 2.2. Advanced Optimization Techniques

- **Domain Wall Encoding (DWE):**  
  We innovatively integrated Domain Wall Encoding with QAOA. DWE enables a more efficient and accurate representation of the portfolio selection problem by inherently enforcing the crucial constraint of selecting exactly K assets. This transforms the constrained problem into an unconstrained binary optimization problem (QUBO), ideally suited for QAOA. This leads to a specialized cost Hamiltonian, $H_C$, for our QAOA objective function:

![Cost Hamiltonian](https://latex.codecogs.com/svg.image?H_C%20%3D%20%5Csum_%7Bi%3Cj%7D%20J_%7Bij%7D%20%5Csigma_i%5Ez%20%5Csigma_j%5Ez%20&plus;%20%5Csum_i%20h_i%20%5Csigma_i%5Ez)

  Here, the coefficients $J_{ij}$ and $h_i$ are meticulously derived from the original portfolio optimization parameters (asset means $\mu$ and covariance matrix $\Sigma$) combined with a DWE-inspired penalty term $\lambda\sum$. This formulation cleverly embeds the portfolio size constraint directly into the energy landscape, ensuring that the lowest energy states correspond precisely to valid portfolios with K selected assets. A significant advantage of this DWE-penalty approach is its synergistic effect with the mixer Hamiltonian, guiding the quantum state evolution to predominantly explore only the valid subspace of solutions, thereby enhancing optimization efficiency.

- **Linear Mixer:**  
  For the mixer component of our QAOA circuit, we opted for a linear mixer. This deliberate choice was driven by its superior computational complexity of $O(N-1)$ compared to the $O(N^2)$ complexity of a complete graph XY-ring mixer. This linear scaling is vital for mitigating the increasing circuit depth and gate count as the number of assets (N) grows, directly contributing to more feasible simulations and potentially better performance on noisy intermediate-scale quantum (NISQ) devices.

- **Parallel Classical Optimization:**  
  We implemented a multiprocessing-based strategy for our classical optimizer. This allows for parallel runs of the classical optimization routine, significantly reducing the total wall time, especially for instances requiring multiple evaluations.

### 2.3. Framework Modifications

Our methodological advancements necessitated significant modifications to the QOkit framework:

- The `examples/QAOA_portfolio_optimization.ipynb` notebook was thoroughly redesigned to integrate the DWE problem definition, implement our parallel execution strategy, and host comprehensive benchmarking workflows.
- The `qokit/qaoa_objective_portfolio.py` module was updated to reflect the new DWE-derived QUBO coefficients, ensuring the QAOA objective accurately reflects the constrained portfolio problem.
- A new utility file, `qaoa_utils.py`, was introduced to house essential functions like `minimize_nlopt` (our classical optimizer interface) and `run_single_optimization`. This modularization was critical for enabling robust multiprocessing within the Jupyter environment, circumventing common serialization issues.

---

## 3. Benchmark Test Design & Baseline Results

Our benchmark design simulated QAOA with depths $p=1,2$ and time steps $T=1,2$, across a range of qubit counts from $N=8$ to $23$. We compared baseline runtimes against our enhanced versions on a Python-based CPU backend. All tests were run on a machine with an Intel Core i7 CPU and 16 GB RAM. Simulations were averaged over three runs to reduce variance and improve result stability.

### 3.1. Overall Performance Improvement

Results showed that for small N, speedups were modest—ranging from 10% to 20% on the Qiskit simulator—but they grew substantially as N increased. We achieved up to 75% performance improvement for $N>14$ when using CPU, demonstrating the power of NumPy-based optimization in quantum-classical hybrid algorithms.

| Number of Assets (N) | Original Time (s) | Optimized Time (s) | Speedup (%) |
|----------------------|-------------------|--------------------|-------------|
| 22                   | 34                | 8.8                | 74%         |
| 23                   | 217               | 45                 | 79.3%       |

*Figure 1. Execution time comparison between non-vectorized, vectorized, and vectorized + Numba implementations of QAOA energy precomputation for increasing number of assets (N). Vectorized + Numba shows superior scalability, maintaining low runtimes as N increases beyond 20.*  
*(Note: A graph illustrating this data would typically be embedded here in a full README.)*

### 3.2. Detailed Average Speedup (Python Simulator)

We observed significant speedups, particularly with the combined vectorized and Numba approach:

| Number of Assets (N) | Vectorized Speedup (x) | Vectorized + Numba Speedup (x) |
|----------------------|-----------------------|-------------------------------|
| 8                    | 1.02                  | 0.50                          |
| 11                   | 1.08                  | 1.19                          |
| 14                   | 1.28                  | 2.28                          |
| 17                   | 1.40                  | 6.53                          |
| 20                   | 1.37                  | 10.16                         |
| 23                   | 1.32                  | 10.31                         |

For instance, at $N=23$, we observed a 1.32x speedup with vectorization alone and a 10.31x speedup with the Python simulator when combined with Numba. The Numba improvement scales particularly well with increasing $p$ values (number of QAOA layers). For example, with $p=6$:

| Number of Assets (N) | Vectorized Speedup (x) | Vectorized + Numba Speedup (x) |
|----------------------|-----------------------|-------------------------------|
| 14                   | 1.12                  | 4.63                          |
| 17                   | 1.12                  | 17.58                         |

### 3.3. Qiskit Simulator Performance

For the Qiskit simulator backend, the observed improvements were marginal compared to the pure Python simulator, especially for larger $p$ values (e.g., $p=6$):

| Number of Assets (N) | Vectorized Speedup (x) | Vectorized + Numba Speedup (x) |
|----------------------|-----------------------|-------------------------------|
| 14                   | 1.26                  | 1.22                          |
| 17                   | 1.12                  | 1.12                          |

This outcome is expected because the Numba optimization primarily applies to the Python-based mixer simulator, which is a key component when using the Python-based Qiskit statevector simulation.

### 3.4. Classical Optimizer Performance

Our simulation results validate the effectiveness of our multiprocessing strategy for classical optimization:

| N  | p  | max_evals | num_parallel_runs | Wall Time (s) |
|----|----|-----------|-------------------|---------------|
| 20 | 1  | 1         | 1                 | 12.55         |
| 20 | 1  | 3         | 1                 | 35.17         |
| 20 | 1  | 1         | 4                 | 26.85         |
| 20 | 12 | 3         | 4                 | 226.17        |

These results highlight the exponential cost of deep or repeated classical optimization and reinforce the value of our multiprocessing strategy. For small instances ($N<20$), wall time remained under 2 seconds regardless of optimization settings. At intermediate N (e.g., $N=12,16$), increasing `max_evals` or parallel runs modestly raised compute time. But at large N (e.g., $N=20$), optimization time increased sharply, validating our multiprocessing strategy.

*Figure 3. Bar chart comparison of wall times under different classical optimization settings at N = 20. Using 4 parallel runs with fewer evaluations per process achieves significantly faster results than deep single-threaded optimization.*  
*(Note: A bar chart illustrating this data would typically be embedded here in a full README.)*

---

## 4. Optimization Use Case: Financial Portfolio Optimization

To contextualize this work, we selected financial portfolio optimization as a prime use case. In such applications, high asset counts ($N>14$) are common, and significant runtime improvements translate directly to broader parameter exploration and faster iteration cycles. This performance enhancement is thus practically relevant for real-world quantum optimization workflows in finance. For example, selecting 5 assets from a pool of 20 while optimizing expected return and maintaining acceptable volatility would require navigating a high-dimensional configuration space—where simulation speed is crucial.

---

## 5. Resource Estimates & Phase 3 Execution Strategy

Looking ahead to Phase 3, we plan to extend our work to GPU-based platforms:

- **GPU Acceleration:** Leveraging CuPy or CuQuantum for accelerated mixer layer simulations.
- **Larger-Scale Experiments:** Running experiments on Amazon Braket’s SV1/DM1 simulators.
- **Hardware Evaluation:** Evaluating performance on NVIDIA A100 hardware.
- **Ambitious Goals:** Our goal is to enable >20x acceleration for $N>40$ and establish a real-time benchmark loop using batching and asynchronous evaluations.
- **Backend Abstraction:** We also plan to integrate a backend abstraction layer to allow flexible deployment across local CPUs, cloud GPUs, and specialized simulators, ensuring that our enhancements scale beyond the current setup.

---

## Conclusion

This Phase 2 effort successfully addressed key performance bottlenecks in QOkit’s QAOA simulation pipeline through a comprehensive combination of NumPy vectorization, Numba JIT compilation, Domain Wall Encoding, and a multiprocessing-based classical optimizer strategy. These improvements collectively pave the way for a more scalable and efficient quantum optimization framework, making QAOA a more viable tool for practical, large-scale problems.

---

## Install

Creating a virtual environment is recommended before installing.

```bash
python -m venv qokit
source qokit/bin/activate
pip install -U pip
```

Install requires `python>=3.9` and `pip >= 23`. It is recommended to update your pip using `pip install --upgrade pip` before install.

```bash
git clone https://github.com/jpmorganchase/QOKit.git
cd QOKit/
pip install -e .
```

Some optional parts of the package require additional dependencies:

- GPU simulation:  
  `pip install -e .[GPU-CUDA12]`
- Generating LP files to solve LABS using commercial IP solvers (`qokit/classical_methods` and `examples/advanced/classical_solvers_for_LABS/`):  
  `pip install -e .[solvers]`

Please note that the GPU dependency is specified for CUDA 12x. For other versions of CUDA, please follow cupy installation instructions.

If compilation fails, try installing just the Python version using:

```bash
QOKIT_PYTHON_ONLY=1 pip install -e .
```

Installation can be verified by running tests using `pytest`.

---
from pickle import FALSE

import numpy as np
import time
import matplotlib.pyplot as plt
import importlib
from qokit.portfolio_optimization import get_problem, get_sk_ini
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective
import qokit.config

N_values = list(range(6, 25,3))
p_values = [1, 2,3]
T_values = [1, 2]
precomputed_energies_options = [None, "vectorized"]

results = {}

for T in T_values:
    for p in p_values:
        for precomputed in precomputed_energies_options:
            if precomputed == "vectorized":
                for use_numba in [False, True]:
                    qokit.config.USE_NUMBA = use_numba
                    import qokit.fur.python.fur
                    importlib.reload(qokit.fur.python.fur)
                    import qokit.fur.python.qaoa_fur
                    importlib.reload(qokit.fur.python.qaoa_fur)
                    import qokit.fur.python.qaoa_simulator
                    importlib.reload(qokit.fur.python.qaoa_simulator)

                    label = "vectorized-numba" if use_numba else "vectorized-nonumba"
                    times = []
                    for N in N_values:
                        po_problem = get_problem(N=N, K=3, q=0.5, seed=1, pre="rule")
                        start = time.time()
                        po_problem = get_problem(N=N, K=3, q=0.5, seed=1, pre="rule")
                        qaoa_obj = get_qaoa_portfolio_objective(
                            po_problem=po_problem,
                            p=p,
                            ini='dicke',
                            mixer='trotter_ring',
                            T=T,
                            simulator='python',
                            precomputed_energies=precomputed
                        )
                        x0 = get_sk_ini(p=p)
                        _ = qaoa_obj(x0).real
                        elapsed = time.time() - start
                        times.append(elapsed)
                        print(f"Completed T={T}, p={p}, precomputed={precomputed}, use_numba={use_numba}, N={N}, times={elapsed}")
                    results[(T, p, label)] = times

            else:
                # For non-vectorized, just use one setting (no Numba)
                qokit.config.USE_NUMBA = False
                import qokit.fur.python.fur
                importlib.reload(qokit.fur.python.fur)
                import qokit.fur.python.qaoa_fur
                importlib.reload(qokit.fur.python.qaoa_fur)
                import qokit.fur.python.qaoa_simulator
                importlib.reload(qokit.fur.python.qaoa_simulator)
                times = []
                for N in N_values:
                    po_problem = get_problem(N=N, K=3, q=0.5, seed=1, pre="rule")
                    start = time.time()
                    po_problem = get_problem(N=N, K=3, q=0.5, seed=1, pre="rule")
                    qaoa_obj = get_qaoa_portfolio_objective(
                        po_problem=po_problem,
                        p=p,
                        ini='dicke',
                        mixer='trotter_ring',
                        T=T,
                        simulator='python',
                        precomputed_energies=precomputed
                    )
                    x0 = get_sk_ini(p=p)
                    _ = qaoa_obj(x0).real
                    elapsed = time.time() - start
                    times.append(elapsed)
                    print(f"Completed T={T}, p={p}, precomputed={precomputed}, use_numba= False, N={N}, times={elapsed}")
                results[(T, p, "nonvec")] = times


# Compute average time per N for each method
avg_times_per_N = {"nonvec": [], "vectorized-nonumba": [], "vectorized-numba": []}
for idx, N in enumerate(N_values):
    for label in avg_times_per_N.keys():
        times = []
        for T in T_values:
            for p in p_values:
                times.append(results[(T, p, label)][idx])
        avg_times_per_N[label].append(np.mean(times))

# Compute speedup per N
speedup_per_N = []
speedup_per_N_numba = []
for idx, N in enumerate(N_values):
    nonvec = avg_times_per_N["nonvec"][idx]
    vec = avg_times_per_N["vectorized-nonumba"][idx]
    vec_numba = avg_times_per_N["vectorized-numba"][idx]
    speedup_per_N.append(nonvec / vec)
    speedup_per_N_numba.append(nonvec / vec_numba)
    print(f"Average speedup for N={N}: vectorized={nonvec/vec:.2f}x, vectorized+numba={nonvec/vec_numba:.2f}x")

# Total average speedup
print(f"\nTotal average speedup (vectorized): {np.mean(speedup_per_N):.2f}x")
print(f"Total average speedup (vectorized+numba): {np.mean(speedup_per_N_numba):.2f}x")

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(N_values, avg_times_per_N["nonvec"], label="Non-vectorized", marker='o')
plt.plot(N_values, avg_times_per_N["vectorized-nonumba"], label="Vectorized", marker='o')
plt.plot(N_values, avg_times_per_N["vectorized-numba"], label="Vectorized+Numba", marker='o')
plt.xlabel("N")
plt.ylabel("Average Time (s)")
plt.title("Average Computation Time vs N")
plt.legend()
plt.tight_layout()
plt.show()
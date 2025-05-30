import time
import matplotlib.pyplot as plt
from qokit.portfolio_optimization import get_problem_vectorized
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective
import numpy as np


Ns = [6, 10, 14, 20, 22]  # Example values; change as needed
ps = list(range(1, 7))  # p from 1 to 6
simulators = ['python', 'qiskit']
K, q, seed, pre = 3, 0.5, 1, 'rule'

modes = ['vectorized', None]
timings = {(sim, N, mode): [] for sim in simulators for N in Ns for mode in modes}

for simulator in simulators:
    for N in Ns:
        for p in ps:
            po_problem = get_problem_vectorized(N=N, K=K, q=q, seed=seed, pre=pre)
            for mode in modes:
                print(f"N={N}, p={p}, simulator={simulator}, mode={mode}")
                # Patch the objective function mode inside get_qaoa_portfolio_objective
                start = time.perf_counter()
                get_qaoa_portfolio_objective(po_problem=po_problem, p=p, ini='dicke', mixer='trotter_ring', T=1, simulator=simulator,precomputed_energies=mode)
                elapsed = time.perf_counter() - start
                timings[(simulator, N, mode)].append(elapsed)


# Summarize speedup
print("\nSpeedup summary (vectorized vs non-vectorized):")
speedup_table = []
total_speedup = []
for simulator in simulators:
    for N in Ns:
        vec_times = timings[(simulator, N, 'vectorized')]
        nonvec_times = timings[(simulator, N, None)]
        speedups = [(v - nv) / nv * 100 if nv > 0 else 0 for v, nv in zip(vec_times, nonvec_times)]
        avg_speedup = np.mean(speedups)
        speedup_table.append((simulator, N, avg_speedup))
        total_speedup.extend(speedups)
        print(f"Simulator: {simulator}, N={N}, Average speedup: {avg_speedup:.1f}%")

if total_speedup:
    overall_avg = np.mean(total_speedup)
    print(f"\nOverall average speedup: {overall_avg:.1f}%")
else:
    print("No timings to compute overall average speedup.")




plt.figure(figsize=(12, 7))
for simulator in simulators:
    for N in Ns:
        for mode in modes:
            plt.plot(ps, timings[(simulator, N, mode)], marker='o', label=f"{mode}, Simulator: {simulator}, N={N}")
plt.xlabel("p (QAOA depth)")
plt.ylabel("Creation time (s)")
plt.title("get_qaoa_portfolio_objective Creation Time: Vectorized vs Non-Vectorized")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Summarize speedup
print("\nSpeedup summary (vectorized vs non-vectorized):")
speedup_table = []
total_speedup = []
for simulator in simulators:
    for N in Ns:
        vec_times = timings[(simulator, N, 'vectorized')]
        nonvec_times = timings[(simulator, N, None)]
        speedups = [(v - nv) / nv * 100 if nv > 0 else 0 for v, nv in zip(vec_times, nonvec_times)]
        avg_speedup = np.mean(speedups)
        speedup_table.append((simulator, N, avg_speedup))
        total_speedup.extend(speedups)
        print(f"Simulator: {simulator}, N={N}, Average speedup: {avg_speedup:.1f}%")

if total_speedup:
    overall_avg = np.mean(total_speedup)
    print(f"\nOverall average speedup: {overall_avg:.1f}%")
else:
    print("No timings to compute overall average speedup.")

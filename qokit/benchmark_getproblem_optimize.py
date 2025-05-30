import time
import matplotlib.pyplot as plt
from qokit.portfolio_optimization import get_problem, get_problem_vectorized

def benchmark(func, N, K, q, seed, pre, repeat=3):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func(N=N, K=K, q=q, seed=seed, pre=pre)
        times.append(time.perf_counter() - start)
    return min(times)

Ns = list(range(4, 25, 2))
q = 0.5
seed = 1
pre = 'rule'

results = {}
improvements = {}

for N in Ns:
    Ks = list(range(2, N-1))
    for K in Ks:
        t1 = benchmark(get_problem, N, K, q, seed, pre)
        t2 = benchmark(get_problem_vectorized, N, K, q, seed, pre)
        perc_impr = 100 * (t1 - t2) / t1 if t1 > 0 else 0
        results[(N, K)] = (t1, t2)
        improvements[(N, K)] = perc_impr
        print(f"N={N}, K={K}: get_problem={t1:.5f}s, get_problem_vectorized={t2:.5f}s, improvement={perc_impr:.2f}%")

# Compute and print average improvement
if improvements:
    avg_improvement = sum(improvements.values()) / len(improvements)
    print(f"\nAverage percentage improvement: {avg_improvement:.2f}%")

# Plot: For each K, plot improvement vs N
plt.figure(figsize=(12, 7))
unique_Ks = sorted(set(k for (_, k) in improvements.keys()))
for K in unique_Ks:
    Ns_for_K = [N for N in Ns if 2 <= K < N-1]
    impr_for_K = [improvements[(N, K)] for N in Ns_for_K if (N, K) in improvements]
    if impr_for_K:
        plt.plot(Ns_for_K, impr_for_K, marker='o', label=f"K={K}")

plt.xlabel("N")
plt.ylabel("Percentage Improvement (%)")
plt.title("Percentage Improvement of get_problem_vectorized over get_problem")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
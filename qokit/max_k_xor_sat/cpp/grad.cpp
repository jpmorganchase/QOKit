/**
 * Gradient via central finite differences, parallelized across parameters.
 *
 * All 4p perturbation evaluations are independent: each one perturbs a
 * single angle by +h or -h and runs a full contraction.
 *
 * Parallelism strategy: run up to max_concurrent tasks simultaneously,
 * each as a single-threaded contraction on its own core.  The number of
 * outer threads equals min(batch_size, available_cores), which may exceed
 * OMP_NUM_THREADS (tuned for internal parallelism within a single eval).
 *
 * The central value is computed first with full internal OpenMP parallelism.
 */

#include "grad.h"
#include "contract.h"
#include <vector>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __linux__
#include <unistd.h>
#endif


double gradient_fd(const double* gammas, const double* betas,
                   int p, int D, int k,
                   double* grad_gammas, double* grad_betas,
                   double h, bool use_dd) {
    // Adaptive FD step: balance truncation error O(h²) vs roundoff O(ε/h).
    // Noise floor ε ≈ sqrt(a)^p × machine_eps, optimal h ≈ ε^(1/3).
    if (h <= 0.0) {
        double a = (double)(D - 1) * (k - 1);
        double eps = use_dd ? 1e-32 : 2.3e-16;
        double noise = (a > 1.0) ? std::pow(a, p / 2.0) * eps : eps;
        noise = std::max(noise, use_dd ? 1e-30 : 1e-14);
        h = std::cbrt(noise);
        h = std::max(h, 1e-8);
        h = std::min(h, 1e-2);
    }

    // Central value — uses all cores via internal OpenMP
    double val = contract_symmetric_tree(gammas, betas, p, D, k, use_dd);

    int n_tasks = 4 * p;
    std::vector<double> results(n_tasks);

    // Memory-aware parallelism: each contraction allocates ~4^p * 16 bytes.
    // Running too many in parallel exhausts RAM.
    size_t tensor_bytes = 1;
    for (int i = 0; i < p; i++) tensor_bytes *= 4;
    tensor_bytes *= 16;  // complex128
    if (use_dd) tensor_bytes *= 2;  // DD stores 2x doubles per component

    // Available RAM (default 700 GiB, override with QAOA_RAM_GIB)
    size_t ram_bytes = 700ULL * 1024 * 1024 * 1024;
    const char* ram_env = getenv("QAOA_RAM_GIB");
    if (ram_env) ram_bytes = (size_t)atol(ram_env) * 1024ULL * 1024 * 1024;

    // Each contraction needs ~3x tensor_bytes (input + intermediates + output)
    size_t per_task_bytes = tensor_bytes * 3;
    int max_concurrent = std::max(1, (int)(ram_bytes / per_task_bytes));
    max_concurrent = std::min(max_concurrent, n_tasks);

    // Available CPU cores for task-level parallelism.  This may exceed
    // OMP_NUM_THREADS, which is typically set for intra-eval parallelism
    // (e.g., 48 for memory-bandwidth-bound ops on a 96-core machine).
    // For task-level FD parallelism, each task is single-threaded and runs
    // on its own core, so we want as many outer threads as tasks/cores.
    int n_cores = 1;
#ifdef _OPENMP
    n_cores = omp_get_max_threads();
#endif
#ifdef __linux__
    {
        long nproc = sysconf(_SC_NPROCESSORS_ONLN);
        if (nproc > 0) n_cores = std::max(n_cores, (int)nproc);
    }
#endif

    // Process tasks in memory-limited batches.  Each batch runs up to
    // max_concurrent single-threaded contractions in parallel.
    for (int batch_start = 0; batch_start < n_tasks;
         batch_start += max_concurrent) {
        int batch_end = std::min(batch_start + max_concurrent, n_tasks);
        int n_batch = batch_end - batch_start;
        int outer_threads = std::min(n_batch, n_cores);

        #pragma omp parallel for schedule(dynamic, 1) num_threads(outer_threads)
        for (int task = batch_start; task < batch_end; task++) {
            std::vector<double> g_local(gammas, gammas + p);
            std::vector<double> b_local(betas, betas + p);
            int group = task / p, idx = task % p;
            if (group == 0) g_local[idx] += h;
            else if (group == 1) g_local[idx] -= h;
            else if (group == 2) b_local[idx] += h;
            else b_local[idx] -= h;
            results[task] = contract_symmetric_tree(
                g_local.data(), b_local.data(), p, D, k, use_dd);
        }
    }

    double inv_2h = 1.0 / (2.0 * h);
    for (int i = 0; i < p; i++) {
        grad_gammas[i] = (results[i] - results[p + i]) * inv_2h;
        grad_betas[i] = (results[2 * p + i] - results[3 * p + i]) * inv_2h;
    }

    return val;
}

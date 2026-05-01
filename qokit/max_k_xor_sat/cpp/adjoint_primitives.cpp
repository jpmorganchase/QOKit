/**
 * Adjoint (reverse-mode) primitives for WHT charge contraction and mode products.
 *
 * Convention: "real gradient" adjoint where adj_z = dL/dRe(z) + i*dL/dIm(z).
 * For holomorphic y = f(z): adj_z = adj_y * conj(f'(z)).
 * For y = a*b: adj_a = adj_y * conj(b), adj_b = adj_y * conj(a).
 * For real parameter: grad_theta = Re(adj_z * conj(dz/dtheta)).
 *
 * Templated on Scalar (C128 or DDComplex).
 */

#include "adjoint.h"
#include "primitives.h"
#include <cstring>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif


template<typename Scalar>
void wht_charge_contract_adjoint(const Scalar* M, const Scalar* T,
                                  const Scalar* adj_out,
                                  Scalar* adj_T, Scalar* adj_M,
                                  size_t R, size_t rest) {
    size_t vec_len = 4 * rest;
    size_t T_size = R * 16 * rest;
    std::memset(adj_T, 0, T_size * sizeof(Scalar));

    constexpr int PAD = 32;
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    std::vector<Scalar> adj_M_local(PAD * n_threads, Scalar(C128(0.0)));

    #pragma omp parallel
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        Scalar* my_adj_M = adj_M_local.data() + tid * PAD;

        #pragma omp for schedule(static) collapse(3)
        for (size_t i = 0; i < R; i++) {
            for (size_t b = 0; b < 4; b++) {
                for (size_t r = 0; r < rest; r++) {
                    size_t t_base = i * 16 * rest + b * rest + r;
                    size_t stride = 4 * rest;
                    size_t base_out = i * vec_len + b * rest + r;

                    Scalar a0 = adj_out[0 * R * vec_len + base_out];
                    Scalar a1 = adj_out[1 * R * vec_len + base_out];
                    Scalar a2 = adj_out[2 * R * vec_len + base_out];
                    Scalar a3 = adj_out[3 * R * vec_len + base_out];

                    // Reverse WHT butterfly (self-transpose)
                    Scalar p02 = a0 + a1, q02 = a0 - a1;
                    Scalar p13 = a2 + a3, q13 = a2 - a3;
                    Scalar adj_e0 = p02 + p13;
                    Scalar adj_e1 = q02 + q13;
                    Scalar adj_e2 = p02 - p13;
                    Scalar adj_e3 = q02 - q13;

                    adj_T[t_base + 0 * stride] += conj_scalar(M[b * 4 + 0]) * adj_e0;
                    adj_T[t_base + 1 * stride] += conj_scalar(M[b * 4 + 1]) * adj_e1;
                    adj_T[t_base + 2 * stride] += conj_scalar(M[b * 4 + 2]) * adj_e2;
                    adj_T[t_base + 3 * stride] += conj_scalar(M[b * 4 + 3]) * adj_e3;

                    my_adj_M[b * 4 + 0] += conj_scalar(T[t_base + 0 * stride]) * adj_e0;
                    my_adj_M[b * 4 + 1] += conj_scalar(T[t_base + 1 * stride]) * adj_e1;
                    my_adj_M[b * 4 + 2] += conj_scalar(T[t_base + 2 * stride]) * adj_e2;
                    my_adj_M[b * 4 + 3] += conj_scalar(T[t_base + 3 * stride]) * adj_e3;
                }
            }
        }
    }

    for (int t = 0; t < n_threads; t++)
        for (int j = 0; j < 16; j++)
            adj_M[j] += adj_M_local[t * PAD + j];
}


template<typename Scalar>
void mode_product_adjoint(const Scalar* W, const Scalar* src,
                           const Scalar* adj_dst,
                           Scalar* adj_src, Scalar* adj_W,
                           size_t total, int ell, int num_rounds) {
    size_t suffix = 1;
    for (int i = ell + 1; i < num_rounds; i++) suffix *= 4;
    size_t stride = 4 * suffix;

    // Pass 1: adj_src = W^H @ adj_dst
    #pragma omp parallel for schedule(static)
    for (size_t flat = 0; flat < total; flat++) {
        size_t p_idx = flat / stride;
        size_t rem = flat % stride;
        int j = (int)(rem / suffix);
        size_t s = rem % suffix;

        Scalar sum(C128(0.0));
        for (int h = 0; h < 4; h++)
            sum += conj_scalar(W[h * 4 + j]) *
                   adj_dst[p_idx * stride + h * suffix + s];
        adj_src[p_idx * stride + j * suffix + s] = sum;
    }

    // Pass 2: adj_W[h,j] = sum adj_dst * conj(src)
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    constexpr int PAD_W = 32;
    std::vector<Scalar> adj_W_local(PAD_W * n_threads, Scalar(C128(0.0)));

    #pragma omp parallel
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        Scalar* my_adj_W = adj_W_local.data() + tid * PAD_W;

        #pragma omp for schedule(static)
        for (size_t flat = 0; flat < total; flat++) {
            size_t p_idx = flat / stride;
            size_t rem = flat % stride;
            int h = (int)(rem / suffix);
            size_t s = rem % suffix;

            Scalar adj_val = adj_dst[p_idx * stride + h * suffix + s];
            size_t base = p_idx * stride + s;

            for (int j = 0; j < 4; j++)
                my_adj_W[h * 4 + j] += conj_scalar(src[base + j * suffix]) * adj_val;
        }
    }

    for (int t = 0; t < n_threads; t++)
        for (int j = 0; j < 16; j++)
            adj_W[j] += adj_W_local[t * PAD_W + j];
}


// Explicit instantiations
template void wht_charge_contract_adjoint<C128>(const C128*, const C128*, const C128*, C128*, C128*, size_t, size_t);
template void wht_charge_contract_adjoint<DDComplex>(const DDComplex*, const DDComplex*, const DDComplex*, DDComplex*, DDComplex*, size_t, size_t);
template void mode_product_adjoint<C128>(const C128*, const C128*, const C128*, C128*, C128*, size_t, int, int);
template void mode_product_adjoint<DDComplex>(const DDComplex*, const DDComplex*, const DDComplex*, DDComplex*, DDComplex*, size_t, int, int);

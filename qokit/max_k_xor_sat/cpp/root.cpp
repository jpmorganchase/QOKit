/**
 * Root contraction using factored rank-1 representation.
 * Templated on Scalar (C128 or DDComplex).
 */

#include "root.h"
#include "dispatch.h"
#include <cmath>
#include <type_traits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif


template<typename Scalar>
static double root_contract_impl(const Scalar* rb, const double* gammas,
                                  const double* betas, int p, int D, int k,
                                  bool verbose, bool mutable_rb = false) {
    size_t rb_size = 1;
    for (int i = 0; i < p; i++) rb_size *= 4;

    Scalar half_k = pow_scalar(Scalar(0.5), k);
    std::vector<Scalar> coeffs(1, half_k);

    // If mutable_rb, reuse rb directly as one ping-pong buffer (saves 4^p memory).
    // Otherwise, copy rb into a new vector (needed when caller uses rb later, e.g. adjoint).
    std::vector<Scalar> factor_owned;
    Scalar* src;
    if (mutable_rb) {
        src = const_cast<Scalar*>(rb);
    } else {
        factor_owned.assign(rb, rb + rb_size);
        hint_alloc(factor_owned.data(), rb_size * sizeof(Scalar));
        src = factor_owned.data();
    }

    std::vector<Scalar> buf_b_vec(rb_size);
    hint_alloc(buf_b_vec.data(), rb_size * sizeof(Scalar));
    first_touch_zero(buf_b_vec.data(), rb_size);
    Scalar* dst = buf_b_vec.data();

    size_t R = 1;
    size_t vec_len = rb_size;

    for (int ell = 0; ell < p - 1; ell++) {
        Scalar M[16], u[4];
        precompute_mixer_and_weights(betas[ell], gammas[ell], M, u);

        size_t rest = vec_len / 16;
        size_t new_vec_len = vec_len / 4;

        wht_dispatch(M, src, dst, R, rest);

        std::vector<Scalar> new_coeffs(4 * R);
        if (4 * R * sizeof(Scalar) >= 2 * 1024 * 1024)
            hint_alloc(new_coeffs.data(), 4 * R * sizeof(Scalar));
        #pragma omp parallel for collapse(2) schedule(static)
        for (int a = 0; a < 4; a++)
            for (size_t j = 0; j < R; j++)
                new_coeffs[a * R + j] = u[a] * coeffs[j];

        coeffs = std::move(new_coeffs);
        std::swap(src, dst);
        R *= 4;
        vec_len = new_vec_len;
    }

    // Final round + Z measurement
    Scalar M[16], u[4];
    precompute_mixer_and_weights(betas[p - 1], gammas[p - 1], M, u);

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif

    Scalar result_val{};
    for (int a = 0; a < 4; a++) {
        Scalar tv[4];
        for (int s = 0; s < 4; s++) {
            Scalar cd(CHARGE_DIAG[a][s]);
            tv[s] = M[0 * 4 + s] * cd - M[3 * 4 + s] * cd;
        }

        if constexpr (std::is_same_v<Scalar, C128>) {
            // Kahan compensated summation: per-thread Kahan accumulators
            // reduce error from O(R * eps) to O(eps) per thread.
            std::vector<double> partial_re(nthreads, 0.0);
            std::vector<double> partial_im(nthreads, 0.0);

            #pragma omp parallel
            {
                int tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                double sum_re = 0.0, c_re = 0.0;
                double sum_im = 0.0, c_im = 0.0;

                #pragma omp for schedule(static)
                for (size_t j = 0; j < R; j++) {
                    C128 z(0.0);
                    for (int s = 0; s < 4; s++) z += src[j * 4 + s] * tv[s];
                    // Compute z^k in DD precision to avoid rounding accumulation
                    C128 zk = dd_cpow(DDComplex(z), k).to_c128();
                    C128 term = coeffs[j] * zk;

                    double y_re = term.real() - c_re;
                    double t_re = sum_re + y_re;
                    c_re = (t_re - sum_re) - y_re;
                    sum_re = t_re;

                    double y_im = term.imag() - c_im;
                    double t_im = sum_im + y_im;
                    c_im = (t_im - sum_im) - y_im;
                    sum_im = t_im;
                }
                partial_re[tid] = sum_re;
                partial_im[tid] = sum_im;
            }

            double accum_re = 0.0, accum_im = 0.0;
            for (int t = 0; t < nthreads; t++) {
                accum_re += partial_re[t];
                accum_im += partial_im[t];
            }
            result_val += u[a] * C128(accum_re, accum_im);
        } else {
            // DD path: use thread-local DDComplex accumulators combined
            // with proper DD addition (fixes naive component-wise reduction).
            std::vector<DDComplex> partial_dd(nthreads);

            #pragma omp parallel
            {
                int tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                DDComplex local_sum;

                #pragma omp for schedule(static)
                for (size_t j = 0; j < R; j++) {
                    DDComplex z;
                    for (int s = 0; s < 4; s++) z += src[j * 4 + s] * tv[s];
                    DDComplex term = coeffs[j] * pow_scalar(z, k);
                    local_sum += term;
                }
                partial_dd[tid] = local_sum;
            }

            DDComplex accum;
            for (int t = 0; t < nthreads; t++) {
                accum += partial_dd[t];
            }
            result_val += u[a] * accum;
        }
    }

    return to_real(result_val);
}


double root_contract(const C128* rb, const double* gammas, const double* betas,
                     int p, int D, int k, bool verbose, bool mutable_rb) {
    return root_contract_impl(rb, gammas, betas, p, D, k, verbose, mutable_rb);
}

double root_contract_dd(const DDComplex* rb, const double* gammas, const double* betas,
                        int p, int D, int k, bool verbose, bool mutable_rb) {
    return root_contract_impl(rb, gammas, betas, p, D, k, verbose, mutable_rb);
}

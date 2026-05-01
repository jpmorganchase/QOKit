/**
 * Backward pass through root contraction.
 * Templated on Scalar (C128 or DDComplex).
 */

#include "adjoint.h"
#include "primitives.h"
#include "dispatch.h"
#include <cmath>
#include <cstring>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif


template<typename Scalar>
std::vector<Scalar> backward_root(double adj_raw,
                                   const Scalar* rb,
                                   const double* gammas, const double* betas,
                                   int p, int D, int k,
                                   double* grad_gammas, double* grad_betas) {
    size_t rb_size = 1;
    for (int i = 0; i < p; i++) rb_size *= 4;

    // ── Replay forward root to cache intermediate states ────────
    Scalar half_k(std::pow(0.5, k));
    std::vector<Scalar> coeffs_init(1, half_k);
    std::vector<Scalar> factor_init(rb, rb + rb_size);

    struct RoundState {
        std::vector<Scalar> coeffs;
        std::vector<Scalar> factor;
        size_t R;
        size_t vec_len;
    };

    std::vector<RoundState> states(p);
    states[0] = {coeffs_init, factor_init, 1, rb_size};

    {
        auto coeffs = coeffs_init;
        auto factor = factor_init;
        size_t R = 1;
        size_t vec_len = rb_size;

        for (int ell = 0; ell < p - 1; ell++) {
            Scalar M[16], u[4];
            precompute_mixer_and_weights<Scalar>(betas[ell], gammas[ell], M, u);

            size_t rest = vec_len / 16;
            std::vector<Scalar> new_factor(4 * R * vec_len / 4);
            size_t new_vec_len = vec_len / 4;

            wht_dispatch(M, factor.data(), new_factor.data(), R, rest);

            std::vector<Scalar> new_coeffs(4 * R);
            for (int a = 0; a < 4; a++)
                for (size_t j = 0; j < R; j++)
                    new_coeffs[a * R + j] = u[a] * coeffs[j];

            coeffs = std::move(new_coeffs);
            factor = std::move(new_factor);
            R *= 4;
            vec_len = new_vec_len;

            if (ell + 1 < p)
                states[ell + 1] = {coeffs, factor, R, vec_len};
        }
    }

    // ── Final round forward + backward ──────────────────────────
    size_t R_final = states[p - 1].R;
    const auto& coeffs_final = states[p - 1].coeffs;
    const auto& factor_final = states[p - 1].factor;

    Scalar M_final[16], u_final[4];
    precompute_mixer_and_weights<Scalar>(betas[p - 1], gammas[p - 1], M_final, u_final);

    // Trace vectors
    Scalar tv_all[4][4];
    for (int a = 0; a < 4; a++)
        for (int s = 0; s < 4; s++) {
            Scalar k0 = M_final[0 * 4 + s] * Scalar(C128(CHARGE_DIAG[a][s]));
            Scalar k3 = M_final[3 * 4 + s] * Scalar(C128(CHARGE_DIAG[a][s]));
            tv_all[a][s] = k0 - k3;
        }

    std::vector<Scalar> adj_coeffs_final(R_final, Scalar(C128(0.0)));
    std::vector<Scalar> adj_factor_final(R_final * 4, Scalar(C128(0.0)));
    Scalar adj_u_final[4] = {};
    Scalar adj_tv[4][4] = {};
    Scalar adj_M_final[16] = {};

    for (int a = 0; a < 4; a++) {
        Scalar adj_accum_a = Scalar(adj_raw) * conj_scalar(u_final[a]);

        double adj_u_re = 0.0, adj_u_im = 0.0;
        int n_threads = 1;
#ifdef _OPENMP
        n_threads = omp_get_max_threads();
#endif
        std::vector<Scalar> adj_tv_local(4 * n_threads, Scalar(C128(0.0)));

        #pragma omp parallel
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            Scalar* my_adj_tv = adj_tv_local.data() + tid * 4;
            double local_adj_u_re = 0.0, local_adj_u_im = 0.0;

            #pragma omp for schedule(static)
            for (size_t j = 0; j < R_final; j++) {
                Scalar z(C128(0.0));
                for (int s = 0; s < 4; s++)
                    z += factor_final[j * 4 + s] * tv_all[a][s];

                Scalar zk = pow_scalar(z, k);
                Scalar term = coeffs_final[j] * zk;

                C128 ct = to_c128(conj_scalar(term));
                local_adj_u_re += ct.real();
                local_adj_u_im += ct.imag();

                adj_coeffs_final[j] += adj_accum_a * conj_scalar(zk);

                Scalar adj_z(C128(0.0));
                if (k > 0 && abs_scalar(z) > 0) {
                    Scalar zk1 = pow_scalar(z, k - 1);
                    Scalar adj_zk = adj_accum_a * conj_scalar(coeffs_final[j]);
                    adj_z = adj_zk * conj_scalar(Scalar((double)k) * zk1);
                }

                for (int s = 0; s < 4; s++) {
                    adj_factor_final[j * 4 + s] += adj_z * conj_scalar(tv_all[a][s]);
                    my_adj_tv[s] += adj_z * conj_scalar(factor_final[j * 4 + s]);
                }
            }

            #pragma omp atomic
            adj_u_re += local_adj_u_re;
            #pragma omp atomic
            adj_u_im += local_adj_u_im;
        }

        adj_u_final[a] += Scalar(adj_raw) * Scalar(C128(adj_u_re, adj_u_im));

        for (int t = 0; t < n_threads; t++)
            for (int s = 0; s < 4; s++)
                adj_tv[a][s] += adj_tv_local[t * 4 + s];
    }

    // Extract grad_gamma from adj_u_final
    {
        C128 du[4];
        root_charge_weights_deriv(gammas[p - 1], du);
        double gg = 0.0;
        for (int a = 0; a < 4; a++)
            gg += to_real(conj_scalar(adj_u_final[a]) * Scalar(du[a]));
        grad_gammas[p - 1] += gg;
    }

    // adj_tv -> adj_M_final
    for (int s = 0; s < 4; s++)
        for (int a = 0; a < 4; a++) {
            Scalar contrib = adj_tv[a][s] * Scalar(C128(CHARGE_DIAG[a][s]));
            adj_M_final[0 * 4 + s] += contrib;
            adj_M_final[3 * 4 + s] -= contrib;
        }
    {
        C128 dM[16];
        doubled_mixer_deriv(betas[p - 1], dM);
        double gb = 0.0;
        for (int j = 0; j < 16; j++)
            gb += to_real(conj_scalar(adj_M_final[j]) * Scalar(dM[j]));
        grad_betas[p - 1] += gb;
    }

    // ── Backward through intermediate rounds ────────────────────
    std::vector<Scalar> adj_coeffs = adj_coeffs_final;
    std::vector<Scalar> adj_factor = adj_factor_final;

    for (int ell = p - 2; ell >= 0; ell--) {
        size_t R = states[ell].R;
        size_t vec_len = states[ell].vec_len;
        const auto& coeffs = states[ell].coeffs;
        const auto& factor = states[ell].factor;

        Scalar M[16], u[4];
        precompute_mixer_and_weights<Scalar>(betas[ell], gammas[ell], M, u);

        Scalar adj_u[4] = {};
        std::vector<Scalar> adj_coeffs_old(R, Scalar(C128(0.0)));
        for (int a = 0; a < 4; a++)
            for (size_t j = 0; j < R; j++) {
                Scalar ac = adj_coeffs[a * R + j];
                adj_u[a] += conj_scalar(coeffs[j]) * ac;
                adj_coeffs_old[j] += conj_scalar(u[a]) * ac;
            }

        {
            C128 du[4];
            root_charge_weights_deriv(gammas[ell], du);
            double gg = 0.0;
            for (int a = 0; a < 4; a++)
                gg += to_real(conj_scalar(adj_u[a]) * Scalar(du[a]));
            grad_gammas[ell] += gg;
        }

        size_t rest = vec_len / 16;
        std::vector<Scalar> adj_T(R * 16 * rest, Scalar(C128(0.0)));
        Scalar adj_M_wht[16] = {};

        wht_charge_contract_adjoint<Scalar>(M, factor.data(),
                                             adj_factor.data(),
                                             adj_T.data(), adj_M_wht,
                                             R, rest);

        {
            C128 dM[16];
            doubled_mixer_deriv(betas[ell], dM);
            double gb = 0.0;
            for (int j = 0; j < 16; j++)
                gb += to_real(conj_scalar(adj_M_wht[j]) * Scalar(dM[j]));
            grad_betas[ell] += gb;
        }

        adj_coeffs = std::move(adj_coeffs_old);
        adj_factor = std::move(adj_T);
    }

    return adj_factor;
}


// Explicit instantiations
template std::vector<C128> backward_root<C128>(double, const C128*, const double*, const double*, int, int, int, double*, double*);
template std::vector<DDComplex> backward_root<DDComplex>(double, const DDComplex*, const double*, const double*, int, int, int, double*, double*);

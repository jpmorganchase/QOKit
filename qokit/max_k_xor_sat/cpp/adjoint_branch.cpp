/**
 * Backward pass through branch contraction.
 * Templated on Scalar (C128 or DDComplex).
 */

#include "adjoint.h"
#include "primitives.h"
#include "dispatch.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif


// ── Forward replay helpers ──────────────────────────────────────

template<typename Scalar>
static std::vector<Scalar> replay_mode_products(const Scalar* F_after_power,
                                                 size_t total, int num_rounds,
                                                 const Scalar W[][16],
                                                 int up_to_ell) {
    if (up_to_ell == 0)
        return std::vector<Scalar>(F_after_power, F_after_power + total);

    std::vector<Scalar> buf1(F_after_power, F_after_power + total);
    std::vector<Scalar> buf2(total);
    Scalar* src = buf1.data();
    Scalar* dst = buf2.data();

    for (int ell = 0; ell < up_to_ell; ell++) {
        size_t prefix = 1;
        for (int i = 0; i < ell; i++) prefix *= 4;
        size_t suffix = 1;
        for (int i = ell + 1; i < num_rounds; i++) suffix *= 4;

        size_t n_items = prefix * 4 * suffix;
        #pragma omp parallel for schedule(static)
        for (size_t flat = 0; flat < n_items; flat++) {
            size_t p_idx = flat / (4 * suffix);
            size_t rem = flat % (4 * suffix);
            int h = (int)(rem / suffix);
            size_t s = rem % suffix;

            Scalar sum(C128(0.0));
            size_t base = p_idx * 4 * suffix + s;
            for (int j = 0; j < 4; j++)
                sum += W[ell][h * 4 + j] * src[base + j * suffix];
            dst[p_idx * 4 * suffix + h * suffix + s] = sum;
        }
        std::swap(src, dst);
    }

    if (src == buf1.data()) return buf1;
    return std::vector<Scalar>(src, src + total);
}


template<typename Scalar>
struct BranchForwardState {
    std::vector<Scalar> V_after_phase1;
    size_t n_ch;
    std::vector<Scalar> t_vec_after_phase2;
    std::vector<Scalar> t_vec_after_perm;
    double t_max;
    std::vector<Scalar> t_vec_normalized;
    std::vector<Scalar> F_after_power;
};


template<typename Scalar>
static BranchForwardState<Scalar> replay_branch_forward(
        const double* gammas, const double* betas,
        int num_rounds, int k,
        const Scalar* child_branch) {
    BranchForwardState<Scalar> state;
    int m = k - 1;

    std::vector<std::array<Scalar, 64>> md_flat;
    precompute_modified_mixers<Scalar>(betas, num_rounds, md_flat);

    int child_rounds = (child_branch != nullptr) ? num_rounds - 1 : 0;

    // Phase 1
    size_t total;
    std::vector<Scalar> V_vec;
    size_t n_ch;

    if (child_branch != nullptr && child_rounds >= 2) {
        size_t child_size = 1;
        for (int i = 0; i < child_rounds; i++) child_size *= 4;

        V_vec.resize(child_size);
        for (size_t i = 0; i < child_size; i++)
            V_vec[i] = child_branch[i] * 0.5;

        n_ch = 1;
        total = child_size;

        PingPong_T<Scalar> pp;
        pp.buf[0] = std::move(V_vec);
        pp.buf[1].resize(total);
        pp.cur = 0;

        for (int ell = 0; ell < child_rounds - 1; ell++) {
            size_t rest = total / (n_ch * 16);
            Scalar M_base[16];
            { C128 M[16]; doubled_mixer(betas[ell], M);
              for (int i = 0; i < 16; i++) M_base[i] = Scalar(M[i]); }
            wht_dispatch(M_base, pp.in(), pp.out(), n_ch, rest);
            n_ch *= 4;
            pp.flip();
        }

        V_vec = std::move(pp.buf[pp.cur]);
    } else if (child_branch != nullptr) {
        V_vec.resize(4);
        for (int i = 0; i < 4; i++)
            V_vec[i] = child_branch[i] * 0.5;
        n_ch = 1;
        total = 4;
    } else {
        V_vec.resize(4, Scalar(C128(0.5, 0.0)));
        n_ch = 1;
        total = 4;
    }

    state.V_after_phase1 = V_vec;
    state.n_ch = n_ch;

    // Phase 2
    int start_mv = std::max(child_rounds - 1, 0);

    Scalar trace_matrix[16];
    compute_trace_matrix<Scalar>(md_flat, num_rounds, trace_matrix);

    size_t t_total = 1;
    for (int i = 0; i < num_rounds; i++) t_total *= 4;

    std::vector<Scalar> t_vec(t_total);
    {
        struct Phase2 {
            const std::vector<std::array<Scalar, 64>>& md;
            int num_rounds;
            const Scalar* trace_mat;
            void run(const Scalar* V, size_t n_rows, int ell,
                     Scalar* out, size_t& out_size) {
                if (ell == num_rounds - 1) {
                    for (size_t i = 0; i < n_rows; i++)
                        for (int a = 0; a < 4; a++) {
                            Scalar sum(C128(0.0));
                            for (int s = 0; s < 4; s++)
                                sum += V[i * 4 + s] * trace_mat[s * 4 + a];
                            out[a * n_rows + i] = sum;
                        }
                    out_size = 4 * n_rows;
                    return;
                }
                size_t offset = 0;
                for (int a = 0; a < 4; a++) {
                    std::vector<Scalar> V_sub(n_rows * 4);
                    for (size_t i = 0; i < n_rows; i++)
                        for (int col = 0; col < 4; col++) {
                            Scalar sum(C128(0.0));
                            for (int s = 0; s < 4; s++)
                                sum += V[i * 4 + s] * md[ell][a * 16 + col * 4 + s];
                            V_sub[i * 4 + col] = sum;
                        }
                    size_t sub_size = 0;
                    run(V_sub.data(), n_rows, ell + 1, out + offset, sub_size);
                    offset += sub_size;
                }
                out_size = offset;
            }
        };
        Phase2 p2{md_flat, num_rounds, trace_matrix};
        size_t actual_size = 0;
        p2.run(V_vec.data(), n_ch, start_mv, t_vec.data(), actual_size);
    }
    state.t_vec_after_phase2 = t_vec;

    // Axis permutation
    int remaining = num_rounds - start_mv;
    if (num_rounds > 1) {
        std::vector<int> perm(num_rounds);
        int idx = 0;
        for (int i = num_rounds - 1; i >= remaining; i--) perm[idx++] = i;
        for (int i = 0; i < remaining; i++) perm[idx++] = i;

        std::vector<Scalar> t_perm(t_total);
        std::vector<size_t> strides(num_rounds);
        strides[num_rounds - 1] = 1;
        for (int i = num_rounds - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * 4;

        #pragma omp parallel for schedule(static)
        for (size_t flat = 0; flat < t_total; flat++) {
            size_t rem = flat;
            size_t multi[32];
            for (int d = 0; d < num_rounds; d++) {
                multi[d] = rem / strides[d];
                rem %= strides[d];
            }
            size_t dst_flat = 0;
            for (int d = 0; d < num_rounds; d++)
                dst_flat += multi[perm[d]] * strides[d];
            t_perm[dst_flat] = t_vec[flat];
        }
        t_vec = std::move(t_perm);
    }
    state.t_vec_after_perm = t_vec;

    // Normalization + power
    state.t_max = 0.0;
    if (m > 1) {
        double t_max = 0.0;
        #pragma omp parallel for reduction(max:t_max) schedule(static)
        for (size_t i = 0; i < t_total; i++) {
            double mag = abs_scalar(t_vec[i]);
            if (mag > t_max) t_max = mag;
        }
        state.t_max = t_max;
        if (t_max > 0) {
            double inv = 1.0 / t_max;
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < t_total; i++)
                t_vec[i] = t_vec[i] * inv;
        }
    }
    state.t_vec_normalized = t_vec;

    std::vector<Scalar> F(t_total);
    if (m == 1) {
        F = t_vec;
    } else {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < t_total; i++)
            F[i] = pow_scalar(t_vec[i], m);
    }
    state.F_after_power = F;

    return state;
}


// ── Main backward_branch ────────────────────────────────────────

template<typename Scalar>
std::vector<Scalar> backward_branch(const Scalar* adj_F,
                                     const double* gammas, const double* betas,
                                     int num_rounds, int k,
                                     const Scalar* child_branch,
                                     double* grad_gammas, double* grad_betas) {
    int m = k - 1;
    size_t t_total = 1;
    for (int i = 0; i < num_rounds; i++) t_total *= 4;

    auto fwd = replay_branch_forward<Scalar>(gammas, betas, num_rounds, k, child_branch);

    // Precompute charge weight matrices
    std::vector<std::array<Scalar, 16>> W;
    precompute_charge_weights<Scalar>(gammas, num_rounds, W);

    // ── Step 5 backward: mode products ──────────────────────────
    std::vector<Scalar> adj_cur(adj_F, adj_F + t_total);
    std::vector<Scalar> adj_src(t_total);

    for (int ell = num_rounds - 1; ell >= 0; ell--) {
        auto mp_input = replay_mode_products<Scalar>(
            fwd.F_after_power.data(), t_total, num_rounds,
            (const Scalar(*)[16])W.data(), ell);

        Scalar adj_W_ell[16] = {};
        mode_product_adjoint<Scalar>(W[ell].data(), mp_input.data(),
                                      adj_cur.data(), adj_src.data(), adj_W_ell,
                                      t_total, ell, num_rounds);

        C128 dW[16];
        charge_weight_matrix_deriv(gammas[ell], dW);
        double g = 0.0;
        for (int j = 0; j < 16; j++)
            g += to_real(conj_scalar(adj_W_ell[j]) * Scalar(dW[j]));
        grad_gammas[ell] += g;

        adj_cur = adj_src;
    }

    // ── Step 4 backward: entrywise power ────────────────────────
    std::vector<Scalar> adj_t_norm(t_total, Scalar(C128(0.0)));
    if (m == 0) {
        // constant — no gradient
    } else if (m == 1) {
        adj_t_norm = adj_cur;
    } else {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < t_total; i++) {
            Scalar tn = fwd.t_vec_normalized[i];
            if (abs_scalar(tn) > 0) {
                Scalar deriv = Scalar((double)m) * pow_scalar(tn, m - 1);
                adj_t_norm[i] = adj_cur[i] * conj_scalar(deriv);
            }
        }
    }

    // Normalization backward
    std::vector<Scalar> adj_t_perm(t_total);
    if (m > 1 && fwd.t_max > 0) {
        double inv_tmax = 1.0 / fwd.t_max;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < t_total; i++)
            adj_t_perm[i] = adj_t_norm[i] * inv_tmax;
    } else {
        adj_t_perm = adj_t_norm;
    }

    // ── Step 3 backward: axis permutation ───────────────────────
    int child_rounds = (child_branch != nullptr) ? num_rounds - 1 : 0;
    int start_mv = std::max(child_rounds - 1, 0);
    int remaining = num_rounds - start_mv;

    std::vector<Scalar> adj_t_phase2(t_total);
    if (num_rounds > 1) {
        std::vector<int> perm(num_rounds);
        int idx = 0;
        for (int i = num_rounds - 1; i >= remaining; i--) perm[idx++] = i;
        for (int i = 0; i < remaining; i++) perm[idx++] = i;

        std::vector<size_t> strides(num_rounds);
        strides[num_rounds - 1] = 1;
        for (int i = num_rounds - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * 4;

        #pragma omp parallel for schedule(static)
        for (size_t flat = 0; flat < t_total; flat++) {
            size_t rem = flat;
            size_t multi[32];
            for (int d = 0; d < num_rounds; d++) {
                multi[d] = rem / strides[d];
                rem %= strides[d];
            }
            size_t dst_flat = 0;
            for (int d = 0; d < num_rounds; d++)
                dst_flat += multi[perm[d]] * strides[d];
            adj_t_phase2[flat] = adj_t_perm[dst_flat];
        }
    } else {
        adj_t_phase2 = adj_t_perm;
    }

    // ── Step 2 backward: Phase 2 ────────────────────────────────
    std::vector<std::array<Scalar, 64>> md_flat;
    precompute_modified_mixers<Scalar>(betas, num_rounds, md_flat);

    Scalar trace_matrix[16];
    compute_trace_matrix<Scalar>(md_flat, num_rounds, trace_matrix);

    std::vector<std::array<Scalar, 64>> adj_md_flat(num_rounds);
    for (int ell = 0; ell < num_rounds; ell++)
        adj_md_flat[ell].fill(Scalar(C128(0.0)));

    struct Phase2Backward {
        const std::vector<std::array<Scalar, 64>>& md;
        std::vector<std::array<Scalar, 64>>& adj_md;
        int num_rounds;
        const Scalar* trace_mat;
        Scalar adj_trace_mat[16];

        void run(const Scalar* V, size_t n_rows, int ell,
                 const Scalar* adj_out, Scalar* adj_V) {
            if (ell == num_rounds - 1) {
                for (size_t i = 0; i < n_rows; i++)
                    for (int s = 0; s < 4; s++) {
                        Scalar sum(C128(0.0));
                        for (int a = 0; a < 4; a++) {
                            sum += adj_out[a * n_rows + i] * conj_scalar(trace_mat[s * 4 + a]);
                            adj_trace_mat[s * 4 + a] +=
                                conj_scalar(V[i * 4 + s]) * adj_out[a * n_rows + i];
                        }
                        adj_V[i * 4 + s] += sum;
                    }
                return;
            }

            size_t sub_out_size = n_rows;
            for (int ll = ell + 1; ll < num_rounds; ll++) sub_out_size *= 4;

            size_t offset = 0;
            for (int a = 0; a < 4; a++) {
                std::vector<Scalar> V_sub(n_rows * 4);
                for (size_t i = 0; i < n_rows; i++)
                    for (int col = 0; col < 4; col++) {
                        Scalar sum(C128(0.0));
                        for (int s = 0; s < 4; s++)
                            sum += V[i * 4 + s] * md[ell][a * 16 + col * 4 + s];
                        V_sub[i * 4 + col] = sum;
                    }

                std::vector<Scalar> adj_V_sub(n_rows * 4, Scalar(C128(0.0)));
                run(V_sub.data(), n_rows, ell + 1,
                    adj_out + offset, adj_V_sub.data());

                for (size_t i = 0; i < n_rows; i++)
                    for (int s = 0; s < 4; s++) {
                        Scalar sum(C128(0.0));
                        for (int col = 0; col < 4; col++) {
                            sum += adj_V_sub[i * 4 + col] *
                                   conj_scalar(md[ell][a * 16 + col * 4 + s]);
                            adj_md[ell][a * 16 + col * 4 + s] +=
                                conj_scalar(V[i * 4 + s]) * adj_V_sub[i * 4 + col];
                        }
                        adj_V[i * 4 + s] += sum;
                    }
                offset += sub_out_size;
            }
        }
    };

    std::vector<Scalar> adj_V(fwd.V_after_phase1.size(), Scalar(C128(0.0)));
    Phase2Backward p2b{md_flat, adj_md_flat, num_rounds, trace_matrix, {}};
    std::memset(p2b.adj_trace_mat, 0, sizeof(p2b.adj_trace_mat));
    p2b.run(fwd.V_after_phase1.data(), fwd.n_ch, start_mv,
            adj_t_phase2.data(), adj_V.data());

    // Extract beta gradients from adj_md_flat and adj_trace_mat
    for (int ell = 0; ell < num_rounds; ell++) {
        Scalar adj_M_ell[16] = {};
        for (int a = 0; a < 4; a++)
            for (int row = 0; row < 4; row++)
                for (int col = 0; col < 4; col++)
                    adj_M_ell[row * 4 + col] +=
                        adj_md_flat[ell][a * 16 + row * 4 + col] * Scalar(C128(CHARGE_DIAG[a][col]));

        if (ell == num_rounds - 1) {
            for (int s = 0; s < 4; s++)
                for (int a = 0; a < 4; a++) {
                    Scalar contrib = p2b.adj_trace_mat[s * 4 + a] * Scalar(C128(CHARGE_DIAG[a][s]));
                    adj_M_ell[0 * 4 + s] += contrib;
                    adj_M_ell[3 * 4 + s] += contrib;
                }
        }

        C128 dM[16];
        doubled_mixer_deriv(betas[ell], dM);
        double gb = 0.0;
        for (int j = 0; j < 16; j++)
            gb += to_real(conj_scalar(adj_M_ell[j]) * Scalar(dM[j]));
        grad_betas[ell] += gb;
    }

    // ── Step 1 backward: Phase 1 ────────────────────────────────
    if (child_branch == nullptr)
        return {};

    if (child_rounds < 2) {
        std::vector<Scalar> adj_child(4);
        for (int i = 0; i < 4; i++)
            adj_child[i] = adj_V[i] * 0.5;
        return adj_child;
    }

    size_t child_size = 1;
    for (int i = 0; i < child_rounds; i++) child_size *= 4;

    std::vector<Scalar> V_init(child_size);
    for (size_t i = 0; i < child_size; i++)
        V_init[i] = child_branch[i] * 0.5;

    // Replay Phase 1 forward
    std::vector<std::vector<Scalar>> phase1_states;
    phase1_states.push_back(V_init);

    {
        size_t n_ch_fwd = 1;
        PingPong_T<Scalar> pp;
        pp.buf[0] = V_init;
        pp.buf[1].resize(child_size);
        pp.cur = 0;

        for (int ell = 0; ell < child_rounds - 1; ell++) {
            size_t rest = child_size / (n_ch_fwd * 16);
            Scalar M_base[16];
            { C128 M[16]; doubled_mixer(betas[ell], M);
              for (int i = 0; i < 16; i++) M_base[i] = Scalar(M[i]); }
            wht_dispatch(M_base, pp.in(), pp.out(), n_ch_fwd, rest);
            n_ch_fwd *= 4;
            pp.flip();
            phase1_states.push_back(
                std::vector<Scalar>(pp.in(), pp.in() + child_size));
        }
    }

    // Backward through Phase 1
    std::vector<Scalar> adj_phase1 = adj_V;
    size_t n_ch_bwd = fwd.n_ch;

    for (int ell = child_rounds - 2; ell >= 0; ell--) {
        n_ch_bwd /= 4;
        size_t rest = child_size / (n_ch_bwd * 16);

        Scalar M_base[16];
        { C128 M[16]; doubled_mixer(betas[ell], M);
          for (int i = 0; i < 16; i++) M_base[i] = Scalar(M[i]); }

        std::vector<Scalar> adj_T(child_size, Scalar(C128(0.0)));
        Scalar adj_M_wht[16] = {};

        wht_charge_contract_adjoint<Scalar>(M_base, phase1_states[ell].data(),
                                             adj_phase1.data(),
                                             adj_T.data(), adj_M_wht,
                                             n_ch_bwd, rest);

        C128 dM[16];
        doubled_mixer_deriv(betas[ell], dM);
        double gb = 0.0;
        for (int j = 0; j < 16; j++)
            gb += to_real(conj_scalar(adj_M_wht[j]) * Scalar(dM[j]));
        grad_betas[ell] += gb;

        adj_phase1 = adj_T;
    }

    std::vector<Scalar> adj_child(child_size);
    for (size_t i = 0; i < child_size; i++)
        adj_child[i] = adj_phase1[i] * 0.5;

    return adj_child;
}


// Explicit instantiations
template std::vector<C128> backward_branch<C128>(const C128*, const double*, const double*, int, int, const C128*, double*, double*);
template std::vector<DDComplex> backward_branch<DDComplex>(const DDComplex*, const double*, const double*, int, int, const DDComplex*, double*, double*);

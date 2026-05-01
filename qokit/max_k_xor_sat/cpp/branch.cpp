/**
 * Branch contraction: Phase 1 (WHT), Phase 2 (iterative with trace),
 * mode products (ping-pong).
 *
 * Templated on Scalar (C128 or DDComplex) to share the algorithm between
 * float64 and double-double precision paths.
 */

#include "branch.h"
#include "dispatch.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <type_traits>  // for std::is_same
#ifdef _OPENMP
#include <omp.h>
#endif


// ── Float64 mode products ─────────────────────────────────────

static void mode_products(C128* F, size_t total, int num_rounds,
                          const C128 W[][16], C128* workspace = nullptr) {
    std::vector<C128> buf_owned;
    C128* buf_ptr;
    if (workspace) {
        buf_ptr = workspace;
    } else {
        buf_owned.resize(total);
        hint_huge_pages(buf_owned.data(), total * sizeof(C128));
        buf_ptr = buf_owned.data();
    }
    C128* src = F;
    C128* dst = buf_ptr;

    size_t suffix_arr[32];
    suffix_arr[num_rounds - 1] = 1;
    for (int i = num_rounds - 2; i >= 0; i--) suffix_arr[i] = suffix_arr[i + 1] * 4;

    for (int ell = 0; ell < num_rounds; ell++) {
        size_t suffix = suffix_arr[ell];
        size_t stride = 4 * suffix;

        #pragma omp parallel for schedule(static)
        for (size_t flat = 0; flat < total; flat++) {
            size_t p_idx = flat / stride;
            size_t rem = flat % stride;
            int h = (int)(rem / suffix);
            size_t s = rem % suffix;

            C128 sum(0.0);
            size_t base = p_idx * stride + s;
            for (int j = 0; j < 4; j++)
                sum += W[ell][h * 4 + j] * src[base + j * suffix];
            dst[p_idx * stride + h * suffix + s] = sum;
        }
        std::swap(src, dst);
    }

    if (src != F) std::memcpy(F, src, total * sizeof(C128));
}

// Mode products dispatch
static void mode_products_dispatch(C128* F, size_t total, int num_rounds,
                                   const std::array<C128, 16>* W, C128* ws) {
    mode_products(F, total, num_rounds, (const C128(*)[16])W, ws);
}
static void mode_products_dispatch(DDComplex* F, size_t total, int num_rounds,
                                   const std::array<DDComplex, 16>* W, DDComplex* ws) {
    mode_products_dd_fused(F, total, num_rounds,
                           reinterpret_cast<const DDComplex(*)[16]>(W), ws);
}


// ── Templated branch contraction ──────────────────────────────

template<typename Scalar>
static std::vector<Scalar> hyperedge_branch_impl(
        const double* gammas, const double* betas,
        int num_rounds, int k,
        const Scalar* child_branch, bool verbose) {

    assert(num_rounds >= 0 && num_rounds <= 32);
    int m = k - 1;

    std::vector<std::array<Scalar, 64>> md_flat;
    precompute_modified_mixers(betas, num_rounds, md_flat);

    std::vector<std::array<Scalar, 16>> W;
    precompute_charge_weights(gammas, num_rounds, W);

    int child_rounds = (child_branch != nullptr) ? num_rounds - 1 : 0;

    // ── Phase 1: WHT iterations consuming child branch ──
    size_t n_ch;
    std::vector<Scalar> V_vec;

    if (child_branch != nullptr && child_rounds >= 2) {
        size_t child_size = 1;
        for (int i = 0; i < child_rounds; i++) child_size *= 4;

        V_vec.resize(child_size);
        hint_alloc(V_vec.data(), child_size * sizeof(Scalar));
        for (size_t i = 0; i < child_size; i++)
            V_vec[i] = Scalar(0.5) * child_branch[i];

        n_ch = 1;
        size_t total = child_size;
        std::vector<Scalar> buf2(total);
        hint_alloc(buf2.data(), total * sizeof(Scalar));
        first_touch_zero(buf2.data(), total);
        Scalar* src = V_vec.data();
        Scalar* dst = buf2.data();

        for (int ell = 0; ell < child_rounds - 1; ell++) {
            size_t rest = total / (n_ch * 16);
            Scalar M_arr[16];
            { C128 M_f64[16]; doubled_mixer(betas[ell], M_f64);
              for (int i = 0; i < 16; i++) M_arr[i] = Scalar(M_f64[i]); }
            wht_dispatch(M_arr, src, dst, n_ch, rest);
            n_ch *= 4;
            std::swap(src, dst);
        }
        if (src != V_vec.data()) V_vec.assign(src, src + total);
    } else if (child_branch != nullptr) {
        V_vec.resize(4);
        for (int i = 0; i < 4; i++) V_vec[i] = Scalar(0.5) * child_branch[i];
        n_ch = 1;
    } else {
        V_vec.resize(4, Scalar(0.5));
        n_ch = 1;
    }

    // ── Phase 2 with trace ──
    int start_mv = std::max(child_rounds - 1, 0);
    Scalar trace_matrix[16];
    compute_trace_matrix(md_flat, num_rounds, trace_matrix);

    size_t t_total = 1;
    for (int i = 0; i < num_rounds; i++) t_total *= 4;
    std::vector<Scalar> t_vec(t_total);
    hint_alloc(t_vec.data(), t_total * sizeof(Scalar));

    // Pre-allocate Phase 2 workspace
    int remaining = num_rounds - start_mv;
    int ws_depth = remaining - 1;
    size_t ws_stride = n_ch * 4;
    std::vector<Scalar> ws_buf;
    if (ws_depth > 0) {
        ws_buf.resize((size_t)ws_depth * ws_stride);
        hint_alloc(ws_buf.data(), ws_buf.size() * sizeof(Scalar));
    }

    struct Phase2 {
        const std::vector<std::array<Scalar, 64>>& md;
        int num_rounds;
        const Scalar* trace_mat;
        Scalar* workspace;
        size_t ws_stride;

        void run(const Scalar* V, size_t n_rows, int ell,
                 Scalar* out, size_t& out_size, int depth) {
            if (ell == num_rounds - 1) {
                for (size_t i = 0; i < n_rows; i++)
                    for (int a = 0; a < 4; a++) {
                        Scalar sum{};
                        for (int s = 0; s < 4; s++)
                            sum += V[i * 4 + s] * trace_mat[s * 4 + a];
                        out[a * n_rows + i] = sum;
                    }
                out_size = 4 * n_rows;
                return;
            }
            Scalar* V_sub = workspace + (size_t)depth * ws_stride;
            size_t offset = 0;
            for (int a = 0; a < 4; a++) {
                for (size_t i = 0; i < n_rows; i++)
                    for (int col = 0; col < 4; col++) {
                        Scalar sum{};
                        for (int s = 0; s < 4; s++)
                            sum += V[i * 4 + s] * md[ell][a * 16 + col * 4 + s];
                        V_sub[i * 4 + col] = sum;
                    }
                size_t sub_size = 0;
                run(V_sub, n_rows, ell + 1, out + offset, sub_size, depth + 1);
                offset += sub_size;
            }
            out_size = offset;
        }
    };

    Phase2 p2{md_flat, num_rounds, trace_matrix, ws_buf.data(), ws_stride};
    size_t actual_size = 0;
    p2.run(V_vec.data(), n_ch, start_mv, t_vec.data(), actual_size, 0);

    // Free Phase 1/2 workspace before axis permutation allocates
    { std::vector<Scalar>().swap(V_vec); }
    { std::vector<Scalar>().swap(ws_buf); }

    // ── Axis permutation ──
    if (num_rounds > 1) {
        std::vector<int> perm(num_rounds);
        int idx = 0;
        for (int i = num_rounds - 1; i >= remaining; i--) perm[idx++] = i;
        for (int i = 0; i < remaining; i++) perm[idx++] = i;

        std::vector<Scalar> t_perm(t_total);
        hint_alloc(t_perm.data(), t_total * sizeof(Scalar));
        std::vector<size_t> strides(num_rounds);
        strides[num_rounds - 1] = 1;
        for (int i = num_rounds - 2; i >= 0; i--) strides[i] = strides[i + 1] * 4;

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

    // ── Entrywise (k-1) power with normalization (in-place) ──
    if (m > 1)
        constexpr bool dd_pow = std::is_same<Scalar, C128>::value;
        normalize_and_pow(t_vec.data(), t_vec.data(), t_total, m, dd_pow);
    // Now t_vec holds the powered tensor. For m==1, t_vec is already correct.

    // ── Mode products (need a workspace buffer) ──
    std::vector<Scalar> ws_mp(t_total);
    hint_alloc(ws_mp.data(), t_total * sizeof(Scalar));
    mode_products_dispatch(t_vec.data(), t_total, num_rounds, W.data(), ws_mp.data());

    return t_vec;
}


// ── Public API ────────────────────────────────────────────────

std::vector<C128> hyperedge_branch(const double* gammas, const double* betas,
                                    int num_rounds, int k,
                                    const C128* child_branch, bool verbose) {
    return hyperedge_branch_impl(gammas, betas, num_rounds, k, child_branch, verbose);
}

std::vector<DDComplex> hyperedge_branch_dd(const double* gammas, const double* betas,
                                           int num_rounds, int k,
                                           const DDComplex* child_branch, bool verbose) {
    return hyperedge_branch_impl(gammas, betas, num_rounds, k, child_branch, verbose);
}

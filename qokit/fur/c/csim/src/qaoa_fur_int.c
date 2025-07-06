/* ─── qaoa_fur_int.c  (FULL FILE) ─────────────────────────────────────────── */
#include "qaoa_fur_int.h"
#include "fur_int.h"
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

/* Uncomment ↓ to get per-layer sample prints */
// #define DEBUG_PRINT 1

/*  fully-quantised int16 QAOA (FUR-X mixer)  */
void apply_qaoa_furx_int(
        int16_t      *r,         /* real  part (quantised) */
        int16_t      *i,         /* imag  part (quantised) */
        float        *scales,    /* per-block scales        */
        unsigned      bits,      /* 4/6/8/16 …              */
        const double *gammas,
        const double *betas,
        const double *diag,
        unsigned      n_q,
        size_t        n_states,
        size_t        p)
{
    const int int_max = (1 << (bits - 1)) - 1;        /*  7/31/127/32 767  */
    const size_t BLOCK = 1024;
    const size_t n_blocks = (n_states + BLOCK - 1) / BLOCK;

    printf("▶ ENTER apply_qaoa_furx_int : n_states=%zu  layers=%zu  bits=%u\n",
           n_states, p, bits);
    fflush(stdout);

    /* ----------------  MAIN QAOA LOOP  ---------------- */
    for (size_t l = 0; l < p; ++l)
    {
        const double γ  = gammas[l];
        const double β  = betas [l];
        printf("  └─ LAYER %zu : γ=%+.6f  β=%+.6f\n", l, γ, β);

        /* ---- Phase separator (diagonal) ---- */
        for (size_t j = 0; j < n_states; ++j)
        {
            /* --- fetch scale --- */
            const size_t blk = j / BLOCK;
            float scale = (blk < n_blocks) ? scales[blk] : 1.0f;
            if (!isfinite(scale) || scale < 1e-12f) scale = 1e-3f;

            /* --- de-quantise current amp --- */
            float re = (float)r[j] / int_max * scale;
            float im = (float)i[j] / int_max * scale;

            /* --- apply exp(-i γ H) --- */
            const float ang = (float)(γ * diag[j]);
            const float c = cosf(ang), s = sinf(ang);

            const float re2 = c * re - s * im;
            const float im2 = s * re + c * im;

            /* --- re-quantise (with clamping) --- */
            const float qre = fminf( fmaxf(re2 / scale, -1.f), 1.f);
            const float qim = fminf( fmaxf(im2 / scale, -1.f), 1.f);

            r[j] = (int16_t)lrintf(qre * int_max);
            i[j] = (int16_t)lrintf(qim * int_max);
        }

        /* ---- Optional debug sample ---- */
    #ifdef DEBUG_PRINT
        for (size_t k = 0; k < 4 && k < n_states; ++k)
            printf("    · amp[%zu] = {%d,%d}\n", k, r[k], i[k]);
    #endif

        /* ---- Mixer (only if β ≠ 0) ---- */
        if (fabs(β) > 1e-7)
            furx_all_int(r, i, scales, (float)β, n_q, n_states, bits);
        else
            printf("    ℹ️  skipped mixer (β≈0)\n");
    }

    /* -------------------------------------------------- */
    /* final sanity: count non-zero amps                  */
    size_t nnz = 0;
    for (size_t j = 0; j < n_states; ++j)
        if (r[j] || i[j]) ++nnz;

    printf("ℹ️  Non-zero amplitudes (quantised) : %zu / %zu\n", nnz, n_states);
    printf("✅ apply_qaoa_furx_int finished\n");
    fflush(stdout);
}
/* ─────────────────────────────────────────────────────────────────────────── */

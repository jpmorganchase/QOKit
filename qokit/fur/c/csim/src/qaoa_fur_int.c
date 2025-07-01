/* ─── qaoa_fur_int.c ────────────────────────────────────────── */
#include "qaoa_fur_int.h"
#include "fur_int.h"
#include "quant_fp.h"

/* identical signature to fp version but extra bits + scales */
void apply_qaoa_furx_int(int16_t *r, int16_t *i,
                         float   *scales,
                         unsigned bits,
                         double *gam, double *bet,
                         const double *diag,
                         unsigned n_q, size_t n_states, size_t p)
{
    for(size_t l=0;l<p;l++){
        /* phase separator still fp ⇒ decode / encode in-place */
        /* decode current block, multiply by exp(-i*γH), encode back ... */
        /* or keep it fp for now – negligible cost vs mixer          */

        furx_all_int(r, i, scales, bet[l], n_q, n_states, bits);
    }
}

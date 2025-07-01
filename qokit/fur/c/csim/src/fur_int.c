/* ─── fur_int.c (excerpt) ───────────────────────────────────── */
#include "fur_int.h"
#include "quant_fp.h"
#include <math.h>
#include <omp.h>

void furx_all_int(int16_t *r, int16_t *i,
                  const float *sc, double theta,
                  unsigned n_q, size_t n_states, int bits)
{
    for(size_t base=0, blk=0; base<n_states; base+=Q_BLOCK, blk++){
        /* 1⃣  decode block to float tmp */
        float tr[Q_BLOCK], ti[Q_BLOCK];
        qfp_decode_block(r+base, i+base, tr, ti, sc[blk], bits);

        /* 2⃣  apply same double-precision rotation kernel
               we already have (copy/paste from original furx_all)
               but for exactly n_q qubits */
        /* ...            */

        /* 3⃣  encode back */
        qfp_encode_block(tr, ti, r+base, i+base,
                         (float*)(sc+blk), bits);   /* re-compute scale */
    }
}

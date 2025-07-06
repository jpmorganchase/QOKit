/* ------------------------------------------------------------------ */
/*  Fixed-point X-mixer in INT8/INT16 for every qubit (β-rotation)    */
/* ------------------------------------------------------------------ */

#include <math.h>
#include <stdint.h>   /* <—  missing before */
#include <stddef.h>   /* <—  missing before */
#include <omp.h>

/* signature unchanged so the rest of the code still links */
void furx_all_int(int16_t *r, int16_t *i,
                  const float * /*scales _unused_*/,
                  float beta,
                  unsigned n_q,
                  size_t n_states,
                  unsigned bits)
{
    const float c       = cosf(beta);
    const float s       = sinf(beta);
    const int   int_max = (1 << (bits - 1)) - 1;   /* 127, 32 767, … */

    for (unsigned q = 0; q < n_q; ++q) {

        size_t stride = 1ull << q;
        size_t step   = stride << 1;

        /* (ia, ib) differ only in bit q */
        #pragma omp parallel for schedule(static)
        for (size_t base = 0; base < n_states; base += step) {
            for (size_t off = 0; off < stride; ++off) {

                size_t ia = base + off;
                size_t ib = ia   + stride;

                /* --- work in raw integers, no /scale --- */
                float ar = r[ia], ai = i[ia];
                float br = r[ib], bi = i[ib];

                float new_ar = c*ar - s*ai;
                float new_ai = s*ar + c*ai;
                float new_br = c*br - s*bi;
                float new_bi = s*br + c*bi;

                /* clamp to representable range */
                new_ar = fmaxf(-int_max, fminf(int_max, new_ar));
                new_ai = fmaxf(-int_max, fminf(int_max, new_ai));
                new_br = fmaxf(-int_max, fminf(int_max, new_br));
                new_bi = fmaxf(-int_max, fminf(int_max, new_bi));

                r[ia] = (int16_t)lrintf(new_ar);
                i[ia] = (int16_t)lrintf(new_ai);
                r[ib] = (int16_t)lrintf(new_br);
                i[ib] = (int16_t)lrintf(new_bi);
            }
        }
    }
}

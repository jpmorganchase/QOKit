/***************************************
* // SPDX-License-Identifier: Apache-2.0
* // Copyright : JP Morgan Chase & Co
****************************************/ 
#include "quant_fp.h"
#include <math.h>

static inline int16_t _imax(int bits){ return (1<<(bits-1))-1; }

/* encode a Q_BLOCK-size slice */
void qfp_encode_block(const float *r,const float *i,
                      int16_t *qr,int16_t *qi,
                      float *scale,int bits)
{
    int16_t imax = _imax(bits);
    /* find max |amp| in block */
    float m = 1e-12f;
    for(size_t k=0;k<Q_BLOCK;k++){
        float a = fabsf(r[k]), b = fabsf(i[k]);
        if(a>b){ if(a>m) m=a; } else { if(b>m) m=b; }
    }
    *scale = m;
    float s = imax / m;
#pragma omp simd
    for(size_t k=0;k<Q_BLOCK;k++){
        qr[k] = (int16_t) lrintf(r[k]*s);
        qi[k] = (int16_t) lrintf(i[k]*s);
    }
}

void qfp_decode_block(const int16_t *qr,const int16_t *qi,
                      float *r,float *i,
                      float scale,int bits)
{
    float inv = scale / _imax(bits);
#pragma omp simd
    for(size_t k=0;k<Q_BLOCK;k++){
        r[k] = qr[k]*inv;
        i[k] = qi[k]*inv;
    }
}

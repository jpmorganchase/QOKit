/***************************************
* // SPDX-License-Identifier: Apache-2.0
* // Copyright : JP Morgan Chase & Co
****************************************/ 
#pragma once
#include <stdint.h>
#include <stddef.h>

#define Q_BLOCK 1024                 /* must match Python side   */

/* Fixed-point encode  (complex64 → int{8,16})  */
void qfp_encode_block(const float *r, const float *i,
                      int16_t *qr, int16_t *qi,
                      float    *scale,
                      int bits);

/* Fixed-point decode  (int{8,16} → complex64)  */
void qfp_decode_block(const int16_t *qr, const int16_t *qi,
                      float *r, float *i,
                      float  scale,
                      int    bits);

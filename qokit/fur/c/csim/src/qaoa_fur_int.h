/***************************************
* // SPDX-License-Identifier: Apache-2.0
* // Copyright : JP Morgan Chase & Co
****************************************/ 
#pragma once
#include <stdint.h>
#include <stddef.h>

#define Q_BLOCK 1024                 /* must match Python side   */

void apply_qaoa_furx_int(int16_t *r, int16_t *i,
                         float   *scales,
                         unsigned bits,
                         double *gam, double *bet,
                         const double *diag,
                         unsigned n_q, size_t n_states, size_t p);

/***************************************
* // SPDX-License-Identifier: Apache-2.0
* // Copyright : JP Morgan Chase & Co
****************************************/ 
#pragma once
#include <stdint.h>
#include <stddef.h>

#define Q_BLOCK 1024                 /* must match Python side   */

/* qaoa_fur_int.h */
void apply_qaoa_furx_int(
    int16_t      *r,
    int16_t      *i,
    float        *scales,
    unsigned      bits,
    const double *gammas,          /* ← add const */
    const double *betas,           /* ← add const */
    const double *diag,
    unsigned      n_q,
    size_t        n_states,
    size_t        p);

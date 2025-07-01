/* ─── fur_int.h ─────────────────────────────────────────────── */
#pragma once
#include <stdint.h>
#include <stddef.h>
void furx_all_int(int16_t *r, int16_t *i,
                  const float *scales,
                  double theta,
                  unsigned n_qubits, size_t n_states,
                  int bits);

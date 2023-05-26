#pragma once
#include <stddef.h>

void furx(double* a_real, double* a_imag, double theta, unsigned int q, size_t n_states);

void furx_all(double* a_real, double* a_imag, double theta, unsigned int n_qubits, size_t n_states);

void furxy(double* a_real, double* a_imag, double theta, unsigned int q1, unsigned int q2, size_t n_states);

void furxy_ring(double* a_real, double* a_imag, double theta, unsigned int n_qubits, size_t n_states);

void furxy_complete(double* a_real, double* a_imag, double theta, unsigned int n_qubits, size_t n_states);

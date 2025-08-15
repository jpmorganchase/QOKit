/***************************************
* // SPDX-License-Identifier: Apache-2.0
* // Copyright : JP Morgan Chase & Co
****************************************/ 

#pragma once
#include <stddef.h>
#include <complex.h>

void furx(double* a_real, double* a_imag, double theta, unsigned int q, size_t n_states);

void furx_all(double* a_real, double* a_imag, double theta, unsigned int n_qubits, size_t n_states);

void c_furx_qudit(double* x_real, double* x_imag, double theta, double* A_mat_real, double* A_mat_imag, int* target_qubits, int target_qubits_len, int n_qubits);

void c_furx_all_qudit(
    double* x_real,
    double* x_imag, 
    double theta, 
    double* A_mat_real, 
    double* A_mat_imag,
    int n_precision,
    int n_qubits);
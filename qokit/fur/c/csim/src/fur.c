/***************************************
* // SPDX-License-Identifier: Apache-2.0
* // Copyright : JP Morgan Chase & Co
****************************************/ 
#ifdef _OPENMP
#include <omp.h>
#endif
#include "fur.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>  // For malloc and free
#include <string.h>  // For memcpy


// Complex matrix multiplication
static void complex_matrix_multiply(double* A_real, double* A_imag, double* B_real, double* B_imag, 
                                  double* C_real, double* C_imag, int dim) {
    // Initialize C to zero
    for (int i = 0; i < dim * dim; i++) {
        C_real[i] = 0.0;
        C_imag[i] = 0.0;
    }
    
    // Perform complex matrix multiplication: C = A * B
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
                C_real[i * dim + j] += A_real[i * dim + k] * B_real[k * dim + j] - 
                                     A_imag[i * dim + k] * B_imag[k * dim + j];
                C_imag[i * dim + j] += A_real[i * dim + k] * B_imag[k * dim + j] + 
                                     A_imag[i * dim + k] * B_real[k * dim + j];
            }
        }
    }
}

// Compute matrix exponential using Taylor series expansion
static void compute_matrix_exponential(double* A_real, double* A_imag, double theta, 
                                    double* exp_real, double* exp_imag, int dim) {
    // Initialize exp matrix as identity
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            if (i == j) {
                exp_real[i * dim + j] = 1.0;
                exp_imag[i * dim + j] = 0.0;
            } else {
                exp_real[i * dim + j] = 0.0;
                exp_imag[i * dim + j] = 0.0;
            }
        }
    }
    
    // Make a copy of -i*theta*A
    double* iA_real = (double*)malloc(dim * dim * sizeof(double));
    double* iA_imag = (double*)malloc(dim * dim * sizeof(double));
    
    // Calculate -i*theta*A = theta * (-i*A)
    // Real part of -i*A is -A_imag
    // Imaginary part of -i*A is A_real
    for (int i = 0; i < dim * dim; i++) {
        iA_real[i] = -theta * A_imag[i];
        iA_imag[i] = theta * A_real[i];
    }
    
    // Temporary arrays for computation
    double* term_real = (double*)malloc(dim * dim * sizeof(double));
    double* term_imag = (double*)malloc(dim * dim * sizeof(double));
    double* temp_real = (double*)malloc(dim * dim * sizeof(double));
    double* temp_imag = (double*)malloc(dim * dim * sizeof(double));
    
    // Copy iA to term as first term in series
    memcpy(term_real, iA_real, dim * dim * sizeof(double));
    memcpy(term_imag, iA_imag, dim * dim * sizeof(double));
    
    // Add first term to result
    for (int i = 0; i < dim * dim; i++) {
        exp_real[i] += term_real[i];
        exp_imag[i] += term_imag[i];
    }
    
    // Taylor series expansion: exp(M) = I + M + M^2/2! + M^3/3! + ...
    double factorial = 1.0;
    for (int n = 2; n <= 15; n++) { // Use 15 terms for good approximation
        factorial *= n;
        
        // Compute next term: M^n = M^(n-1) * M
        complex_matrix_multiply(term_real, term_imag, iA_real, iA_imag, temp_real, temp_imag, dim);
        
        // Copy result back to term
        memcpy(term_real, temp_real, dim * dim * sizeof(double));
        memcpy(term_imag, temp_imag, dim * dim * sizeof(double));
        
        // Add term/factorial to result
        for (int i = 0; i < dim * dim; i++) {
            exp_real[i] += term_real[i] / factorial;
            exp_imag[i] += term_imag[i] / factorial;
        }
    }
    
    // Free temporary arrays
    free(iA_real);
    free(iA_imag);
    free(term_real);
    free(term_imag);
    free(temp_real);
    free(temp_imag);
}

// Define the kernel function for block processing
static void furx_qudit_kernel(
    double* x_real,
    double* x_imag,
    double* temp_real,
    double* temp_imag,
    double* rotation_real,
    double* rotation_imag,
    int* target_qubits,
    int target_qubits_len,
    int n_qubits,
    int block_idx, 
    int group_size, 
    int q_offset) {
    
    long qudit_dim = 1L << target_qubits_len;
    long n_states = 1L << n_qubits;
    
    // Calculate block dimensions
    unsigned int data_size = 1 << group_size;
    unsigned int stride = 1 << q_offset;
    unsigned int index_mask = stride - 1;
    unsigned int stride_mask = ~index_mask;
    unsigned int offset = ((stride_mask & block_idx) << group_size) | (index_mask & block_idx);
    
    // Create a bit mask for extracting target qubits
    long* masks = (long*)malloc(target_qubits_len * sizeof(long));
    if (!masks) {
        fprintf(stderr, "Error: Failed to allocate memory for masks\n");
        return;
    }
    
    for (int i = 0; i < target_qubits_len; i++) {
        masks[i] = 1L << target_qubits[i];
    }
    
    // Process block of states
    for (unsigned int idx = 0; idx < data_size; ++idx) {
        unsigned int global_idx = offset + idx * stride;
        
        // Skip if out of bounds (possible for last blocks)
        if (global_idx >= n_states) continue;
        
        // Extract the bit values for the target qubits
        int target_bits = 0;
        for (int q_idx = 0; q_idx < target_qubits_len; q_idx++) {
            if (global_idx & masks[q_idx]) {
                target_bits |= (1 << q_idx);
            }
        }
        
        // Apply rotation
        double real_sum = 0.0;
        double imag_sum = 0.0;
        for (int j = 0; j < qudit_dim; j++) {
            // Create index with modified target bits
            long new_idx = global_idx;
            for (int q_idx = 0; q_idx < target_qubits_len; q_idx++) {
                // Clear the bit at target position
                new_idx &= ~masks[q_idx];
                // Set the bit if it's set in j
                if (j & (1 << q_idx)) {
                    new_idx |= masks[q_idx];
                }
            }
            double rot_real = rotation_real[target_bits * qudit_dim + j];
            double rot_imag = rotation_imag[target_bits * qudit_dim + j];
            double temp_real_val = temp_real[new_idx];
            double temp_imag_val = temp_imag[new_idx];
            
            real_sum += rot_real * temp_real_val - rot_imag * temp_imag_val;
            imag_sum += rot_real * temp_imag_val + rot_imag * temp_real_val;
        }
        x_real[global_idx] = real_sum;
        x_imag[global_idx] = imag_sum;
    }
    
    free(masks);
}

// Replace the c_furx_qudit function with this updated version
void c_furx_qudit(
    double* x_real,
    double* x_imag,
    double theta,
    double* A_real,
    double* A_imag,
    int* target_qubits,
    int target_qubits_len,
    int n_qubits) {
    
    // Calculate dimensions
    long n_states = 1L << n_qubits;
    long qudit_dim = 1L << target_qubits_len;
    
    // Create matrix exponential of the rotation
    double* rotation_real = (double*)malloc(qudit_dim * qudit_dim * sizeof(double));
    double* rotation_imag = (double*)malloc(qudit_dim * qudit_dim * sizeof(double));
    if (!rotation_real || !rotation_imag) {
        if (rotation_real) free(rotation_real);
        if (rotation_imag) free(rotation_imag);
        fprintf(stderr, "Error: Failed to allocate memory for rotation matrix\n");
        return;
    }
    
    // Compute the matrix exponential exp(-i*theta*A)
    compute_matrix_exponential(A_real, A_imag, theta, rotation_real, rotation_imag, qudit_dim);
    
    // Copy the original state vector
    double* temp_real = (double*)malloc(n_states * sizeof(double));
    double* temp_imag = (double*)malloc(n_states * sizeof(double));
    if (!temp_real || !temp_imag) {
        free(rotation_real);
        free(rotation_imag);
        if (temp_real) free(temp_real);
        if (temp_imag) free(temp_imag);
        fprintf(stderr, "Error: Failed to allocate memory for temporary state\n");
        return;
    }
    
    memcpy(temp_real, x_real, n_states * sizeof(double));
    memcpy(temp_imag, x_imag, n_states * sizeof(double));
    
    // Parallelization parameters
    const unsigned int group_size = 10;  // Process 10 qubits at a time
    const unsigned int group_blocks = n_states >> group_size;
    const unsigned int last_group_size = n_qubits % group_size;
    const unsigned int last_group_blocks = n_states >> last_group_size;
    
    // Process blocks in parallel
    for (int q_offset = 0; q_offset < n_qubits - last_group_size; q_offset += group_size) {
        #pragma omp parallel for
        for (int i = 0; i < group_blocks; ++i) {
            furx_qudit_kernel(x_real, x_imag, temp_real, temp_imag, rotation_real, rotation_imag, 
                              target_qubits, target_qubits_len, n_qubits, i, group_size, q_offset);
        }
    }
    
    // Handle any remaining qubits
    if (last_group_size > 0) {
        const unsigned int q_offset = n_qubits - last_group_size;
        #pragma omp parallel for
        for (int i = 0; i < last_group_blocks; ++i) {
            furx_qudit_kernel(x_real, x_imag, temp_real, temp_imag, rotation_real, rotation_imag, 
                              target_qubits, target_qubits_len, n_qubits, i, last_group_size, q_offset);
        }
    }
    
    free(rotation_real);
    free(rotation_imag);
    free(temp_real);
    free(temp_imag);
}


// Function to apply rotation to all qudits
void c_furx_all_qudit(
    double* x_real,
    double* x_imag,
    double theta,
    double* A_real,
    double* A_imag,
    int n_precision,
    int n_qubits) {
    
    int num_qudits = n_qubits / n_precision;
    
    // Validate dimensions
    if (n_qubits % n_precision != 0) {
        fprintf(stderr, "Error: n_qubits must be divisible by n_precision\n");
        return;
    }
    
    int* target_qubits = (int*)malloc(n_precision * sizeof(int));
    if (!target_qubits) {
        fprintf(stderr, "Error: Failed to allocate memory for target qubits\n");
        return;
    }

    for (int i = 0; i < num_qudits; i++) {
        for (int j = 0; j < n_precision; j++) {
            target_qubits[j] = i * n_precision + j;
        }
        
        c_furx_qudit(x_real, x_imag, theta, A_real, A_imag, target_qubits, n_precision, n_qubits);
    }
    
    free(target_qubits);
}
void furx_kernel(double* a_real, double* a_imag, const double a, const double b, 
    const int block_idx, const int n_q, const int q_offset, const int state_mask){

    const unsigned int data_size = 1 << n_q;
    const unsigned int stride_size = data_size >> 1;

    const unsigned int stride = 1 << q_offset;
    const unsigned int index_mask = stride - 1;
    const unsigned int stride_mask = ~index_mask;
    const unsigned int offset = ((stride_mask & block_idx) << n_q) | (index_mask & block_idx);

    // temporary local arrays
    double real[data_size], imag[data_size];

    // load global data into local array
    for(int idx = 0; idx < data_size; ++idx){
        const unsigned int data_idx = offset + idx*stride;
        real[idx] = a_real[data_idx];
        imag[idx] = a_imag[data_idx];
    }

    // perform n_q steps
    for(int q = 0; q < n_q; ++q){
        const unsigned int mask1 = (1 << q) - 1;
        const unsigned int mask2 = state_mask - mask1;

        for(int tid = 0; tid < stride_size; ++tid){
            const unsigned int ia = ((tid & mask1) | ((tid & mask2) << 1));
            const unsigned int ib = (ia | (1 << q));

            const double ar = real[ia];
            const double ai = imag[ia];
            const double br = real[ib];
            const double bi = imag[ib];

            real[ia] = a*ar - b*bi;
            imag[ia] = a*ai + b*br;
            real[ib] = a*br - b*ai;
            imag[ib] = a*bi + b*ar;
        }
    }
    
    // load local data into global array
    for(int idx = 0; idx < data_size; ++idx){
        const unsigned int data_idx = offset + idx*stride;
        a_real[data_idx] = real[idx];
        a_imag[data_idx] = imag[idx];
    }
}

/**
 * apply to a statevector single-qubit Pauli-X rotations all all qubits with the
 * same rotation angle, i.e.,
 *      U(theta) = sum_{j} exp(-i*theta*X_j)
 * where X_j is the Pauli-X operator applied on the jth qubit.
 * @param sv_real array of length n containing real parts of the statevector
 * @param sv_imag array of length n containing imaginary parts of the statevector
 * @param theta rotation angle
 * @param n_qubits total number of qubits represented by the statevector
 * @param n_states size of the statevector
 */ 
void furx_all(double* a_real, double* a_imag, double theta, unsigned int n_qubits, size_t n_states)
{
    const size_t state_mask = (n_states - 1) >> 1;

    const double a = cos(theta);
    const double b = -sin(theta);

    const unsigned int group_size = 10;
    const unsigned int group_blocks = n_states >> group_size;
    const unsigned int last_group_size = n_qubits % group_size;
    const unsigned int last_group_blocks = n_states >> last_group_size;

    for(int q_offset = 0; q_offset < n_qubits - last_group_size; q_offset += group_size){
        #pragma omp parallel for
        for(int i = 0; i < group_blocks; ++i)
            furx_kernel(a_real, a_imag, a, b,i, group_size, q_offset,state_mask);
    }

    if(last_group_size > 0){
        const unsigned int q_offset = n_qubits - last_group_size;
        #pragma omp parallel for
        for(int i = 0; i < last_group_blocks; ++i)
            furx_kernel(a_real, a_imag, a, b, i, last_group_size, q_offset, state_mask);
    }
}


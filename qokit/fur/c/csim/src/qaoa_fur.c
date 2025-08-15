/***************************************
* // SPDX-License-Identifier: Apache-2.0
* // Copyright : JP Morgan Chase & Co
****************************************/ 

#include <fur.h>
#include <diagonal.h>
#include <qaoa_fur.h>

/**
 * apply a QAOA with the X mixer defined by
 * U(beta) = sum_{j} exp(-i*beta*X_j)
 * where X_j is the Pauli-X operator applied on the jth qubit
 * @param sv_real array of length n containing real parts of the statevector
 * @param sv_imag array of length n containing imaginary parts of the statevector
 * @param gammas parameters for the phase separating layers 
 * @param betas parameters for the mixing layers 
 * @param hc_diag array of length n containing diagonal elements of the diagonal cost Hamiltonian
 * @param n_qubits total number of qubits represented by the statevector
 * @param n_states size of the statevector
 * @param n_layers number of QAOA layers
 */
void apply_qaoa_furx(double* sv_real, double* sv_imag, double* const gammas, double* const betas, double* const hc_diag, unsigned int n_qubits, size_t n_states, size_t n_layers)
{
    for (size_t i=0; i<n_layers; i++)
    {
        apply_diagonal(sv_real, sv_imag, -0.5*gammas[i], hc_diag, n_states);
        furx_all(sv_real, sv_imag, betas[i], n_qubits, n_states);
    }
}

/**
 * apply a QAOA with the X mixer defined by
 * U(beta) = sum_{j} exp(-i*beta*X_j)
 * where X_j is the Pauli-X operator applied on the jth qubit
 * @param sv_real array of length n containing real parts of the statevector
 * @param sv_imag array of length n containing imaginary parts of the statevector
 * @param gammas parameters for the phase separating layers 
 * @param betas parameters for the mixing layers 
 * @param hc_diag array of length n containing diagonal elements of the diagonal cost Hamiltonian
 * @param n_qubits total number of qubits represented by the statevector
 * @param n_precision precision of the qudits
 * @param A_mat_real real part of the A matrix
 * @param A_mat_imag imaginary part of the A matrix
 * @param n_states size of the statevector
 * @param n_layers number of QAOA layers
 */
void apply_qaoa_furx_qudit(
    double* sv_real, 
    double* sv_imag,
    double* const gammas, 
    double* const betas, 
    double* const hc_diag, 
    double* A_mat_real,
    double* A_mat_imag,
    unsigned int n_precision, 
    unsigned int n_qubits,
    size_t n_states,
    size_t n_layers)
{
    for (size_t i = 0; i < n_layers; i++) {
        apply_diagonal(sv_real, sv_imag, -0.5 * gammas[i], hc_diag, n_states);
        c_furx_all_qudit(sv_real, 
            sv_imag, 
            betas[i],
            A_mat_real, 
            A_mat_imag, 
            n_precision,
            n_qubits); // Ensure all required arguments are passed
    }
}


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
 * apply a QAOA with the XY-ring mixer defined by
 * U(beta) = sum_{j} exp(-i*beta*(X_{j}X_{j+1}+Y_{j}Y_{j+1})/2)
 * where X_j and Y_j are the Pauli-X and Pauli-Y operators applied on the jth qubit, respectively.
 * @param sv_real array of length n containing real parts of the statevector
 * @param sv_imag array of length n containing imaginary parts of the statevector
 * @param gammas parameters for the phase separating layers 
 * @param betas parameters for the mixing layers 
 * @param hc_diag array of length n containing diagonal elements of the diagonal cost Hamiltonian
 * @param n_qubits total number of qubits represented by the statevector
 * @param n_states size of the statevector
 * @param n_layers number of QAOA layers
 * @param n_trotters number of Trotter steps in each XY mixer layer
 */ 
void apply_qaoa_furxy_ring(double* sv_real, double* sv_imag, double* const gammas, double* const betas, double* const hc_diag, unsigned int n_qubits, size_t n_states, size_t n_layers, size_t n_trotters)
{
    for (size_t i=0; i<n_layers; i++)
    {
        apply_diagonal(sv_real, sv_imag, -0.5*gammas[i], hc_diag, n_states);
        for (size_t j=0; j<n_trotters; j++)
        {
            furxy_ring(sv_real, sv_imag, betas[i]/n_trotters, n_qubits, n_states);
        }
    }
}


/**
 * apply a QAOA with the XY-complete mixer defined by
 * U(beta) = sum_{j,k} exp(-i*beta*(X_{j}X_{k}+Y_{j}Y_{k})/2)
 * where X_j and Y_j are the Pauli-X and Pauli-Y operators applied on the jth qubit, respectively.
 * @param sv_real array of length n containing real parts of the statevector
 * @param sv_imag array of length n containing imaginary parts of the statevector
 * @param gammas parameters for the phase separating layers 
 * @param betas parameters for the mixing layers 
 * @param hc_diag array of length n containing diagonal elements of the diagonal cost Hamiltonian
 * @param n_qubits total number of qubits represented by the statevector
 * @param n_states size of the statevector
 * @param n_layers number of QAOA layers
 * @param n_trotters number of Trotter steps in each XY mixer layer
 */ 
void apply_qaoa_furxy_complete(double* sv_real, double* sv_imag, double* const gammas, double* const betas, double* const hc_diag, unsigned int n_qubits, size_t n_states, size_t n_layers, size_t n_trotters)
{
    for (size_t i=0; i<n_layers; i++)
    {
        apply_diagonal(sv_real, sv_imag, -0.5*gammas[i], hc_diag, n_states);
        for (size_t j=0; j<n_trotters; j++)
        {
            furxy_complete(sv_real, sv_imag, betas[i]/n_trotters, n_qubits, n_states);
        }
    }
}

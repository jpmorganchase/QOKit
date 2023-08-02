/***************************************
* // SPDX-License-Identifier: Apache-2.0
* // Copyright : JP Morgan Chase & Co
****************************************/ 

#include <fur.h>
#include <stdio.h>
#include <math.h>

/**
 * apply to a statevector a single-qubit Pauli-X rotation defined by
 * Rx(theta) = exp(-i*theta*X)
 * where X is the Pauli-X operator.
 * @param sv_real array of length n containing real parts of the statevector
 * @param sv_imag array of length n containing imaginary parts of the statevector
 * @param theta rotation angle
 * @param q index of qubit on which the rotation is applied
 * @param n_states size of the statevector
 */ 
void furx(double* a_real, double* a_imag, double theta, unsigned int q, size_t n_states)
{
    // number of groups of states on which the operation is applied locally
    size_t num_groups = n_states / 2;

    // helper digit masks for constructing the locality indices
    // the mask is applied on the index through all groups of local operations
    size_t mask1 = ((size_t)1<<q) - 1;  // digits lower than q
    size_t mask2 = mask1 ^ ((n_states-1) >> 1);  // digits higher than q
    
    // pre-compute coefficients in transformation
    double cx = cos(theta), sx = sin(theta);

    #pragma omp parallel for
    for (size_t i = 0; i < num_groups; i++)
    {
        size_t i1 = (i&mask1) | ((i&mask2)<<1);
        size_t i2 = i1 | ((size_t)1<<q);

        double a1r = a_real[i1];
        double a2r = a_real[i2];
        double a1i = a_imag[i1];
        double a2i = a_imag[i2];

        a_real[i1] = cx * a1r + sx * a2i;
        a_real[i2] = sx * a1i + cx * a2r;
        
        a_imag[i1] = cx * a1i - sx * a2r;
        a_imag[i2] = -sx * a1r + cx * a2i;
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
    for (unsigned int i=0; i< n_qubits; i++)
    {
        furx(a_real, a_imag, theta, i, n_states);
    }
}

/**
 * apply to a statevector a two-qubit XX+YY rotation defined by
 * Rx(theta) = exp(-i*theta*(XX+YY)/2)
 * where X and Y are the Pauli-X and Pauli-Y operators, respectively.
 * @param sv_real array of length n containing real parts of the statevector
 * @param sv_imag array of length n containing imaginary parts of the statevector
 * @param theta rotation angle
 * @param q1 index of the first qubit on which the rotation is applied
 * @param q2 index of the second qubit on which the rotation is applied
 * @param n_states size of the statevector
 */ 
void furxy(double* a_real, double* a_imag, double theta, unsigned int q1, unsigned int q2, size_t n_states)
{
    // make sure q1 < q2
    if (q1 > q2)
    {
        q1 ^= q2;
        q2 ^= q1;
        q1 ^= q2;
    }

    // number of groups of states on which the operation is applied locally
    size_t num_groups = n_states / 4;

    // helper digit masks for constructing the locality indices
    // the mask is applied on the index through all groups of local operations
    size_t mask1 = ((size_t)1<<q1) - 1;  // digits lower than q1
    size_t mask2 = ((size_t)1<<(q2-1)) - 1;  // digits lower than q2
    size_t maskm = mask1^mask2;  // digits in between
    mask2 ^= (n_states-1) >> 2;  // digits higher than q2
    
    // pre-compute coefficients in transformation
    double cx = cos(theta), sx = sin(theta);

    #pragma omp parallel for
    for (size_t i = 0; i < num_groups; i++)
    {
        size_t i0 = (i&mask1) | ((i&maskm)<<1) | ((i&mask2)<<2);
        size_t i1 = i0 | ((size_t)1<<q1);
        size_t i2 = i0 | ((size_t)1<<q2);

        double a1r = a_real[i1];
        double a2r = a_real[i2];
        double a1i = a_imag[i1];
        double a2i = a_imag[i2];

        a_real[i1] = cx * a1r + sx * a2i;
        a_real[i2] = sx * a1i + cx * a2r;
        
        a_imag[i1] = cx * a1i - sx * a2r;
        a_imag[i2] = -sx * a1r + cx * a2i;
    }
}

/**
 * apply to a statevector single-qubit Pauli-X rotations all all qubits with the
 * same rotation angle, i.e.,
 *      U(theta) = sum_{j} exp(-i*theta*X_j)
 * where X_j is the Pauli-X operator applied on the jth qubit
 * @param sv_real array of length n containing real parts of the statevector
 * @param sv_imag array of length n containing imaginary parts of the statevector
 * @param theta rotation angle
 * @param n_qubits total number of qubits represented by the statevector
 * @param n_states size of the statevector
 */ 
void furxy_ring(double* a_real, double* a_imag, double theta, unsigned int n_qubits, size_t n_states)
{
    
    for (int i=0; i<2; i++)
    {
        for (unsigned int j=i; j<n_qubits-1; j+=2)
        {
            furxy(a_real, a_imag, theta, j, j+1, n_states);
        }
    }
    furxy(a_real, a_imag, theta, 0, n_qubits-1, n_states);
}

/**
 * apply to a statevector two-qubit XX+YY rotations defined by
 * Rx(theta) = exp(-i*theta*(XX+YY)/2)
 * on all all adjacent pairs of qubits with wrap-around.
 * @param sv_real array of length n containing real parts of the statevector
 * @param sv_imag array of length n containing imaginary parts of the statevector
 * @param theta rotation angle
 * @param n_qubits total number of qubits represented by the statevector
 * @param n_states size of the statevector
 */ 
void furxy_complete(double* a_real, double* a_imag, double theta, unsigned int n_qubits, size_t n_states)
{
    
    for (unsigned int i=0; i<n_qubits-1; i++)
    {
        for (unsigned int j=i+1; j<n_qubits; j++)
        {
            furxy(a_real, a_imag, theta, i, j, n_states);
        }
    }
}

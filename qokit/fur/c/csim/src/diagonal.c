/***************************************
* // SPDX-License-Identifier: Apache-2.0
* // Copyright : JP Morgan Chase & Co
****************************************/ 
#include <stdlib.h>
#include <math.h>
#include <diagonal.h>


/**
 * apply a exp(-i * theta * H) to a statevector where H is a diagonal matrix
 * @param sv_real array of length n containing real parts of the statevector
 * @param sv_imag array of length n containing imaginary parts of the statevector
 * @param theta multiplicative weight applied to the diagonal matrix
 * @param diag array of length n containing the diagonal elements of the diagonal matrix
 * @param n size of the statevector and matrix diagonal
 */ 
void apply_diagonal(double* sv_real, double* sv_imag, double theta, double* const diag, size_t n)
{    
    #pragma omp parallel for
    for (size_t i=0; i<n; i++)
    {
        double exp_real = cos(theta * diag[i]);
        double exp_imag = sin(theta * diag[i]);
        double res_real = sv_real[i]*exp_real - sv_imag[i]*exp_imag;
        double res_imag = sv_real[i]*exp_imag + sv_imag[i]*exp_real;
        sv_real[i] = res_real;
        sv_imag[i] = res_imag;
    }
}

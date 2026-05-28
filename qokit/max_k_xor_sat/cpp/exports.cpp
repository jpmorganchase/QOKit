/**
 * C-linkage exports for Python ctypes binding.
 *
 * Wraps the C++ API with extern "C" to avoid name mangling.
 */

#include "contract.h"
#include "grad.h"

extern "C" {

double qaoa_contract(const double* gammas, const double* betas,
                     int p, int D, int k,
                     int use_dd, int verbose) {
    return contract_symmetric_tree(gammas, betas, p, D, k,
                                   use_dd != 0, verbose != 0);
}

double qaoa_contract_grad(const double* gammas, const double* betas,
                          int p, int D, int k,
                          double* grad_gammas, double* grad_betas,
                          int use_dd) {
    return contract_with_grad(gammas, betas, p, D, k,
                              grad_gammas, grad_betas, use_dd != 0);
}

long long qaoa_light_cone_size(int p, int D, int k) {
    return light_cone_size(p, D, k);
}

}  // extern "C"

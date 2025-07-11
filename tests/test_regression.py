from __future__ import annotations
import numpy as np
from qokit.portfolio_optimization import get_problem
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective


def _get_obj(p=1):
    po = get_problem(N=6, K=3, q=0.4, pre="rule")
    return get_qaoa_portfolio_objective(
        po, p=p, precomputed_energies="vectorized"
    )


def test_batch_equals_scalar() -> None:
    """Batched objective must match scalar path exactly."""
    obj = _get_obj(p=1)
    rng = np.random.default_rng(0)
    theta_stack = rng.random((5, 2))            # 5 random θ

    e_batch = obj(theta_stack)
    e_loop  = np.array([obj(t) for t in theta_stack])

    np.testing.assert_allclose(e_batch, e_loop, rtol=1e-12, atol=1e-12)


def test_phase_cache_consistency() -> None:
    """Repeated γ on same diagonal must give identical energies."""
    obj = _get_obj(p=1)
    theta = np.array([0.3, 0.2])

    e1 = obj(theta)
    e2 = obj(theta)           # hits cache second time

    assert e1 == e2


def test_lbfgs_reaches_bobyqa_quality() -> None:
    """L-BFGS-B final energy should be no worse than BOBYQA."""
    obj = _get_obj(p=2)
    rng = np.random.default_rng(0)
    x0 = rng.random(4)

    # BOBYQA reference
    import nlopt
    o = nlopt.opt(nlopt.LN_BOBYQA, 4)
    o.set_min_objective(lambda x, g: obj(x))
    o.set_initial_step(0.2); o.set_maxeval(50)
    ref = o.optimize(x0)

    # L-BFGS-B
    from scipy.optimize import minimize



    res = minimize(obj, x0, method="L-BFGS-B", jac=False,
                   options={"maxiter": 50})

    assert obj(res.x) <= obj(ref) + 1e-8

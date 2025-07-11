"""
Batch-mode sanity checks for the vectorised objective.

Run:  pytest -q
"""
from __future__ import annotations
import numpy as np
from qokit.portfolio_optimization import get_problem
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective


def test_vectorised_equals_loop() -> None:
    """Batched energies must match the per-sample loop exactly."""
    po = get_problem(N=6, K=3, q=0.4, pre="rule")

    obj = get_qaoa_portfolio_objective(
        po,
        p=1,
        mixer="trotter_ring",
        precomputed_energies="vectorized",
    )

    rng = np.random.default_rng(0)
    # two random parameter vectors
    theta_a = rng.random(2)          # (2p,)
    theta_b = rng.random(2)

    e_loop = np.array([obj(theta_a), obj(theta_b)])
    e_vec  = obj(np.stack([theta_a, theta_b]))  # (B,2p) batch

    np.testing.assert_allclose(e_loop, e_vec, rtol=1e-12, atol=1e-12)

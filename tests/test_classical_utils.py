###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from qokit.classical_methods.utils import BestBoundAborter


def test_best_bound_aborter_notify_start():
    aborter = BestBoundAborter(max_best_bound=10)
    assert aborter.last_obj is None

    aborter.notify_start()
    assert aborter.last_obj is None

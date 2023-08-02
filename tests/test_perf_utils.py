###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import time
from contextlib import contextmanager

from numpy import True_
from qokit.perf_utils import timeit, set_timer


def test_timeit_context_manager():
    with timeit("Test Code"):
        time.sleep(1)

    # No assertion. Test fails if exception is raised


def test_set_timer():
    set_timer(True)

    with timeit("Test Timer"):
        time.sleep(1)

    set_timer(False)

    # No assertion. Test fails if exception is raised


def test_timeout_decorator():
    @timeit("Test Function")
    def test_func():
        time.sleep(1)

    test_func()


# No assertion. Test fails if exception is raised

###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import time
from contextlib import contextmanager
from typing import Optional

_TIMER_ON: bool = False


def set_timer(on: bool = True):
    """
    turn ON/OFF timer output
    """
    global _TIMER_ON
    _TIMER_ON = on
    print(f"Set timer to {'ON' if on else 'OFF'}")


@contextmanager
def timeit(name: Optional[str] = None):
    """
    simple timer context manager
    usage 1:
        with timeit("my awsome code"):
            <some very cool stuff ...>

    usage 2:
        @timeit("my awsome code"):
        def my_func():
            <some very cool stuff ...>

    output:
        >>> [TIME] my awsome code took <xxxx> seconds to run
    """
    start = time.monotonic()
    try:
        yield
    finally:
        if _TIMER_ON:
            print(f"[TIME] {name} took {time.monotonic() - start} seconds to run")

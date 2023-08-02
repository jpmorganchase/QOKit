###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from pathlib import Path


code_dir = Path(__file__).parent
libpath = code_dir / "libcsim.so"


def is_available() -> bool:
    """
    returns True if the C simulator is available
    """
    return libpath.is_file()

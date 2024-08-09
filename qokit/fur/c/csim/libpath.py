###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from pathlib import Path
import glob


code_dir = Path(__file__).parent

libcsim = glob.glob(f"{code_dir}/libcsim*.so")
if len(libcsim) == 0:
    raise FileNotFoundError(f"libcsim*.so not found in the path {code_dir}. Please run make or setup.py")

libpath = code_dir / libcsim[0]


def is_available() -> bool:
    """
    returns True if the C simulator is available
    """
    return libpath.is_file()

###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import importlib
import warnings


class LasyModule:
    def __init__(self, modulename):
        self.modulename = modulename
        self.module = None
        self.printed_could_not_import = False

    def __getattr__(self, attr):
        if self.module is None:
            try:
                self.module = importlib.import_module(self.modulename)
            except (ImportError, ModuleNotFoundError):
                if not self.printed_could_not_import:
                    self.printed_could_not_import = True
                    warnings.warn(f"LazyModule: {self.modulename} is missing.", ImportWarning)
                raise
        return self.module.__getattribute__(attr)


MPI = LasyModule("mpi4py.MPI")

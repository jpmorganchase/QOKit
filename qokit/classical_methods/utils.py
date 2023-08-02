###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from docplex.mp.progress import ProgressListener, ProgressClock


class BestBoundAborter(ProgressListener):
    """
    Custom aborter to stop when finding a feasible solution matching the bound.
    see: https://ibmdecisionoptimization.github.io/docplex-doc/mp/_modules/docplex/mp/progress.html#ProgressClock
    https://dataplatform.cloud.ibm.com/exchange/public/entry/view/6e2bffa5869dacbae6500c7037ecd36f
    """

    def __init__(self, max_best_bound=0):
        super(BestBoundAborter, self).__init__(ProgressClock.BestBound)
        self.max_best_bound = max_best_bound
        self.last_obj = None

    def notify_start(self):
        super(BestBoundAborter, self).notify_start()
        self.last_obj = None

    def notify_progress(self, pdata):
        super(BestBoundAborter, self).notify_progress(pdata)
        if pdata.has_incumbent:
            self.last_obj = pdata.current_objective
            if self.last_obj <= self.max_best_bound:
                print(f"_____ FOUND Feasible solution {self.last_obj} smaller than stopping condition {self.max_best_bound}")
                self.abort()

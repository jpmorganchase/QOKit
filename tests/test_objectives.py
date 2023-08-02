###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import warnings
import itertools
import numpy as np
import pandas as pd
import numba.cuda
import pytest
from pathlib import Path
from tqdm import tqdm

from qokit.qaoa_objective_labs import get_qaoa_labs_objective
from qokit.utils import get_all_best_known
from qokit.fur import get_available_simulator_names
from qokit.parameter_utils import get_best_known_parameters_for_LABS_wrt_overlap, get_best_known_parameters_for_LABS_wrt_overlap_for_p

test_objectives_folder = Path(__file__).parent

simulators_to_run = get_available_simulator_names("x")


class TestBestParamsMatchValues:
    df = get_all_best_known()
    # fast version: sample a few rows randomly
    nrows = 5
    rows = df[df["N"] < 15].sample(nrows, axis=0)

    @pytest.mark.parametrize("simulator", simulators_to_run)
    @pytest.mark.parametrize("objective", ("expectation", "overlap"))
    def test_best_params_match_values(self, objective, simulator):
        if objective == "expectation":
            suffix_to_check = " opt4MF"
            col_name = "merit factor" + suffix_to_check
            convert_result = lambda x: -x
        else:
            suffix_to_check = " opt4overlap"
            col_name = "overlap" + suffix_to_check
            convert_result = lambda x: 1 - x

        for _, row in tqdm(
            self.rows.iterrows(),
            desc=f'Verifying the DataFrame with best known parameters (objective="{objective}", simulator="{simulator}")',
            total=self.nrows,
        ):
            f = get_qaoa_labs_objective(row["N"], row["p"], objective=objective, simulator=simulator)

            x = np.hstack((row["gamma" + suffix_to_check], row["beta" + suffix_to_check]))

            if pd.notna(row[col_name]):  # ignore missing values
                assert np.isclose(
                    f(x), convert_result(row[col_name])
                ), f'Output values from simulator "{simulator}" and objective "{objective}" do not match with known values'


@pytest.mark.parametrize("simulator", simulators_to_run)
def test_return_both_objectives(simulator):
    N = 7
    p = 4

    theta = np.random.uniform(0, 1, 2 * p)

    e1 = get_qaoa_labs_objective(N, p, objective="expectation", simulator=simulator)(theta)
    o1 = get_qaoa_labs_objective(N, p, objective="overlap", simulator=simulator)(theta)
    e2, o2 = get_qaoa_labs_objective(N, p, objective="expectation and overlap", simulator=simulator)(theta)
    assert np.isclose(e1, e2)
    assert np.isclose(o1, o2)


@pytest.mark.parametrize("simulator", simulators_to_run)
def test_best_known_LABS_parameters(simulator):
    N = 9
    p = 5

    df = get_best_known_parameters_for_LABS_wrt_overlap(N)
    row = df[df["p"] == p].squeeze()
    gamma, beta = get_best_known_parameters_for_LABS_wrt_overlap_for_p(N, p)

    o = get_qaoa_labs_objective(N, p, objective="overlap", parameterization="gamma beta", simulator=simulator)(gamma, beta)
    assert np.isclose(o, 1 - row["overlap"])

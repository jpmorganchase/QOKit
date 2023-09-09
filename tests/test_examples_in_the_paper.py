# Test example code provided in ARXIV_ID TBA
# imports are inside the tests to make sure no additional dependencies are necessary

# dummy parameters
import pytest

gamma = [0.0, 0.78539816, 1.57079633, 2.35619449, 3.14159265]
beta = [3.14159265, 2.35619449, 1.57079633, 0.78539816, 0.0]


# number of qubits reduced compared to the example in the paper to make the test run faster
def test_listing_1():
    import qokit

    simclass = qokit.fur.choose_simulator(name="auto")
    n = 8  # number of qubits
    # terms for all-to-all MaxCut with weight 0.3
    terms = [(0.3, (i, j)) for i in range(n) for j in range(i + 1, n)]
    sim = simclass(n, terms=terms)
    # get precomputed cost vector
    costs = sim.get_cost_diagonal()
    result = sim.simulate_qaoa(gamma, beta)
    E = sim.get_expectation(result)


def test_listing_2():
    import qokit

    simclass = qokit.fur.choose_simulator_xycomplete()
    n = 8
    terms = qokit.labs.get_terms(n)
    sim = simclass(n, terms=terms)
    result = sim.simulate_qaoa(gamma, beta)
    E = sim.get_expectation(result)


@pytest.mark.skip(reason="cusvmpi is not yet merged")
def test_listing_3():
    import qokit

    simclass = qokit.fur.choose_simulator(name="cusvmpi")
    n = 40
    terms = qokit.labs.get_terms(n)
    sim = simclass(n, terms=terms)
    result = sim.simulate_qaoa(gamma, beta)
    E = sim.get_expectation(result, preserve_state=False)

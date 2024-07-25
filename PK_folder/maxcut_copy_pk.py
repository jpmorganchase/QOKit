import qokit
import numpy as np
from itertools import combinations
import scipy
import networkx as nx
import time
from matplotlib import pyplot as plt

#for visualization in the end
def compute_graph_colors(G, solution_index):
    N = G.number_of_nodes()
    vert_colors = list(map(lambda c : "green" if c == '1' else "blue", f"{int(solution_index):0{N}b}"))
    vert_colors.reverse()
    edge_colors = [("red" if vert_colors[u] != vert_colors[v] else "black") for u,v in G.edges()]
    return (vert_colors, edge_colors)

#%% printing functions for convenience
def print_result(N, terms, gamma, beta, adjective = ""):
    p = len(gamma)
    print(f"{adjective}Gamma: {gamma}")
    print(f"{adjective}Beta: {beta}")
    print(f"{adjective}Objective: {-inv_max_cut_objective(N, p, terms)(np.hstack([gamma, beta]))}")
    print()

def print_probabilities(G, terms, gamma, beta, threshold = 0, only_likely = False, include_cost = True):
    N = G.number_of_nodes()
    probs = get_probabilities(N, terms, gamma, beta)
    for element in np.column_stack((np.array(range(2**N)), probs)):
        [index, prob] = element
        likely_str = ""
        if (prob > threshold):
            likely_str = " <- Likely outcome"

        cost_str = ""
        if (include_cost):
            contained_verts = list(map(lambda c : c == '1', f"{int(index):0{N}b}"))
            contained_verts.reverse()
            cut_count = evaluate_cut(G, contained_verts)
            cost_str = f" ({cut_count})"

        if (not only_likely):
            print(f"{int(index):0{N}b}{cost_str} - {prob}{likely_str}")
        else:
            if (prob > threshold):
                print(f"{int(index):0{N}b}{cost_str} - {prob}")

def print_state_vector(G, terms, gamma, beta, include_cost = True):
    N = G.number_of_nodes()
    sv = get_state_vector(N, terms, gamma, beta)
    for element in np.column_stack((np.array(range(2**N)), sv)):
        [index, amplitude] = element
        cost_str = ""
        if (include_cost):
            contained_verts = list(map(lambda c : c == '1', f"{int(index):0{N}b}"))
            contained_verts.reverse()
            cut_count = evaluate_cut(G, contained_verts)
            cost_str = f" ({cut_count})"
        
        print(f"{int(index):0{N}b}{cost_str} - {amplitude}")

#%% plotting functions

#plot the expectations of the end result vector for QAOA over time 
def plot_expectation(expectations, N, p, start_time):
    def make_time_relative(input):
        time, expectation = input
        return (time - start_time, expectation)

    time_relative_expectations = list(map(make_time_relative, expectations))
    plt.scatter(*zip(*time_relative_expectations))
    plt.title(f"MaxCut (N = {N}, p = {p})")
    plt.xlabel("Time (seconds)")
    plt.ylabel("QAOA Expectation")
    plt.show()

# TODO: Create plot with both expectation and measurements
#a function to plot the costs
def plot_measurements(measurements, N, p, start_time):
    def make_time_relative(input):
        time, measurement = input
        return (time - start_time, measurement)

    time_relative_measurements = list(map(make_time_relative, measurements))
    plt.scatter(*zip(*time_relative_measurements))
    plt.title(f"MaxCut (N = {N}, p = {p})")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Measurement (best of 10)")
    plt.show()

#%% 
def random_graph(N, prob_connect = 0.7):
    A = np.random.choice([0, 1], (N, N), p=[1 - prob_connect, prob_connect])
    np.fill_diagonal(A, 0)  # No self-loops
    A = np.triu(A)  # Use only the upper triangle
    A += A.T  # Make the matrix symmetric
    return (A, nx.from_numpy_array(A))

def max_cut_terms_for_graph(G):
    return list(map((lambda edge : (-0.5, edge)), G.edges)) + [((G.number_of_edges()/2.0), ())]

def optimal_solution_with_index(sim):
    costs = sim.get_cost_diagonal()
    solution = max(costs)
    solution_index = np.where(costs == solution)[0][0]
    return (solution, solution_index)

def evaluate_cut(G, bool_verts):
    cuts = 0
    for (u, v) in G.edges():
        if (bool_verts[u] != bool_verts[v]):
            cuts += 1
    return cuts

def get_simulator(N, terms, sim_or_none = None):
    if (sim_or_none is None):
        simclass = qokit.fur.choose_simulator(name='auto')
        return simclass(N, terms=terms)
    else:
        return sim_or_none
    
def get_result(N, terms, gamma, beta, sim = None, result = None):
    if (result is None):
        simulator = get_simulator(N, terms, sim)
        return simulator.simulate_qaoa(gamma, beta)
    else:
        return result
    
def get_simulator_and_result(N, terms, gamma, beta, sim = None, result = None):
    simulator = get_simulator(N, terms, sim)
    if (result is None):
        result = get_result(N, terms, gamma, beta, simulator)
    return (simulator, result)

def get_probabilities(N, terms, gamma, beta, sim = None, result = None):
    simulator, result = get_simulator_and_result(N, terms, gamma, beta, sim, result)
    return simulator.get_probabilities(result, preserve_state = True)

def get_expectation(N, terms, gamma, beta, sim = None, result = None):
    simulator, result = get_simulator_and_result(N, terms, gamma, beta, sim, result)
    return simulator.get_expectation(result, preserve_state = True)

def get_state_vector(N, terms, gamma, beta):
    simclass = qokit.fur.choose_simulator(name='auto')
    sim = simclass(N, terms=terms)
    _result = sim.simulate_qaoa(gamma, beta)
    return sim.get_statevector(_result)

def inv_max_cut_objective(N, p, terms, expectations = None, measurements = None, sim = None):
    def f(*args):
        gamma, beta = args[0][:p], args[0][p:]
        current_time = time.time()
        simulator = get_simulator(N, terms, sim)
        probs = get_probabilities(N, terms, gamma, beta, sim)
        costs = simulator.get_cost_diagonal()
        expectation = np.dot(costs, probs)

        if (expectations != None):
            expectations.append((current_time, expectation))

        if (measurements != None):
            measurement = max(np.random.choice(costs, 10, p=probs))
            measurements.append((current_time, measurement))
        
        return -expectation
    return f


##
##write this instead as optionally calculating the expectations and measurements
#something like calculate_expectation = TRUE or FALSE, and if false it entirely skips the computations
def optimize(N, terms, init_gamma, init_beta, expectations = None, measurements = None, sim = None, optimizer = 'COBYLA'):
    '''
    This function aims to find the optimal gamma and beta for QAOA for the MaxCut problem. 
    arguments: 
        'N' is the number of nodes of the graph. TYPE: int
        'terms' contains the terms of the cost Hamiltonian. TYPE: list
        'init_gamma' and 'init_beta' are the initial starting parameters for gamma and beta
        
        optional arguments: 
        'optimizer' specifies what optimization method SciPy should use. See scipy.optimize documentation for options.
    '''
    if  (expectations != None):
        expectations.clear()
    if (measurements != None):
        measurements.clear()
    
    p = len(init_gamma)
    assert len(init_beta) == p, "Gamma and Beta must have the same length"
    init_freq = np.hstack([init_gamma, init_beta])

    #scipy uses minimize for everything, but we want to maximize for MaxCut, so we take the negative 
    #of the MaxCut objective value in inv_max_cut_objective 
    #scipy supported optimizers: 
    res = scipy.optimize.minimize(inv_max_cut_objective(N, p, terms, expectations, measurements, sim), init_freq, method = optimizer, options={'rhobeg': 0.01/N})

    gamma, beta = res.x[:p], res.x[p:]
    return (gamma, beta)

# TODO: Compare to randomly trying cuts (and keeping best)
def run_experiment(seed, N, p, optimizer = 'COBYLA'):
    np.random.seed(seed)
    (A, G) = random_graph(N, 0.5) 
    terms = max_cut_terms_for_graph(G)
    sim = get_simulator(N, terms)
    expectations = []
    measurements = []

    solution, solution_index = optimal_solution_with_index(sim)
    print(f"Optimal Solution: {solution_index:0{N}b} - {solution}")
    vert_colors, edge_colors = compute_graph_colors(G, solution_index)

    init_gamma, init_beta = np.random.rand(2, p)
    print_result(N, terms, init_gamma, init_beta, "Initial ")

    start_time = time.time()
    gamma, beta = optimize(N, terms, init_gamma, init_beta, expectations, measurements, sim, optimizer = optimizer)
    end_time = time.time()
    print(f"Time to optimize: {end_time - start_time} seconds\n")

    print_result(N, terms, gamma, beta, "Final ")
    print_probabilities(G, terms, gamma, beta, threshold=0.5/N, only_likely=True)
    print()
    plot_expectation(expectations, N, p, start_time)
    plot_measurements(measurements, N, p, start_time)

    nx.draw(G, nx.circular_layout(G), node_color=vert_colors, edge_color=edge_colors, with_labels=True)

def collect_state_vectors(num_vecs, G, p):
    N = G.number_of_nodes()
    terms = max_cut_terms_for_graph(G)
    qaoa_state_vectors = []

    for _ in range(num_vecs):
        init_gamma, init_beta = np.random.rand(2, p)
        gamma, beta = optimize(N, terms, init_gamma, init_beta)
        qaoa_state_vectors.append(get_state_vector(N, terms, gamma, beta))

    return qaoa_state_vectors

        
#(_, G) = random_graph(N=7)
#collect_state_vectors(10, G, p=3)
run_experiment(seed = 0, N = 4, p = 2, optimizer = 'COBYLA')
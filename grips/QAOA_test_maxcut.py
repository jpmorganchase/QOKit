
import typing
import qokit
import numpy as np
import scipy
import time
import networkx as nx
from qokit.fur.qaoa_simulator_base import QAOAFastSimulatorBase, TermsType
import QAOA_simulator as qs

'''
This is a script for testing the QAOA_simulator code. 
In particular, it runs max cut on a small graph to test the QAOA_run function. 
'''

#%% These two functions are to build the ising model of the graph here 

#generate a random graph
def random_graph(N, prob_connect = 0.7):
    A = np.random.choice([0, 1], (N, N), p=[1 - prob_connect, prob_connect])
    np.fill_diagonal(A, 0)  # No self-loops
    A = np.triu(A)  # Use only the upper triangle
    A += A.T  # Make the matrix symmetric
    return (A, nx.from_numpy_array(A))

#build the ising model for a graph to use for QAOA maxcut cost
def max_cut_terms_for_graph(G):
    return list(map((lambda edge : (-0.5, edge)), G.edges)) + [((G.number_of_edges()/2.0), ())]


#%% Now build the model and solve with QAOA

#first, set parameters
N = 5 #graph size
p = 3 #circuit depth for QAOA
optimizer_method = 'COBYLA' #classical optimizer to use
init_gamma, init_beta = np.random.rand(2, p) #initial values
(_, G) = random_graph(N, 0.5)  #generate a random graph for G (the '_' we dont need, just networkx syntax)
ising_model = max_cut_terms_for_graph(G) #build the ising model for MaxCut on this graph
sim = qs.get_simulator(N, ising_model) #simulator for this ising model

#now solve with QAOA_run with these parameters
qaoa_result = qs.QAOA_run(
    ising_model,
    N,
    p,
    init_gamma,
    init_beta,
    optimizer_method=optimizer_method)


#print the results 
print(f'With parameters N = {N}, p = {p}, method {optimizer_method}, we got:\n\n')

#print(f'State was {qaoa_result["state"]}\n') #suppressing this printing since it's noninformative
print(f'Gamma was                {qaoa_result["gamma"]}')
print(f'Beta was                 {qaoa_result["beta"]}')
print(f'Expetation was           {qaoa_result["expectation"]}')
print(f'Overlap was              {qaoa_result["overlap"]}')
print(f'Runtime was              {qaoa_result["runtime"]}')
print(f'Number of QAOA calls was {qaoa_result["num_QAOA_calls"]}')

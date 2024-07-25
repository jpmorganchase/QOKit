# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:07:59 2024

@author: kerge
"""

import networkx as nx
import gurobipy as gp
from gurobipy import GRB
import random
import time 
import numpy as np

# Parameters
N = 100  # Number of vertices

# # Step 1: Generate a random unweighted graph
# G = nx.Graph()
# G.add_nodes_from(range(N))
# for i in range(N):
#     for j in range(i + 1, N):
#         if random.random() < 0.5:
#             G.add_edge(i, j)


# Step 1: Generate a random unweighted graph using NumPy
# Create a random adjacency matrix
print('Generating graph... \n')
adj_matrix = np.random.rand(N, N) < 0.5
np.fill_diagonal(adj_matrix, 0)  # No self-loops
adj_matrix = np.triu(adj_matrix)  # Use only the upper triangle
adj_matrix += adj_matrix.T  # Make the matrix symmetric

#Convert the adjacency matrix to a NetworkX graph
print('Converting graph... \n')
G = nx.from_numpy_array(adj_matrix)


#Formulate and solve the max cut problem using Gurobi
print('Starting Gurobi and building model...\n')
try:
    m = gp.Model("max_cut")

    # Create variables
    x = m.addVars(N, vtype=GRB.BINARY, name="x")

    # Objective function: maximize the number of edges between the two sets
    m.setObjective(gp.quicksum((1 - 2 * x[i]) * x[j] for i, j in G.edges()), GRB.MAXIMIZE)

     # Set the MIPGap parameter to 0.05 (5%) for a solution guaranteed to be at least 95% optimal
    m.setParam('MIPGap', 0.00)
    
    print('\n Solving now...\n')
    t_optstart = time.time()
    # Optimize model
    m.optimize()

    # Extract the solution
    cut = [i for i in range(N) if x[i].x > 0.5]
    non_cut = [i for i in range(N) if x[i].x <= 0.5]

    print(f"Cut set: {cut}")
    print(f"Non-cut set: {non_cut}")
    print(f"Max cut value: {m.objVal}")
    print(f"Solved in {time.time()-t_optstart} seconds")

except gp.GurobiError as e:
    print(f"Gurobi Error: {e.errno} - {e.message}")

except AttributeError:
    print("Encountered an attribute error")


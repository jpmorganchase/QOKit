###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from docplex.mp.model import Model


def generate_lp(n, path: str = "", cplex_compatible=True):
    if cplex_compatible:
        model = generate_model_cplex_compatible(n, path)
    else:
        model = generate_model_cplex_non_compatible(n, path)
    if not path:
        path = f"LABS_{int(n)}.lp"
    model.export_as_lp(path)


def c_k(k, N):
    return list([i, i + k] for i in range(1, N - k + 1))


def generate_model_cplex_non_compatible(n, path):
    model = Model(f"LABS of size {n}")
    b_vars = model.binary_var_list([i for i in range(1, n + 1)], lb=0, ub=1, name="b")
    c_vars = model.integer_var_list([k for k in range(1, n)], lb=-n, ub=n, name="C")

    for k in range(1, n):
        # at most n-k terms in C_k
        c_vars[k - 1].ub = n - k
        c_vars[k - 1].lb = -n + k

    for k in range(1, n):
        model.add_constraint((c_vars[k - 1] == sum([(2 * b_vars[x[0] - 1] - 1) * (2 * b_vars[x[1] - 1] - 1) for x in c_k(k, n)])))

    model.minimize(sum([c_vars[k - 1] * c_vars[k - 1] for k in range(1, n)]))
    return model


def generate_model_cplex_compatible(n, path):
    model = Model(f"LABS of size {n}")
    b_vars_x = model.binary_var_list([i for i in range(1, n + 1)], lb=0, ub=1, name="b")
    b_vars_y = model.binary_var_list([(x[0], x[1]) for k in range(1, n) for x in c_k(k, n)], lb=0, ub=1, name="b_y")
    c_vars = model.integer_var_list([k for k in range(1, n)], lb=-n, ub=n, name="C")

    for k in range(1, n):
        # at most n-k terms in C_k
        c_vars[k - 1].ub = n - k
        c_vars[k - 1].lb = -n + k

    for k in range(1, n):
        model.add_constraint(
            -c_vars[k - 1]
            + 4 * sum([[b for b in b_vars_y if b.name == f"b_y_{x[0]}_{x[1]}"][0] for x in c_k(k, n)])
            - 2 * sum([b_vars_x[i - 1] for i in range(n - k)])
            - 2 * sum([b_vars_x[i + k - 1] for i in range(n - k)])
            == k - n
        )

    for k in range(1, n):
        for x in c_k(k, n):
            model.add_constraint([b for b in b_vars_y if b.name == f"b_y_{x[0]}_{x[1]}"][0] <= (b_vars_x[x[0] - 1] + b_vars_x[x[1] - 1]) / 2)
            model.add_constraint([b for b in b_vars_y if b.name == f"b_y_{x[0]}_{x[1]}"][0] >= b_vars_x[x[0] - 1] + b_vars_x[x[1] - 1] - 1)

    model.minimize(sum([c_vars[k - 1] * c_vars[k - 1] for k in range(1, n)]))
    return model

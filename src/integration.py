import numpy as np

from scipy.integrate import odeint


def build_kernel(kernel_rule, N: int):
    K = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            K[i, j] = kernel_rule(i, j)
    return K


def delta(a, b):
    return 1 if a == b else 0


def dunbar(i, j):
    return i * j + 2 / 150 * (i + j) + 4 / 150 / 150


def make_tiered(theta, N):
    def tiered(i, j):
        if i + j > theta * N:
            return 2 * i * j
        return 0

    return tiered


K_DICT = {
    "product": lambda i, j,: i * j,
    "r_prod": lambda i, j: 1 / (i * j) if i * j > 0 else 0,
    "sum": lambda i, j: i + j,
    "constant": lambda i, j: 1,
    "dunbar": dunbar,
    "chipping_1": lambda i, j: (delta(i, 1) + delta(j, 1)),
    "chipping_5": lambda i, j: (delta(i, 5) + delta(j, 5)),
    "tiered_33": make_tiered(0.33, 100),
    "tiered_50": make_tiered(0.5, 100),
    "tiered_60": make_tiered(0.6, 100),
    "abs": lambda i, j: abs(i - j),
}


## main integrators
def agg_frag_dot(x, t, K, F, N):
    dx = np.zeros_like(x)
    cost_number = 0
    cost_amount = 0
    for k in range(1, N + 1):
        for i in range(1, k):
            j = k - i
            dx[k] += 0.5 * K[i, j] * x[i] * x[j]  ## agg inflow
            frag_out = 0.5 * x[k] * F[i, j]
            dx[k] -= frag_out  ## frag outflow
            cost_number += 1 / (1 + np.exp(-frag_out * 1000))
            cost_amount += frag_out * min(i, j)
        for j in range(1, N + 1 - k):
            dx[k] -= x[k] * K[k, j] * x[j]  ## agg outflow
            frag_in = F[k, j] * x[k + j]  ## frag inflow
            dx[k] += frag_in
            cost_number += 1 / (1 + np.exp(-frag_in * 1000))
            cost_amount += frag_in * min(j, k)
    dx[N + 1] = cost_number  # number of frag
    dx[N + 2] = cost_amount  # amount of frag
    return dx


def agg_frag_meq(
    agg_rule,
    frag_rule,
    N,
    t_length=5,
    t_steps=100,
    initial_conditions={1: 1},
):
    t_vec = np.linspace(0, t_length, t_steps * (t_length + 1))
    # x shape: first N+1 for cluster size, then info storage
    x_0 = np.zeros(N + 1 + 2, dtype=float)
    # setup, initially all monomers, in current version. by probability
    for size, prop in initial_conditions.items():
        x_0[size] = prop
    agg_kernel, agg_multiplier = agg_rule
    frg_kernel, frg_multiplier = frag_rule

    K = build_kernel(K_DICT[agg_kernel], N + 1)
    K *= agg_multiplier
    F = build_kernel(K_DICT[frg_kernel], N + 1)
    F *= frg_multiplier

    G = lambda x, t: agg_frag_dot(x, t, K, F, N)
    x_path = odeint(G, x_0, t_vec)
    x_path, cost_path = x_path[:, :-2], x_path[:, -2:]

    mass = np.dot(np.arange(N + 1), x_path.T)
    if not np.allclose(mass, mass[0]):
        print(f"Mass not conserved: {mass}")

    return t_vec, x_path, cost_path

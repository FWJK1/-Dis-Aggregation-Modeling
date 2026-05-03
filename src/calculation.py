import numpy as np


def steady_state(x_path, t_steps=100, atol=1e-3):
    length = len(x_path)
    if not np.allclose(x_path[-1], x_path[-20:], atol=atol):
        return 0
    for count in range(1, int(0.5 * length), t_steps):
        if not np.allclose(x_path[-1], x_path[-count], atol=atol):
            return length - count
        if np.allclose(x_path[-1], x_path[count], atol=atol):
            return count
    return 0


def calculate_domination(x_path, cutoff: float, sizes=None):
    N = x_path.shape[1] - 1
    if sizes is None:
        sizes = np.arange(N + 1)
    min_dom_idx = int(N * cutoff)
    dominating_prob = x_path[-1, min_dom_idx:]
    dominating_masses = dominating_prob * sizes[min_dom_idx:]
    return dominating_masses.sum()


def steady_state_slope(cost_path, t_vec, s_idx):
    if s_idx:
        grad = np.gradient(cost_path[:, 1], t_vec)
        steady_grad = grad[s_idx:].mean()
        return steady_grad
    return 0

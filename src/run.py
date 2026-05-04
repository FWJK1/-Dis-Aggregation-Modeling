# library imports
import argparse
from itertools import product
from pathlib import Path

# package imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# project imports
from src.viz import heatmap_cluster_sizes
from src.integration import agg_frag_meq
from src.calculation import (
    steady_state,
    calculate_domination,
    steady_state_slope,
)

# package global configs
plt.style.use(["ggplot"])

# Define page width
IMG_WIDTH = 6
IMG_HEIGHT = 6

initial_condition_dict = {
    "All Monomer": {1: 1},
    "Di_Monomer": {1: 0.5, 2: 0.25},
    "First 10": {i: 1 / (i * 10) for i in range(1, 11)},
}

INIT_CONDS = ["All Monomer", "Di_Monomer", "First 10"]
KERNELS = [
    "product",
    "r_prod",
    "sum",
    "constant",
    "dunbar",
    "chipping_1",
    "chipping_5",
]

# 3 * 7 * 7 = 147
COMBOS = list(product(INIT_CONDS, KERNELS, KERNELS))


def main_loop(
    initial_cond,
    agg_kernel,
    frg_kernel,
    task_id,
    N=100,
    t_length=100,
):
    agg_multipliers = [0.01, 0.1, 1]
    frg_multipliers = [0.01, 0.1, 1]
    count = len(agg_multipliers) * len(frg_multipliers)

    results_dict = {
        "agg_kernel": [agg_kernel] * count,
        "agg_multiplier": [],
        "frg_kernel": [frg_kernel] * count,
        "frg_multiplier": [],
        "N": [N] * count,
        "t_length": [t_length] * count,
        "s_idx": [],
        "s_slope": [],
        "init_cond": [initial_cond] * count,
        "dom60": [],
        "dom80": [],
    }
    ncols = 3
    nrows = (count + 1) // ncols
    sizes = np.arange(N + 1)

    _, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(IMG_WIDTH * ncols, IMG_HEIGHT * nrows),
    )
    axes = axes.flatten()
    for i, (
        agg_multiplier,
        frg_multiplier,
    ) in enumerate(
        product(
            agg_multipliers,
            frg_multipliers,
        )
    ):
        agg_rule = (agg_kernel, agg_multiplier)
        frag_rule = (frg_kernel, frg_multiplier)
        t_steps = 100
        ax = axes[i]
        title_string = rf"$N={N}, K={agg_multiplier}*${agg_kernel}, $F={frg_multiplier}*${frg_kernel}"

        t_vec, x_path, cost_path = agg_frag_meq(
            agg_rule=agg_rule,
            frag_rule=frag_rule,
            N=N,
            t_length=t_length,
            t_steps=t_steps,
            initial_conditions=initial_condition_dict[initial_cond],
        )

        heatmap_cluster_sizes(ax, t_length, x_path, title_string=title_string)

        s_idx = steady_state(x_path)
        s_slope = steady_state_slope(cost_path, t_vec, s_idx)
        dom60 = calculate_domination(x_path, 0.6, sizes)
        dom80 = calculate_domination(x_path, 0.8, sizes)
        ax.text(
            0.5,
            -0.2,
            rf"$s={s_idx // t_length},\ \dot{{\kappa}}_s={s_slope:.3f},\ g_{{0.6}}={dom60:.3f}$",
            transform=ax.transAxes,
            ha="center",
        )
        print(f"finished block {i}")
        results_dict["agg_multiplier"].append(agg_multiplier)
        results_dict["frg_multiplier"].append(frg_multiplier)
        results_dict["s_idx"].append(s_idx)
        results_dict["s_slope"].append(s_slope)
        results_dict["dom60"].append(dom60)
        results_dict["dom80"].append(dom80)

    df = pd.DataFrame(results_dict)
    out_dir = Path("project/data/per_task")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"results_{task_id:03d}.csv", index=False)

    fig_dir = Path("project/figs")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / f"HEATPLOT__{initial_cond}-{agg_kernel}-{frg_kernel}-{N}.pdf"
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--t_length", type=int, default=100)
    args = parser.parse_args()

    if not 0 <= args.task_id < len(COMBOS):
        raise SystemExit(f"task_id {args.task_id} out of range [0, {len(COMBOS)})")

    init_cond, agg_kernel, frg_kernel = COMBOS[args.task_id]
    print(
        f"task_id={args.task_id}: init_cond={init_cond!r}, "
        f"agg_kernel={agg_kernel!r}, frg_kernel={frg_kernel!r}"
    )
    main_loop(
        init_cond,
        agg_kernel,
        frg_kernel,
        task_id=args.task_id,
        N=args.N,
        t_length=args.t_length,
    )

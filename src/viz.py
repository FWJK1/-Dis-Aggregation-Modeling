import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors


def heatmap_cluster_sizes(
    ax, t_length, x_path, title_string="", shown_steps=5
):
    if ax is not None:
        sns.heatmap(
            x_path.T,
            ax=ax,
            cmap="viridis",
            norm=mcolors.LogNorm(vmin=1e-6, vmax=1),
        )
        ax.set_ylabel("Cluster Size")
        ax.set_xticks(np.linspace(0, x_path.shape[0], shown_steps))
        ax.set_xticklabels(
            np.round(np.linspace(0, t_length, shown_steps), 1)
        )
        ax.set_title(title_string)

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.letters import load_patterns

BG_COLOR = '#fff5ec'


def compute_dot_matrix(stored: dict) -> tuple[np.ndarray, list[str]]:
    names = list(stored.keys())
    flat = {n: p.flatten() for n, p in stored.items()}
    N = len(next(iter(flat.values())))
    n = len(names)
    matrix = np.zeros((n, n))
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            matrix[i, j] = float(np.dot(flat[ni], flat[nj])) / N
    return matrix, names


def plot_heatmap(matrix, names, out_path):
    n = len(names)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "ortho", ["#378ADD", "#fff5ec", "#D85A30"], N=256
    )

    fig, ax = plt.subplots(figsize=(max(4, n * 1.1 + 1), max(4, n * 1.1 + 1)),
                           facecolor=BG_COLOR)
    fig.set_dpi(150)
    ax.set_facecolor(BG_COLOR)

    x_edges = np.arange(n + 1) - 0.5
    y_edges = np.arange(n + 1) - 0.5
    mesh = ax.pcolormesh(
        x_edges, y_edges, matrix,
        cmap=cmap, vmin=-1, vmax=1,
        edgecolors='black', linewidth=0.8,
    )
    ax.invert_yaxis()

    ax.set_xticks(range(n))
    ax.set_xticklabels(names, fontsize=11, fontweight='bold', color='#333')
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=11, fontweight='bold', color='#333')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            txt_color = 'white' if abs(val) > 0.55 else '#333'
            ax.text(j, i, f"{val:.2f}",
                    ha='center', va='center',
                    fontsize=10, fontweight='bold', color=txt_color)

    cbar = fig.colorbar(mesh, ax=ax, orientation='vertical',
                        fraction=0.046, pad=0.04, shrink=0.8)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_ticklabels(['-1\n(inverse)', '-0.5', '0\n(orthogonal)', '0.5', '1\n(identical)'])
    cbar.ax.tick_params(labelsize=7)

    ax.set_title("Dot product per pair (normalized by N)",
                 fontsize=10, color='#444', pad=14)
    for spine in ax.spines.values():
        spine.set_edgecolor('#ccc')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Saved: {out_path}")





def print_summary(matrix, names):
    n = len(names)
    mask = ~np.eye(n, dtype=bool)
    print("\nNormalized dot product:")
    print(f"{'':>4}" + "".join(f"{name:>8}" for name in names))
    for i, ni in enumerate(names):
        row = f"{ni:>4}" + "".join(f"{matrix[i,j]:>8.2f}" for j in range(n))
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Pattern orthogonality analysis")
    parser.add_argument("patterns_file")
    parser.add_argument("--out-dir", default=".",
                        help="Directory to write PNGs into (default: current dir)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stored = load_patterns(args.patterns_file)
    if not stored:
        raise ValueError(f"No patterns found in {args.patterns_file}")

    matrix, names = compute_dot_matrix(stored)

    plot_heatmap(matrix, names, out_dir / "heatmap.png")
    print_summary(matrix, names)


if __name__ == "__main__":
    main()
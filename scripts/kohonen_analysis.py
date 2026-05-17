"""
Kohonen network analysis driven by a JSON configuration file.

Usage:
    python scripts/kohonen_analysis.py --config configs/kohonen_5x5_default.json

The script:
    1. Loads hyperparameters from the JSON config.
    2. Trains a Kohonen network on the (standardized) Europe dataset.
    3. Generates three plots into results/plots/<config_name>/:
        - countries.png   : country labels per winning neuron
        - umatrix.png     : average distance between neighboring neurons
        - activations.png : count of countries per neuron
    4. Prints and saves a summary of the clusters found.

Add new experiments by dropping a new JSON file under configs/ — no code
changes needed.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.kohonen import Kohonen
from utils.preprocessing import load_europe


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train and analyze a Kohonen SOM.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the JSON config file (e.g. configs/kohonen_5x5_default.json).",
    )
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
})


def _add_grid(ax, k):
    """Draw thin white lines between cells."""
    for x in np.arange(-0.5, k, 1):
        ax.axhline(x, color="white", linewidth=1.2)
        ax.axvline(x, color="white", linewidth=1.2)


def _text_color(bg_norm):
    """White text on dark backgrounds, dark text on light ones."""
    return "white" if bg_norm > 0.5 else "#1a1a1a"


# -----------------------------------------------------------------------------
# Plot functions
# -----------------------------------------------------------------------------
def plot_countries(red, X, countries, K, out_path):
    activations = red.activations_per_neuron(X)

    labels = {}
    for country, x in zip(countries, X):
        i, j = red._winner(x)
        labels.setdefault((i, j), []).append(country)

    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(activations, cmap="Blues", origin="upper", vmin=0)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("países por neurona", fontsize=10)

    act_max = activations.max() or 1
    for (i, j), names in labels.items():
        text = "\n".join(names)
        color = _text_color(activations[i, j] / act_max)
        ax.text(j, i, text, ha="center", va="center",
                fontsize=8, fontweight="bold", color=color,
                linespacing=1.4)

    _add_grid(ax, K)
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([f"col {j}" for j in range(K)])
    ax.set_yticklabels([f"fila {i}" for i in range(K)])
    ax.set_title(f"Países agrupados por la red de Kohonen ({K}×{K})", pad=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return labels


def plot_umatrix(red, K, out_path):
    u = red.u_matrix()

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(u, cmap="gray_r", origin="upper")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("distancia promedio a vecinas", fontsize=10)

    u_min, u_max = u.min(), u.max()
    for i in range(K):
        for j in range(K):
            norm = (u[i, j] - u_min) / (u_max - u_min + 1e-9)
            color = _text_color(norm)
            ax.text(j, i, f"{u[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    _add_grid(ax, K)
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([f"col {j}" for j in range(K)])
    ax.set_yticklabels([f"fila {i}" for i in range(K)])
    ax.set_title("U-matrix — distancia promedio entre neuronas vecinas\n"
                 "Zonas oscuras indican fronteras entre clusters", pad=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return u


def plot_activations(red, X, K, out_path):
    activations = red.activations_per_neuron(X)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(activations, cmap="YlOrRd", origin="upper", vmin=0)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("cantidad de países", fontsize=10)

    act_max = activations.max() or 1
    for i in range(K):
        for j in range(K):
            norm = activations[i, j] / act_max
            color = _text_color(norm)
            label = str(activations[i, j]) if activations[i, j] > 0 else "—"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    _add_grid(ax, K)
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([f"col {j}" for j in range(K)])
    ax.set_yticklabels([f"fila {i}" for i in range(K)])
    ax.set_title("Activaciones por neurona — cuántos países caen en cada celda", pad=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return activations


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    experiment_name = config_path.stem
    out_dir = REPO_ROOT / "results" / "plots" / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {experiment_name}")
    print(f"Config:     {config_path}")
    print(f"Output dir: {out_dir}")
    print(f"Hyperparameters: {json.dumps(config, indent=2)}")

    countries, X, variables = load_europe()
    print(f"\nLoaded {len(countries)} samples with {X.shape[1]} variables")

    K = config["k"]
    red = Kohonen(
        k=K,
        input_dim=X.shape[1],
        eta_0=config["eta_0"],
        radius_0=config.get("radius_0"),
        weight_init=config.get("weight_init", "samples"),
        similarity=config.get("similarity", "euclidean"),
        seed=config.get("seed"),
    )
    red.fit(X, n_iter=config.get("n_iter"))

    print("\nGenerating plots...")
    labels = plot_countries(red, X, countries, K, out_dir / "countries.png")
    print(f"  ✓ {out_dir / 'countries.png'}")

    u = plot_umatrix(red, K, out_dir / "umatrix.png")
    print(f"  ✓ {out_dir / 'umatrix.png'}")

    activations = plot_activations(red, X, K, out_dir / "activations.png")
    print(f"  ✓ {out_dir / 'activations.png'}")

    print("\nClusters encontrados:")
    print("-" * 50)
    for (i, j), names in sorted(labels.items()):
        print(f"  celda ({i}, {j}): {', '.join(names)}")

    print(f"\nNeuronas activas: {(activations > 0).sum()} de {K * K}")
    print(f"Neuronas muertas: {(activations == 0).sum()}")
    print(f"Distancia máxima a vecinos (frontera más fuerte): "
          f"{u.max():.3f} en celda {np.unravel_index(u.argmax(), u.shape)}")

    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Config: {json.dumps(config, indent=2)}\n\n")
        f.write("Clusters encontrados:\n")
        for (i, j), names in sorted(labels.items()):
            f.write(f"  celda ({i}, {j}): {', '.join(names)}\n")
        f.write(f"\nNeuronas activas: {(activations > 0).sum()} de {K * K}\n")
        f.write(f"Neuronas muertas: {(activations == 0).sum()}\n")
        f.write(f"Distancia máxima a vecinos (frontera más fuerte): "
                f"{u.max():.3f} en celda {np.unravel_index(u.argmax(), u.shape)}\n")
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()

"""
Kohonen network analysis driven by a JSON configuration file.

Usage:
    python scripts/kohonen_analysis.py --config configs/kohonen_5x5_default.json

The script:
    1. Loads hyperparameters from the JSON config.
    2. Trains a Kohonen network on the (standardized) Europe dataset.
    3. Generates four plots into results/plots/<config_name>/:
        - countries.png          : country labels + activation count per neuron (plasma)
        - umatrix_countries.png  : u-matrix values + country labels overlaid
        - variables.png          : average value per neuron, one panel per variable
        - europe_geographic.png  : clusters projected onto a real map of Europe
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
import matplotlib.patheffects as pe

try:
    import geopandas as gpd
    _HAS_GEOPANDAS = True
except ImportError:
    _HAS_GEOPANDAS = False

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.kohonen import Kohonen
from utils.preprocessing import load_europe


# Natural Earth may use different country name variants than the CSV.
_CSV_TO_GEO_NAME = {
    "Czech Republic": "Czechia",
}


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
    """White text on dark backgrounds, dark text on light ones (gray_r scale)."""
    return "white" if bg_norm > 0.5 else "#1a1a1a"


def _plasma_text_color(norm):
    """For plasma colormap: white on dark (low values), black from orange onward."""
    return "#1a1a1a" if norm >= 0.6 else "white"


# -----------------------------------------------------------------------------
# Plot functions
# -----------------------------------------------------------------------------
def plot_countries(red, X, countries, K, out_path):
    activations = red.activations_per_neuron(X)

    labels = {}
    for country, x in zip(countries, X):
        i, j = red._winner(x)
        labels.setdefault((int(i), int(j)), []).append(country)

    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(activations, cmap="plasma", origin="upper", vmin=0)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("países por neurona", fontsize=10)

    act_max = activations.max() or 1
    for (i, j), names in labels.items():
        norm = activations[i, j] / act_max
        color = _plasma_text_color(norm)
        count_label = str(activations[i, j])
        country_text = "\n".join(names)
        ax.text(j, i - 0.22, count_label, ha="center", va="center",
                fontsize=11, fontweight="bold", color=color)
        ax.text(j, i + 0.18, country_text, ha="center", va="center",
                fontsize=7, color=color, linespacing=1.3)

    # Dead neurons: show dash
    for i in range(K):
        for j in range(K):
            if (i, j) not in labels:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=12, color="white", alpha=0.4)

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



def plot_umatrix_with_countries(red, X, countries, K, out_path):
    u = red.u_matrix()

    labels = {}
    for country, x in zip(countries, X):
        i, j = red._winner(x)
        labels.setdefault((int(i), int(j)), []).append(country)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(u, cmap="gray_r", origin="upper")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("distancia promedio a vecinas", fontsize=10)

    u_min, u_max = u.min(), u.max()
    for i in range(K):
        for j in range(K):
            norm = (u[i, j] - u_min) / (u_max - u_min + 1e-9)
            color = _text_color(norm)
            # U-matrix value: larger, at top of cell
            ax.text(j, i - 0.2, f"{u[i, j]:.2f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)
            # Country names: smaller, below the value
            if (i, j) in labels:
                names_text = "\n".join(labels[(i, j)])
                ax.text(j, i + 0.18, names_text, ha="center", va="center",
                        fontsize=6, color=color, linespacing=1.3)

    _add_grid(ax, K)
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([f"col {j}" for j in range(K)])
    ax.set_yticklabels([f"fila {i}" for i in range(K)])
    ax.set_title("U-matrix con países\n"
                 "Zonas oscuras indican fronteras entre clusters", pad=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_variable_heatmaps(red, X, countries, variables, K, out_path):
    """One heatmap per variable: average value of that variable per neuron."""
    # Build cell -> list of sample indices
    cell_samples = {}
    for idx, x in enumerate(X):
        i, j = red._winner(x)
        cell_samples.setdefault((i, j), []).append(idx)

    n_vars = len(variables)
    ncols = 4
    nrows = (n_vars + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 3.2))
    axes = axes.flatten()

    for v, var_name in enumerate(variables):
        grid = np.full((K, K), np.nan)
        for (i, j), idxs in cell_samples.items():
            grid[i, j] = X[idxs, v].mean()

        ax = axes[v]
        v_min, v_max = np.nanmin(grid), np.nanmax(grid)
        im = ax.imshow(grid, cmap="plasma", origin="upper",
                       vmin=v_min, vmax=v_max)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for i in range(K):
            for j in range(K):
                if np.isnan(grid[i, j]):
                    continue
                norm = (grid[i, j] - v_min) / (v_max - v_min + 1e-9)
                color = _plasma_text_color(norm)
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, fontweight="bold", color=color)

        _add_grid(ax, K)
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels([str(j) for j in range(K)], fontsize=7)
        ax.set_yticklabels([str(i) for i in range(K)], fontsize=7)
        ax.set_title(var_name, fontsize=11, fontweight="bold", pad=6)

    # Hide unused subplots
    for ax in axes[n_vars:]:
        ax.set_visible(False)

    fig.suptitle("Valor promedio por neurona — una variable a la vez",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()



def plot_europe_map(labels, experiment_name, out_path):
    """Color each European country by its winning neuron using a plasma palette."""
    if not _HAS_GEOPANDAS:
        print("  ! geopandas not installed — skipping europe_geographic.png")
        return

    # Invert labels {cell: [countries]} -> {country: cell}
    country_to_cell = {c: cell for cell, names in labels.items() for c in names}
    unique_cells = sorted(set(country_to_cell.values()))
    cell_to_cluster = {cell: idx for idx, cell in enumerate(unique_cells)}
    n_clusters = len(unique_cells)

    try:
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    except (AttributeError, FileNotFoundError):
        url = ("https://naciscdn.org/naturalearth/110m/cultural/"
               "ne_110m_admin_0_countries.zip")
        world = gpd.read_file(url)
        world = world.rename(columns={"NAME": "name", "CONTINENT": "continent"})

    europe = world[world["continent"] == "Europe"].copy()

    def assign_cluster(geo_name):
        for csv_name, cell in country_to_cell.items():
            if _CSV_TO_GEO_NAME.get(csv_name, csv_name) == geo_name:
                return float(cell_to_cluster[cell])
        return np.nan

    europe["cluster"] = europe["name"].apply(assign_cluster)

    cmap = plt.colormaps["plasma"].resampled(n_clusters)

    fig, ax = plt.subplots(figsize=(14, 12))

    europe[europe["cluster"].isna()].plot(
        ax=ax, color="lightgrey", edgecolor="#555555",
        linewidth=0.4, hatch="///",
    )
    europe[europe["cluster"].notna()].plot(
        column="cluster", ax=ax,
        cmap=cmap, vmin=0, vmax=max(n_clusters - 1, 1),
        edgecolor="black", linewidth=0.5,
    )

    stroke = [pe.withStroke(linewidth=2.5, foreground="black")]
    for _, row in europe[europe["cluster"].notna()].iterrows():
        centroid = row["geometry"].centroid
        ax.annotate(row["name"], xy=(centroid.x, centroid.y),
                    ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white",
                    path_effects=stroke)

    legend_handles = []
    for cell, cluster_id in cell_to_cluster.items():
        color = cmap(cluster_id / max(n_clusters - 1, 1))
        members = [c for c, cc in country_to_cell.items() if cc == cell]
        label = f"celda {cell}: {', '.join(members)}"
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc=color, ec="black",
                           linewidth=0.5, label=label)
        )
    legend_handles.append(
        plt.Rectangle((0, 0), 1, 1, fc="lightgrey", ec="#555555",
                       linewidth=0.5, hatch="///", label="no incluido")
    )
    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(1.02, 1), borderaxespad=0,
              fontsize=7, framealpha=0.92,
              title="Clusters de Kohonen", title_fontsize=8)

    ax.set_xlim(-25, 45)
    ax.set_ylim(34, 72)
    ax.set_aspect("equal")
    ax.set_title(
        f"Clusters de Kohonen proyectados sobre Europa — {experiment_name}\n"
        "Cada color representa una neurona ganadora distinta",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Longitud", fontsize=10)
    ax.set_ylabel("Latitud", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


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

    u = red.u_matrix()

    plot_umatrix_with_countries(red, X, countries, K, out_dir / "umatrix_countries.png")
    print(f"  ✓ {out_dir / 'umatrix_countries.png'}")

    plot_variable_heatmaps(red, X, countries, variables, K, out_dir / "variables.png")
    print(f"  ✓ {out_dir / 'variables.png'}")

    plot_europe_map(labels, experiment_name, out_dir / "europe_geographic.png")
    print(f"  ✓ {out_dir / 'europe_geographic.png'}")

    print("\nClusters encontrados:")
    print("-" * 50)
    for (i, j), names in sorted(labels.items()):
        print(f"  celda ({i}, {j}): {', '.join(names)}")

    n_active = len(labels)
    n_dead = K * K - n_active
    max_cell = tuple(int(x) for x in np.unravel_index(u.argmax(), u.shape))

    print(f"\nNeuronas activas: {n_active} de {K * K}")
    print(f"Neuronas muertas: {n_dead}")
    print(f"Distancia máxima a vecinos (frontera más fuerte): "
          f"{u.max():.3f} en celda {max_cell}")

    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Config: {json.dumps(config, indent=2)}\n\n")
        f.write("Clusters encontrados:\n")
        for (i, j), names in sorted(labels.items()):
            f.write(f"  celda ({i}, {j}): {', '.join(names)}\n")
        f.write(f"\nNeuronas activas: {n_active} de {K * K}\n")
        f.write(f"Neuronas muertas: {n_dead}\n")
        f.write(f"Distancia máxima a vecinos (frontera más fuerte): "
                f"{u.max():.3f} en celda {max_cell}\n")
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()

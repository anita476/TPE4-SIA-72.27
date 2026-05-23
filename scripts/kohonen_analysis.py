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
import textwrap
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
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.transparent": True,
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
                fontsize=12, fontweight="bold", color=color)
        ax.text(j, i + 0.18, country_text, ha="center", va="center",
                fontsize=12, color=color, linespacing=1.3)

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
    plt.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
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
                    fontsize=12, fontweight="bold", color=color)
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
    plt.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
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
    plt.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close()



def plot_cluster_profiles(labels, X, countries, variables, out_path):
    """
    One mini bar chart per active cluster. Each bar = mean of a variable for
    countries in that cluster; error bar = ±1 std dev.
    Tall bar + small error → defining characteristic of the cluster.
    Tall bar + large error → variable is not cohesive in that cluster.
    """
    country_to_idx = {c: i for i, c in enumerate(countries)}
    active_cells = sorted(labels.items())
    n_clusters = len(active_cells)

    ncols = 4
    nrows = (n_clusters + ncols - 1) // ncols
    bar_colors = plt.cm.tab10(np.linspace(0, 0.7, len(variables)))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.8, nrows * 3.2),
                             sharey=True)
    axes = axes.flatten()

    x = np.arange(len(variables))

    for idx, (cell, names) in enumerate(active_cells):
        ax = axes[idx]
        idxs = [country_to_idx[c] for c in names]
        samples = X[idxs]
        means = samples.mean(axis=0)
        stds = samples.std(axis=0) if len(idxs) > 1 else np.zeros(len(variables))

        bars = ax.bar(x, means, yerr=stds, capsize=4,
                      color=bar_colors, edgecolor="black", linewidth=0.5,
                      error_kw=dict(elinewidth=1.2, ecolor="black", capthick=1.2))

        ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(variables, rotation=40, ha="right", fontsize=8)
        ax.set_title(f"celda {cell}\n{', '.join(names)}",
                     fontsize=8, fontweight="bold", pad=4)
        ax.set_ylabel("std scores" if idx % ncols == 0 else "", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[n_clusters:]:
        ax.set_visible(False)

    fig.suptitle("Perfil de cada cluster — media ± desvío por variable (datos estandarizados)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close()


def plot_cohesion_table(labels, X, countries, variables, out_path):
    """
    For each active neuron, compute the std dev of each variable across its
    countries. Low std = countries in that cell are genuinely similar.
    Saves a PNG table sorted by mean std (best clusters first).
    """
    country_to_idx = {c: i for i, c in enumerate(countries)}

    rows = []
    for cell, names in sorted(labels.items()):
        idxs = [country_to_idx[c] for c in names]
        samples = X[idxs]
        stds = samples.std(axis=0) if len(idxs) > 1 else np.zeros(X.shape[1])
        rows.append({
            "Celda": str(cell),
            "Países": ", ".join(names),
            **{var: round(float(std), 3) for var, std in zip(variables, stds)},
            "Media": round(float(stds.mean()), 3),
        })

    rows.sort(key=lambda r: r["Media"])

    # Save CSV alongside the PNG
    import pandas as pd
    csv_path = Path(str(out_path).replace(".png", ".csv"))
    pd.DataFrame(rows).rename(columns={"Media": "Media std"}).to_csv(csv_path, index=False)

    col_labels = ["Celda", "Países"] + list(variables) + ["Media std"]
    cell_data = [
        [r["Celda"], r["Países"]] + [r[v] for v in variables] + [r["Media"]]
        for r in rows
    ]

    n_rows = len(cell_data)
    n_cols = len(col_labels)

    fig, ax = plt.subplots(figsize=(max(14, n_cols * 1.4), 0.5 + 0.55 * (n_rows + 1)))
    ax.axis("off")

    tbl = ax.table(cellText=cell_data, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    # Header
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor("#2E3A4E")
        cell.set_text_props(color="white", fontweight="bold")

    # Rows: alternating bg, highlight "Países" and "Media" columns
    for i, row in enumerate(rows, start=1):
        bg = "#F2F4F7" if i % 2 == 0 else "white"
        # Color the "Media std" cell by quality: green=low, red=high
        mean_val = row["Media"]
        all_means = [r["Media"] for r in rows]
        norm = (mean_val - min(all_means)) / (max(all_means) - min(all_means) + 1e-9)
        media_bg = plt.cm.RdYlGn_r(norm * 0.7 + 0.15)  # avoid extremes

        for j in range(n_cols):
            c = tbl[i, j]
            if j == n_cols - 1:  # Media column
                c.set_facecolor(media_bg)
                c.set_text_props(fontweight="bold")
            elif j == 1:  # Países column
                c.set_facecolor(bg)
                c.set_text_props(color="#2E3A4E", fontweight="bold")
                c.set_width(0.35)
            else:
                c.set_facecolor(bg)
            c.set_edgecolor("#CCCCCC")

    fig.patch.set_alpha(0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
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
        wrapped = textwrap.fill(", ".join(members), width=28)
        label = f"celda {cell}:\n{wrapped}"
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
    plt.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close()


def plot_schedules(eta_0, radius_0, n_iter, out_path):
    t = np.arange(n_iter)
    eta    = eta_0 / (t + 1)
    radius = np.maximum(1.0, radius_0 / (t + 1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(t, eta, color="#1E88E5", linewidth=2)
    ax1.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax1.set_ylabel("η(t)  —  learning rate")
    ax1.set_ylim(bottom=0)
    ax1.set_facecolor("none")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(alpha=0.2, linestyle="--")
    ax1.text(0.97, 0.82, r"$\eta(t) = \eta_0\,/\,(t+1)$",
             transform=ax1.transAxes, ha="right", fontsize=12,
             color="#1E88E5",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6))

    ax2.plot(t, radius, color="#FB8C00", linewidth=2)
    ax2.axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.5,
                label="floor = 1")
    ax2.set_ylabel("R(t)  —  neighborhood radius")
    ax2.set_xlabel("iteration  t")
    ax2.set_ylim(bottom=0)
    ax2.set_facecolor("none")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(alpha=0.2, linestyle="--")
    ax2.legend(framealpha=0.6)
    ax2.text(0.97, 0.82, r"$R(t) = \max\!\left(1,\; R_0\,/\,(t+1)\right)$",
             transform=ax2.transAxes, ha="right", fontsize=12,
             color="#FB8C00",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6))

    t_floor_r = int(radius_0 - 1)
    if 0 < t_floor_r < n_iter:
        ax2.axvline(t_floor_r, color="#FB8C00", linewidth=1,
                    linestyle="--", alpha=0.6)
        ax2.text(t_floor_r + n_iter * 0.01,
                 ax2.get_ylim()[1] * 0.6, "R = 1 (floor)",
                 fontsize=9, color="#FB8C00", alpha=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
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
    n_iter = config.get("n_iter") or 500 * X.shape[1]
    red.fit(X, n_iter=n_iter)

    print("\nGenerating plots...")
    plot_schedules(config["eta_0"], config.get("radius_0") or float(K),
                   n_iter, out_dir / "schedules.png")
    print(f"  ✓ {out_dir / 'schedules.png'}")

    labels = plot_countries(red, X, countries, K, out_dir / "countries.png")
    print(f"  ✓ {out_dir / 'countries.png'}")

    u = red.u_matrix()

    plot_umatrix_with_countries(red, X, countries, K, out_dir / "umatrix_countries.png")
    print(f"  ✓ {out_dir / 'umatrix_countries.png'}")

    plot_variable_heatmaps(red, X, countries, variables, K, out_dir / "variables.png")
    print(f"  ✓ {out_dir / 'variables.png'}")

    plot_europe_map(labels, experiment_name, out_dir / "europe_geographic.png")
    print(f"  ✓ {out_dir / 'europe_geographic.png'}")

    plot_cohesion_table(labels, X, countries, variables, out_dir / "cohesion_table.png")
    print(f"  ✓ {out_dir / 'cohesion_table.png'}")
    print(f"  ✓ {out_dir / 'cohesion_table.csv'}")

    plot_cluster_profiles(labels, X, countries, variables, out_dir / "cluster_profiles.png")
    print(f"  ✓ {out_dir / 'cluster_profiles.png'}")

    print("\nClusters encontrados:")
    print("-" * 50)
    for (i, j), names in sorted(labels.items()):
        print(f"  celda ({i}, {j}): {', '.join(names)}")

    n_active = len(labels)
    n_dead = K * K - n_active
    max_cell = tuple(int(x) for x in np.unravel_index(u.argmax(), u.shape))
    qe = red.quantization_error(X)

    print(f"\nNeuronas activas: {n_active} de {K * K}")
    print(f"Neuronas muertas: {n_dead}")
    print(f"Quantization error: {qe:.4f}")
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
        f.write(f"Quantization error: {qe:.4f}\n")
        f.write(f"Distancia máxima a vecinos (frontera más fuerte): "
                f"{u.max():.3f} en celda {max_cell}\n")
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()

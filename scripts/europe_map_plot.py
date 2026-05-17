"""
Project Kohonen clusters onto a geographical map of Europe.

For a trained Kohonen network, this script colors each country on a real
map of Europe according to its winning neuron — so coherence between
economic clusters and geographic regions becomes visually obvious.

Usage:
    python scripts/europe_map_plot.py --config configs/kohonen_5x5_default.json

Output: results/plots/<config_name>/europe_geographic.png

Requires:
    pip install geopandas matplotlib
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.kohonen import Kohonen
from utils.preprocessing import load_europe


# Natural Earth may use different country name variants than the CSV.
CSV_TO_GEO_NAME = {
    "Czech Republic": "Czechia",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Kohonen clusters on a Europe map.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the JSON config used for training.")
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return json.load(f)


def train_network(config, X):
    red = Kohonen(
        k=config["k"],
        input_dim=X.shape[1],
        eta_0=config["eta_0"],
        radius_0=config.get("radius_0"),
        weight_init=config.get("weight_init", "samples"),
        similarity=config.get("similarity", "euclidean"),
        seed=config.get("seed"),
    )
    red.fit(X, n_iter=config.get("n_iter"))
    return red


def main():
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    experiment_name = config_path.stem

    out_dir = REPO_ROOT / "results" / "plots" / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Train and compute winning neurons
    # -------------------------------------------------------------------------
    countries, X, _ = load_europe()
    red = train_network(config, X)

    country_to_cell = {}
    for country, x in zip(countries, X):
        i, j = red._winner(x)
        country_to_cell[country] = (int(i), int(j))

    unique_cells = sorted(set(country_to_cell.values()))
    cell_to_cluster = {cell: idx for idx, cell in enumerate(unique_cells)}
    country_to_cluster = {c: cell_to_cluster[cell]
                          for c, cell in country_to_cell.items()}

    n_clusters = len(unique_cells)
    print(f"Found {n_clusters} unique cells (clusters) for {len(countries)} countries")

    # -------------------------------------------------------------------------
    # 2. Load European shapefile
    # -------------------------------------------------------------------------
    try:
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    except (AttributeError, FileNotFoundError):
        url = ("https://naciscdn.org/naturalearth/110m/cultural/"
               "ne_110m_admin_0_countries.zip")
        world = gpd.read_file(url)
        world = world.rename(columns={"NAME": "name", "CONTINENT": "continent"})

    europe = world[world["continent"] == "Europe"].copy()

    # -------------------------------------------------------------------------
    # 3. Assign cluster id to each country in the shapefile
    # -------------------------------------------------------------------------
    def assign_cluster(geo_name):
        for csv_name, cluster_id in country_to_cluster.items():
            if CSV_TO_GEO_NAME.get(csv_name, csv_name) == geo_name:
                return float(cluster_id)
        return np.nan

    europe["cluster"] = europe["name"].apply(assign_cluster)

    matched = europe["cluster"].notna().sum()
    print(f"Countries matched on map: {matched}")
    not_matched = [c for c in countries
                   if CSV_TO_GEO_NAME.get(c, c) not in europe["name"].values]
    if not_matched:
        print(f"Countries not found in shapefile: {not_matched}")

    # -------------------------------------------------------------------------
    # 4. Plot
    # -------------------------------------------------------------------------
    # Use plasma: same palette as the activations heatmap, normalized over
    # the number of distinct clusters so each gets a well-separated color.
    cmap = plt.colormaps["plasma"].resampled(n_clusters)

    fig, ax = plt.subplots(figsize=(14, 12))

    # Countries not in the dataset: light grey hatched
    europe[europe["cluster"].isna()].plot(
        ax=ax,
        color="lightgrey",
        edgecolor="#555555",
        linewidth=0.4,
        hatch="///",
    )

    # Countries in the dataset: colored by cluster
    europe[europe["cluster"].notna()].plot(
        column="cluster",
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=n_clusters - 1,
        edgecolor="black",
        linewidth=0.5,
    )

    # Country name labels
    for _, row in europe[europe["cluster"].notna()].iterrows():
        centroid = row["geometry"].centroid
        ax.annotate(
            row["name"],
            xy=(centroid.x, centroid.y),
            ha="center", va="center",
            fontsize=7, fontweight="bold", color="white",
        )

    # Legend: one entry per cluster cell
    legend_handles = []
    for cell, cluster_id in cell_to_cluster.items():
        color = cmap(cluster_id / max(n_clusters - 1, 1))
        members = [c for c, cc in country_to_cell.items() if cc == cell]
        wrapped = textwrap.fill(", ".join(members), width=28)
        label = f"celda {cell}:\n{wrapped}"
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc=color, ec="black", linewidth=0.5,
                           label=label)
        )
    legend_handles.append(
        plt.Rectangle((0, 0), 1, 1, fc="lightgrey", ec="#555555", linewidth=0.5,
                       hatch="///", label="no incluido en el dataset")
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

    out_path = out_dir / "europe_geographic.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Saved: {out_path}")


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_pca(seed, data_dir: Path, out_dir: Path):
    df = pd.read_csv(data_dir)
    countries = df['Country']
    X = df.drop(columns=['Country'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca  = PCA(n_components=1, random_state=seed)
    X_pca   = pca.fit_transform(X_scaled)

    var  = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, s=50)

    for i, country in enumerate(countries):
        ax.annotate(country, (X_pca[i, 0], X_pca[i, 1]), fontsize=8)

    ax.set_xlabel(f"PC1 ({var[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% variance)")
    ax.set_title("PCA projection — Europe")
    ax.legend()

    ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")

    fig.tight_layout()
    p = out_dir / "pca.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {p}")

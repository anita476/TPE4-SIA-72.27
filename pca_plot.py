import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_pca(seed, data_dir: Path, out_dir: Path):
    df = pd.read_csv(data_dir)
    countries = df['Country']
    X = df.drop(columns=['Country'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca  = PCA(n_components=2, random_state=seed)
    X_pca   = pca.fit_transform(X_scaled)

    var  = pca.explained_variance_ratio_ * 100
    
    # Get feature names and loadings
    feature_names = X.columns
    loadings = pca.components_
    
    # Find top contributing features for PC1 and PC2
    top_pc1_idx = np.argsort(np.abs(loadings[0]))[-2:][::-1]
    top_pc2_idx = np.argsort(np.abs(loadings[1]))[-2:][::-1]

    top_pc1_features = ", ".join([feature_names[i] for i in top_pc1_idx])
    top_pc2_features = ", ".join([feature_names[i] for i in top_pc2_idx])

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, s=50)

    for i, country in enumerate(countries):
        ax.annotate(country, (X_pca[i, 0], X_pca[i, 1]), fontsize=8)

    features = X.columns
    components = pca.components_
    scale = np.max(np.abs(X_pca)) / np.max(np.abs(components))
    for j, feat in enumerate(features):
        cx, cy = components[0, j] * scale, components[1, j] * scale
        ax.annotate("", xy=(cx, cy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="crimson", lw=1.5))
        ax.text(cx * 1.12, cy * 1.12, feat, color="crimson", fontsize=8, fontweight="bold")

    ax.set_xlabel(f"PC1 ({var[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% variance)")
    ax.set_title("PCA biplot — Europe")

    ax.set_xlabel(f"PC1 ({var[0]:.1f}% variance)\n[{top_pc1_features}]")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% variance)\n[{top_pc2_features}]")

    fig.tight_layout()
    p = out_dir / "pca.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {p}")

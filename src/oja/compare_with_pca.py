import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

from src.oja.oja_neuron import OjaNeuron
from utils.preprocessing import load_europe


def pc1_reference(X):
    pca = PCA(n_components=1)
    pca.fit(X)
    return pca.components_[0], pca.explained_variance_ratio_[0], pca


def compare(w_oja, w_ref):
    dot = float(np.dot(w_oja, w_ref))
    similarity = abs(dot)
    w_aligned = w_oja if dot >= 0 else -w_oja
    return w_aligned, similarity


def main():
    countries, X, features = load_europe(Path("data/europe.csv"))
    pc1_ref, var_ratio, pca = pc1_reference(X)
    oja = OjaNeuron(seed=1)
    w = oja.fit(X)
    w_aligned, similarity = compare(w, pc1_ref)

    table = pd.DataFrame({
        "Oja (alineado)": w_aligned, "PC1 sklearn": pc1_ref, 
        "|diff|": np.abs(w_aligned - pc1_ref)}, index=features).round(4)

    scores_oja = X @ w_aligned
    scores_ref = pca.transform(X).ravel()

    print(f"Varianza explicada por PC1 (sklearn): {var_ratio*100:.2f}%")
    print(f"Norma de w (Oja): {np.linalg.norm(w):.6f}")
    print(f"Similitud |cos|: {similarity:.6f}")
    print(f"Error máximo por componente: {table['|diff|'].max():.6f}")
    print()
    print("Coeficientes:")
    print(table)
    print()

    ranking = pd.DataFrame({
        "Country": countries, "Oja score": scores_oja, "PC1 sklearn": scores_ref,
    }).sort_values("Oja score", ascending=False).round(4).reset_index(drop=True)
    print("Ranking de países")
    print(ranking.to_string(index=False))

if __name__ == "__main__":
    main()
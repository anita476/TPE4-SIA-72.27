from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.oja.oja_neuron import OjaNeuron
from src.oja.sanger import SangerNetwork
from utils.preprocessing import load_europe


DATA_PATH = Path("data/europe.csv")
OUT_DIR = Path("results/oja")

BG_COLOR = "#fff5ec"
plt.rcParams["figure.facecolor"] = BG_COLOR
plt.rcParams["axes.facecolor"] = BG_COLOR
plt.rcParams["savefig.facecolor"] = BG_COLOR

# Helpers
def align_sign(w, w_ref):
    return w if float(np.dot(w, w_ref)) >= 0 else -w


def _sci_latex(value):
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def _nan_stat(fn, arr):
    return float(fn(arr)) if np.any(np.isfinite(arr)) else float("nan")


def _save(fig, out_dir, name):
    fig.tight_layout()
    path = out_dir / name
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _band(ax, x, mean, std, color, *, alpha=0.2, clamp=True, **plot_kw):
    ax.plot(x, mean, color=color, **plot_kw)
    lower = np.maximum(mean - std, 1e-12) if clamp else mean - std
    ax.fill_between(x, lower, mean + std, color=color, alpha=alpha)


def train_with_history(X, lr, epochs, decay, seed=0, init_w=None, **kwargs):
    oja = OjaNeuron(learning_rate=lr, epochs=epochs, decay=decay, seed=seed,
                    record_history=True, **kwargs)
    w_final = oja.fit(X, init_w=init_w)
    return oja.history_, w_final


def _seed_histories(X, pc1_ref, lr, epochs, decay, n_seeds, **kwargs):
    hists = []
    for s in range(n_seeds):
        history, _ = train_with_history(X, lr=lr, epochs=epochs, decay=decay,
                                        seed=s, **kwargs)
        hists.append(np.array([align_sign(w, pc1_ref) for w in history]))
    return np.array(hists)

# Experimentos

def experiment_convergence(X, features, pc1_ref, out_dir,
                           lr, epochs, decay, n_seeds=5):
    hists = _seed_histories(X, pc1_ref, lr, epochs, decay, n_seeds)
    norms = np.linalg.norm(hists, axis=2)            # (n_seeds, epochs+1)
    mean_hist, std_hist = hists.mean(0), hists.std(0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    t = np.arange(hists.shape[1])
    cmap = plt.get_cmap("tab10")
    for j, feat in enumerate(features):
        _band(axes[0], t, mean_hist[:, j], std_hist[:, j], cmap(j % 10),
              alpha=0.15, clamp=False, label=feat, lw=1.4)
    for j in range(len(features)):
        axes[0].axhline(pc1_ref[j], color="grey", ls=":", lw=0.6)
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Valor del peso $w_j$")
    axes[0].set_title("Convergencia de los pesos de Oja")
    axes[0].legend(fontsize=7, loc="best", ncol=2)
    axes[0].grid(alpha=0.3)

    _band(axes[1], t, norms.mean(0), norms.std(0), "crimson", clamp=False, lw=1.6)
    axes[1].axhline(1.0, color="black", ls="--", lw=0.8, label="||w|| = 1")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("||w||")
    axes[1].set_title("Norma del vector de pesos")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    _save(fig, out_dir, "convergence_weights.png")


def experiment_error_to_reference(X, pc1_ref, out_dir, lr, epochs, decay, n_seeds=5):
    """MSD entre scores de Oja (w normalizado) y scores de sklearn, por época."""
    hists = _seed_histories(X, pc1_ref, lr, epochs, decay, n_seeds)
    dirs = hists / np.linalg.norm(hists, axis=2, keepdims=True)
    scores_ref = X @ pc1_ref
    scores = dirs @ X.T                              # (n_seeds, epochs+1, n_samples)
    curves = np.mean((scores - scores_ref) ** 2, axis=2)
    t = np.arange(curves.shape[1])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _band(ax, t, curves.mean(0), curves.std(0), "steelblue", alpha=0.25, lw=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("Época")
    ax.set_ylabel(r"MSD$(X\cdot w_t,\,X\cdot w_{ref})$ (log)")
    ax.set_title("MSD de scores Oja vs. sklearn")
    ax.grid(alpha=0.3, which="both")
    _save(fig, out_dir, "error_to_reference.png")


def experiment_lr_decay_heatmap(X, pc1_ref, out_dir, epochs, decay,
                                lrs=None, decays=None, n_seeds=5):
    """Var(X·w_oja) / λ₁(PCA). 1.0 = recupera el óptimo de PCA; < 1
    alineación parcial; NaN ('div') cuando Oja diverge (w → nan/inf).
    """
    if lrs is None:
        lrs = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    if decays is None:
        decays = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    decays = sorted(set(decays) | {decay})

    lambda1 = float(np.var(X @ pc1_ref, ddof=1))     # varianza máxima en 1 dirección
    scores_ref = X @ pc1_ref

    rows = []
    for lr in lrs:
        for d in decays:
            ratios, mses = [], []
            for s in range(n_seeds):
                w = OjaNeuron(learning_rate=lr, decay=d, epochs=epochs, seed=s).fit(X)
                scores = X @ w
                if not np.all(np.isfinite(scores)):
                    ratios.append(float("nan"))
                    mses.append(float("nan"))
                    continue
                ratios.append(float(np.var(scores, ddof=1)) / lambda1)
                scores_aligned = X @ align_sign(w, pc1_ref)
                mses.append(float(np.mean((scores_aligned - scores_ref) ** 2)))
            arr, mse_arr = np.array(ratios), np.array(mses)
            rows.append({
                "lr": lr, "decay": d,
                "ratio_mean": _nan_stat(np.nanmean, arr),
                "mse_mean": _nan_stat(np.nanmean, mse_arr),
            })
    df = pd.DataFrame(rows)

    pivot = df.pivot(index="decay", columns="lr", values="ratio_mean").sort_index().sort_index(axis=1)
    pivot_mse = df.pivot(index="decay", columns="lr", values="mse_mean").sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower",
                   cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:g}" for v in pivot.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:g}" for v in pivot.index])
    ax.set_xlabel("Learning rate inicial (η₀)")
    ax.set_ylabel("Decay")
    ax.set_title(r"$\mathrm{Var}/\lambda_1$ (color) y MSE vs. sklearn")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v, m = pivot.values[i, j], pivot_mse.values[i, j]
            if np.isnan(v):
                ax.text(j, i, "div", ha="center", va="center", fontsize=7, color="red")
            else:
                mse_str = "div" if np.isnan(m) else f"${_sci_latex(m)}$"
                ax.text(j, i, f"{v:.4f}\nMSE={mse_str}", ha="center", va="center",
                        fontsize=6.5, color="white" if v < 0.6 else "black")
    fig.colorbar(im, ax=ax, label=r"$\mathrm{Var}/\lambda_1$")
    _save(fig, out_dir, "lr_decay_heatmap.png")


def experiment_loadings(features, w_oja, w_ref, out_dir, w_oja_std=None):
    x = np.arange(len(features))
    width = 0.4
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars_oja = ax.bar(x - width / 2, w_oja, width, label="Oja", color="steelblue",
                      yerr=w_oja_std, capsize=3, error_kw={"lw": 0.8, "ecolor": "black"})
    bars_ref = ax.bar(x + width / 2, w_ref, width, label="PCA", color="orange")
    for bars, vals in [(bars_oja, w_oja), (bars_ref, w_ref)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.01 if v >= 0 else -0.01),
                    f"{v:.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=30, ha="right")
    ax.set_ylabel("Loading sobre PC1")
    ax.set_title("Loadings de la PC1: Oja vs. PCA")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    _save(fig, out_dir, "loadings.png")


def experiment_scores_scatter(countries, X, pc1_ref, scores_ref, out_dir,
                              lr, epochs, decay, n_seeds=5):
    """Scatter de scores Oja vs. sklearn con la recta y=x (media de n_seeds)."""
    seeds_scores, mses, r2s = [], [], []
    ss_tot = float(np.sum((scores_ref - scores_ref.mean()) ** 2))
    for s in range(n_seeds):
        w = align_sign(OjaNeuron(learning_rate=lr, epochs=epochs, decay=decay, seed=s).fit(X), pc1_ref)
        scores_s = X @ w
        seeds_scores.append(scores_s)
        mses.append(float(np.mean((scores_s - scores_ref) ** 2)))
        ss_res = float(np.sum((scores_s - scores_ref) ** 2))
        r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else float("nan"))
    seeds_scores = np.array(seeds_scores)
    scores_mean, scores_std = seeds_scores.mean(0), seeds_scores.std(0)
    mse, r2 = float(np.mean(mses)), float(np.mean(r2s))

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.errorbar(scores_ref, scores_mean, yerr=scores_std,
                fmt="o", color="navy", ms=5, capsize=2, lw=0.8, zorder=3)
    for rank, i in enumerate(np.argsort(scores_ref)):
        dx, ha = (6, "left") if rank % 2 == 0 else (-6, "right")
        ax.annotate(countries[i], (scores_ref[i], scores_mean[i]), fontsize=7,
                    xytext=(dx, 4), textcoords="offset points", ha=ha,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6),
                    zorder=4)
    lo = min(scores_ref.min(), scores_mean.min())
    hi = max(scores_ref.max(), scores_mean.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y = x")
    ax.set_xlabel("Score PC1 sklearn")
    ax.set_ylabel("Score PC1 Oja")
    ax.set_title(rf"Scores Oja vs. sklearn (MSE$={_sci_latex(mse)}$, R²={r2:.4f})")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "scores_scatter.png")


def experiment_unnormalized_init(X, pc1_ref, out_dir, lr, epochs, decay, n_seeds=5):
    rng = np.random.default_rng(123)
    base_dir = rng.uniform(-1, 1, size=X.shape[1])
    base_dir /= np.linalg.norm(base_dir)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    cmap = plt.get_cmap("tab10")
    scores_ref = X @ pc1_ref
    for k, n0 in enumerate([0.1, 0.5, 1.0, 2.0, 5.0]):
        hists = _seed_histories(X, pc1_ref, lr, epochs, decay, n_seeds, init_w=base_dir * n0)
        norms = np.linalg.norm(hists, axis=2)        # ||w|| es invariante al signo
        dirs = hists / np.linalg.norm(hists, axis=2, keepdims=True)
        scores = dirs @ X.T                          # (n_seeds, epochs+1, n_samples)
        errs = np.mean((scores - scores_ref) ** 2, axis=2)
        color = cmap(k % 10)
        t = np.arange(norms.shape[1])
        _band(axes[0], t, norms.mean(0), norms.std(0), color, label=f"||w₀|| = {n0}", lw=1.4)
        _band(axes[1], t, errs.mean(0), errs.std(0), color, label=f"||w₀|| = {n0}", lw=1.4)

    axes[0].axhline(1.0, color="black", ls="--", lw=0.8, label="||w|| = 1")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("||w||")
    axes[0].set_title("La regla de Oja normaliza ||w|| → 1")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")

    axes[1].set_yscale("log")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel(r"MSD$(X\cdot w_t,\,X\cdot w_{ref})$ (log)")
    axes[1].set_title("MSD de scores respecto de la PC1")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    _save(fig, out_dir, "unnormalized_init.png")


def experiment_init_strategy(X, pc1_ref, out_dir, lr, epochs, decay, n_seeds=5):
    strategies = ["normalized", "unnormalized"]
    err_curves = {s: [] for s in strategies}
    norm_curves = {s: [] for s in strategies}
    scores_ref = X @ pc1_ref

    for seed in range(n_seeds):
        for s in strategies:
            oja = OjaNeuron(learning_rate=lr, decay=decay, epochs=epochs, seed=seed,
                            normalize_initial_weights=(s == "normalized"),
                            record_history=True)
            oja.fit(X)
            hist = np.array([align_sign(w, pc1_ref) for w in oja.history_])
            dirs = hist / np.linalg.norm(hist, axis=1, keepdims=True)
            scores = dirs @ X.T                      # (epochs+1, n_samples)
            err_curves[s].append(np.mean((scores - scores_ref) ** 2, axis=1))
            norm_curves[s].append(np.linalg.norm(oja.history_, axis=1))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"normalized": "steelblue", "unnormalized": "crimson"}
    t = np.arange(epochs + 1)
    for s in strategies:
        err = np.array(err_curves[s])
        _band(ax, t, err.mean(0), err.std(0), colors[s], label=s, lw=1.8)
        nrm = np.array(norm_curves[s])
        _band(ax2, t, nrm.mean(0), nrm.std(0), colors[s], clamp=False, label=s, lw=1.8)

    ax.set_yscale("log")
    ax.set_xlim(0, 30)
    ax.set_xlabel("Época")
    ax.set_ylabel(r"MSD$(X\cdot w_t,\,X\cdot w_{ref})$ (log)")
    ax.set_title("MSD de scores a la PC1")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    ax2.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7)
    ax2.set_xlim(0, 30)
    ax2.set_xlabel("Época")
    ax2.set_ylabel(r"$\|w_t\|$")
    ax2.set_title("Norma del vector de pesos")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("Inicialización: normalizada vs. sin normalizar (primeras 30 épocas)")
    _save(fig, out_dir, "init_strategy.png")


def experiment_shuffle(X, pc1_ref, out_dir, lr, epochs, decay, n_seeds=5):
    strategies = [True, False]
    err_curves = {s: [] for s in strategies}
    scores_ref = X @ pc1_ref

    for seed in range(n_seeds):
        for s in strategies:
            oja = OjaNeuron(learning_rate=lr, decay=decay, epochs=epochs,
                            seed=seed, shuffle=s, record_history=True)
            oja.fit(X)
            hist = np.array([align_sign(w, pc1_ref) for w in oja.history_])
            dirs = hist / np.linalg.norm(hist, axis=1, keepdims=True)
            scores = dirs @ X.T
            err_curves[s].append(np.mean((scores - scores_ref) ** 2, axis=1))

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {True: "steelblue", False: "crimson"}
    labels = {True: "shuffle=True", False: "shuffle=False"}
    t = np.arange(epochs + 1)
    for s in strategies:
        curves = np.array(err_curves[s])
        _band(ax, t, curves.mean(0), curves.std(0), colors[s], label=labels[s], lw=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("Época")
    ax.set_ylabel(r"MSD$(X\cdot w_t,\,X\cdot w_{ref})$ (log)")
    ax.set_title("Efecto del shuffle de muestras")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    _save(fig, out_dir, "shuffle.png")


def _plot_variance_spectrum(ratios, cum, out_dir, name, title, ratios_std=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    idx = np.arange(1, len(ratios) + 1)
    colors = ["steelblue"] + ["lightgrey"] * (len(ratios) - 1)
    bars = ax.bar(idx, ratios * 100, yerr=None if ratios_std is None else ratios_std * 100,
                  color=colors, edgecolor="black", lw=0.6,
                  capsize=3, error_kw={"lw": 0.8})
    for b, v in zip(bars, ratios * 100):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.4f}%",
                ha="center", va="bottom", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(idx, cum * 100, color="crimson", marker="o", lw=1.4, label="Acumulado")
    for xi, yi in zip(idx, cum * 100):
        ax2.annotate(f"{yi:.2f}%", (xi, yi), textcoords="offset points",
                     xytext=(6, -10), color="crimson", fontsize=8)
    ax2.set_ylabel("Varianza acumulada (%)", color="crimson")
    ax2.tick_params(axis="y", colors="crimson")
    ax2.set_ylim(0, 110)
    ax.set_ylim(0, max(ratios * 100) * (1.25 if ratios_std is not None else 1.18))
    ax.set_xlabel("Componente principal")
    ax.set_ylabel("Varianza explicada (%)")
    ax.set_title(title)
    ax.set_xticks(idx)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, out_dir, name)


def experiment_explained_variance_spectrum(X, out_dir):
    ratios = PCA().fit(X).explained_variance_ratio_
    cum = np.cumsum(ratios)
    _plot_variance_spectrum(ratios, cum, out_dir, "explained_variance_spectrum.png",
                            "Componentes principales y varianza explicada")


def experiment_explained_variance_spectrum_sanger(X, out_dir, lr, epochs, decay, n_seeds=5):
    # Sanger sin renormalización final necesita LR chico y decay agresivo para
    # que las componentes 2..k no se escapen por perturbación de deflación.
    total_var = float(np.sum(np.var(X, axis=0)))
    ratios_runs = []
    for s in range(n_seeds):
        W = SangerNetwork(n_components=7, learning_rate=lr,
                          epochs=epochs, decay=decay, seed=s).fit(X)
        ratios_runs.append(np.var(X @ W.T, axis=0) / total_var)
    ratios_arr = np.array(ratios_runs)
    ratios, ratios_std = ratios_arr.mean(0), ratios_arr.std(0)
    cum = np.cumsum(ratios)
    _plot_variance_spectrum(ratios, cum, out_dir, "explained_variance_spectrum_sanger.png",
                            "Componentes principales y varianza explicada (Sanger)",
                            ratios_std=ratios_std)


def experiment_variance_captured(X, w_oja, pca, out_dir, w_seeds=None):
    total_var = float(np.var(X, axis=0, ddof=1).sum())
    var_ref = float(pca.explained_variance_[0])
    ratio_ref = var_ref / total_var

    if w_seeds is not None:
        ratios_oja = np.array([float(np.var(X @ w_s, ddof=1)) / total_var for w_s in w_seeds])
        ratio_oja, ratio_oja_std = float(ratios_oja.mean()), float(ratios_oja.std())
        var_oja = ratio_oja * total_var
    else:
        var_oja = float(np.var(X @ w_oja, ddof=1))
        ratio_oja, ratio_oja_std = var_oja / total_var, 0.0

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    yerr = [ratio_oja_std * 100, np.nan] if w_seeds is not None else None
    bars = ax.bar(["Oja", "PCA"], [ratio_oja * 100, ratio_ref * 100],
                  color=["steelblue", "orange"], edgecolor="black", lw=0.6,
                  yerr=yerr, capsize=4, error_kw={"lw": 0.8, "ecolor": "black"})
    for b, v in zip(bars, [ratio_oja * 100, ratio_ref * 100]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.4f}%", ha="center", fontsize=10)
    ax.set_ylabel("Varianza capturada por PC1 (%)")
    ax.set_title("Varianza Capturada: Oja vs PCA")
    ax.set_ylim(0, max(ratio_oja, ratio_ref) * 100 * 1.15)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, out_dir, "variance_captured.png")


def experiment_pc1_ranking_biplot(X, countries, features, w_oja, out_dir, scores_std=None):
    scores = X @ w_oja
    xmax = float(np.max(np.abs(scores)))
    pad = xmax * 0.01
    loads = w_oja * 0.9 * xmax          

    order_f = np.argsort(w_oja)
    feats_sorted = [features[i] for i in order_f]
    loads_sorted, w_sorted = loads[order_f], w_oja[order_f]

    order_c = np.argsort(scores)
    countries_sorted = [countries[i] for i in order_c]
    scores_sorted = scores[order_c]
    std_sorted = scores_std[order_c] if scores_std is not None else None

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 11),
        gridspec_kw={"height_ratios": [len(features), len(countries)]})

    yf = np.arange(len(feats_sorted))
    for j, (load, wv) in enumerate(zip(loads_sorted, w_sorted)):
        color = "seagreen" if load >= 0 else "indianred"
        ax_top.annotate("", xy=(load, j), xytext=(0, j),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6))
        ax_top.text(load + (pad if load >= 0 else -pad), j, f"{wv:.3f}",
                    va="center", ha="left" if load >= 0 else "right", fontsize=7)
    ax_top.set_yticks(yf)
    ax_top.set_yticklabels(feats_sorted, fontsize=8)
    ax_top.set_ylim(-0.6, len(feats_sorted) - 0.4)
    ax_top.axvline(0, color="black", lw=0.6)
    ax_top.set_title("Cargas de variables sobre la PC1 (Oja)")
    ax_top.grid(alpha=0.3, axis="x")

    yc = np.arange(len(countries_sorted))
    colors = ["seagreen" if v >= 0 else "indianred" for v in scores_sorted]
    ax_bot.barh(yc, scores_sorted, color=colors, xerr=std_sorted, capsize=2,
                error_kw={"lw": 0.6, "ecolor": "black"})
    for y, v in zip(yc, scores_sorted):
        ax_bot.text(v + (pad if v >= 0 else -pad), y, f"{v:.2f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=7)
    ax_bot.set_yticks(yc)
    ax_bot.set_yticklabels(countries_sorted, fontsize=8)
    ax_bot.set_xlim(-xmax * 1.18, xmax * 1.18)
    ax_bot.axvline(0, color="black", lw=0.6)
    ax_bot.set_xlabel("Score de PC1 (Oja)  /  loading escalado")
    ax_bot.set_title("Ranking de países sobre la PC1")
    ax_bot.grid(alpha=0.3, axis="x")
    _save(fig, out_dir, "pc1_ranking_biplot.png")


def experiment_hebb_vs_oja(X, out_dir, lr, epochs, n_seeds=5):
    n = X.shape[1]
    norms_hebb_runs, norms_oja_runs = [], []
    for s in range(n_seeds):
        rng = np.random.default_rng(s)
        w0 = rng.uniform(0, 1, size=n)
        w0 /= np.linalg.norm(w0)

        w = w0.copy()
        norms_hebb = [np.linalg.norm(w)]
        rng_h = np.random.default_rng(s)
        for _ in range(epochs):
            for i in rng_h.permutation(X.shape[0]):
                x = X[i]
                w = w + lr * float(np.dot(x, w)) * x
            norms_hebb.append(float(np.linalg.norm(w)))
        norms_hebb_runs.append(norms_hebb)

        oja = OjaNeuron(learning_rate=lr, epochs=epochs, decay=0.0, seed=s, record_history=True)
        oja.fit(X, init_w=w0.copy())
        norms_oja_runs.append(np.linalg.norm(oja.history_, axis=1))

    hebb_arr, oja_arr = np.array(norms_hebb_runs), np.array(norms_oja_runs)
    hebb_mean, hebb_std = hebb_arr.mean(0), hebb_arr.std(0)
    oja_mean, oja_std = oja_arr.mean(0), oja_arr.std(0)

    t = np.arange(epochs + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    _band(ax, t, hebb_mean, hebb_std, "crimson", lw=1.8, marker="o", markersize=3,
          label=rf"Hebb ($\|w\|_{{final}} = {_sci_latex(hebb_mean[-1])}$)")
    _band(ax, t, oja_mean, oja_std, "steelblue", lw=1.8, marker="s", markersize=3,
          label=rf"Oja ($\|w\|_{{final}} = {oja_mean[-1]:.4f}$)")
    ax.axhline(1.0, color="black", ls="--", lw=0.8, label="||w|| = 1")
    ax.set_yscale("log")
    ax.set_xlabel("Época")
    ax.set_ylabel("||w|| (log)")
    ax.set_title("Hebb vs Oja")
    ax.legend(loc="best")
    ax.grid(alpha=0.3, which="both")
    _save(fig, out_dir, "hebb_vs_oja.png")

# Main
def main():
    LR, EPOCHS, DECAY, N_SEEDS = 0.1, 1000, 0.01, 5
    HEBB_LR, HEBB_EPOCHS = 0.001, 200

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    countries, X, features = load_europe(DATA_PATH)

    pca = PCA(n_components=1).fit(X)
    pc1_ref = pca.components_[0]

    w_seeds = np.array([
        align_sign(OjaNeuron(learning_rate=LR, epochs=EPOCHS, decay=DECAY, seed=s).fit(X), pc1_ref)
        for s in range(N_SEEDS)
    ])
    w_oja_mean = w_seeds.mean(axis=0)
    w_oja = w_oja_mean / np.linalg.norm(w_oja_mean)
    w_oja_std = w_seeds.std(axis=0)

    scores_per_seed = X @ w_seeds.T
    scores_oja_std = scores_per_seed.std(axis=1)
    scores_ref = pca.transform(X).ravel()

    print("Varianza explicada (PCA / Sanger)...")
    experiment_explained_variance_spectrum(X, OUT_DIR)
    experiment_explained_variance_spectrum_sanger(X, OUT_DIR, lr=LR, epochs=EPOCHS, decay=DECAY)

    print("Convergencia e inicialización...")
    experiment_convergence(X, features, pc1_ref, OUT_DIR, lr=LR, epochs=EPOCHS, decay=DECAY)
    experiment_error_to_reference(X, pc1_ref, OUT_DIR, lr=LR, epochs=EPOCHS, decay=DECAY)
    experiment_unnormalized_init(X, pc1_ref, OUT_DIR, lr=LR, epochs=EPOCHS, decay=DECAY)
    experiment_init_strategy(X, pc1_ref, OUT_DIR, lr=LR, epochs=EPOCHS, decay=DECAY)
    experiment_hebb_vs_oja(X, OUT_DIR, lr=HEBB_LR, epochs=HEBB_EPOCHS)

    print("Sensibilidad (lr/decay) y shuffle...")
    experiment_lr_decay_heatmap(X, pc1_ref, OUT_DIR, epochs=EPOCHS, decay=DECAY)
    experiment_shuffle(X, pc1_ref, OUT_DIR, lr=LR, epochs=EPOCHS, decay=DECAY)

    print("Loadings y ranking de países...")
    experiment_loadings(features, w_oja, pc1_ref, OUT_DIR, w_oja_std=w_oja_std)
    experiment_pc1_ranking_biplot(X, countries, features, w_oja, OUT_DIR, scores_std=scores_oja_std)

    print("Scores y varianza capturada...")
    experiment_scores_scatter(countries, X, pc1_ref, scores_ref, OUT_DIR,
                              lr=LR, epochs=EPOCHS, decay=DECAY)
    experiment_variance_captured(X, w_oja, pca, OUT_DIR, w_seeds=w_seeds)

    print(f"Listo. Resultados en: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

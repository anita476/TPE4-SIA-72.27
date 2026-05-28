"""
Spatial-gradient stability analysis.

Mide, para cada variable y para el eje de bienestar, si forman un gradiente
espacial sobre la grilla del SOM, y si ese gradiente es estable entre semillas.

Metodo: para cada mapa entrenado y cada variable (o el promedio de bienestar),
se ajusta una regresion lineal  valor ~ fila + columna  sobre las k*k neuronas.
  - R^2  -> cuan ordenado/lineal es el gradiente (1 = eje perfecto, 0 = sin patron)
  - (b_fila, b_col) -> direccion del gradiente; su angulo da la orientacion
El R^2 es invariante a rotacion/reflexion (mide si HAY gradiente, no hacia donde).
La orientacion (angulo) SI cambia con la semilla: es lo que queremos exponer.

Config fija: k=4, eta_0=0.1, R0=4, n_iter=8000, init=samples, haykin.
Varia solo: seed.

Salidas: results/plots/spatial_gradient/  (results.txt + figura)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.kohonen.kohonen import Kohonen
from utils.preprocessing import load_europe

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
K        = 4
ETA_0    = 0.1
RADIUS_0 = 4.0
N_ITER   = 8000
INIT     = "samples"
SEEDS    = list(range(1, 11))          # 20 seeds

plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.facecolor": "none",
    "axes.facecolor":   "none",
    "savefig.transparent": True,
})

WELLBEING = ["GDP", "Life.expect", "Pop.growth"]   # variables del eje de bienestar

OUT_DIR = REPO_ROOT / "results" / "plots" / "spatial_gradient"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Per-seed outputs
# -----------------------------------------------------------------------------
def _save_hitmap(som, seed, countries, X, out_path):
    acts   = som.activations_per_neuron(X)
    labels = {}
    for idx, x in enumerate(X):
        i, j = som._winner(x)
        labels.setdefault((int(i), int(j)), []).append(countries[idx])
    dead_n  = int((acts == 0).sum())
    act_max = acts.max() or 1
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(acts, cmap="plasma", origin="upper", vmin=0, vmax=act_max)
    for i in range(K):
        for j in range(K):
            norm  = acts[i, j] / act_max
            color = "#1a1a1a" if norm >= 0.6 else "white"
            if (i, j) in labels:
                ax.text(j, i - 0.18, str(acts[i, j]), ha="center", va="center",
                        fontsize=8, fontweight="bold", color=color)
                ax.text(j, i + 0.22,
                        "\n".join(n[:3].upper() for n in labels[(i, j)]),
                        ha="center", va="center",
                        fontsize=6.5, color=color, linespacing=1.2)
            else:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=9, color="white", alpha=0.45)
    for v in np.arange(-0.5, K, 1):
        ax.axhline(v, color="white", linewidth=0.8)
        ax.axvline(v, color="white", linewidth=0.8)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(range(K), fontsize=8)
    ax.set_yticklabels(range(K), fontsize=8)
    ax.set_title(f"Hit map — seed {seed}  (dead={dead_n})",
                 fontsize=10, fontweight="bold", pad=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close()


def _save_component_planes(som, seed, variables, out_path_arrows, out_path_no_arrows):
    """Component planes in plasma colormap, per-variable scale. Two versions:
    one with gradient arrows, one without."""
    n_vars = len(variables)
    ncols  = 4
    nrows  = (n_vars + ncols - 1) // ncols
    jj, ii = np.meshgrid(np.arange(K), np.arange(K))

    for show_arrows, out_path in [(True, out_path_arrows), (False, out_path_no_arrows)]:
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 3.2, nrows * 3.0),
                                 layout="constrained")
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for v_idx, (ax, var) in enumerate(zip(axes_flat, variables)):
            plane  = som.weights[:, :, v_idx]
            v_min, v_max = plane.min(), plane.max()
            im = ax.imshow(plane, cmap="plasma", origin="upper",
                           vmin=v_min, vmax=v_max, aspect="equal")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if show_arrows:
                gr, gc = np.gradient(plane)
                scale  = max(np.sqrt(gr**2 + gc**2).max(), 1e-9)
                ax.quiver(jj, ii, gc / scale, -gr / scale,
                          color="white", alpha=0.8, scale=6,
                          width=0.012, headwidth=3, headlength=4)

            for v in np.arange(-0.5, K, 1):
                ax.axhline(v, color="white", linewidth=0.6)
                ax.axvline(v, color="white", linewidth=0.6)
            ax.set_xticks(range(K)); ax.set_yticks(range(K))
            ax.set_xticklabels(range(K), fontsize=6)
            ax.set_yticklabels(range(K), fontsize=6)
            r2, *_ = grid_gradient(plane, K)
            ax.set_title(f"{var}  R²={r2:.2f}", fontsize=9, fontweight="bold", pad=5)

        for ax in axes_flat[n_vars:]:
            ax.set_visible(False)

        suffix = "with_arrows" if show_arrows else "no_arrows"
        fig.suptitle(f"Component planes ({suffix}) — seed {seed}  "
                     f"(k={K}, η₀={ETA_0}, R₀={RADIUS_0}, {INIT}, haykin)",
                     fontsize=11, fontweight="bold")
        plt.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
        plt.close()


def _save_seed_txt(som, seed, countries, X, variables, r2s, out_path):
    acts   = som.activations_per_neuron(X)
    u      = som.u_matrix()
    dead_n = int((acts == 0).sum())
    cells  = {}
    for idx, x in enumerate(X):
        i, j = som._winner(x)
        cells.setdefault((int(i), int(j)), []).append(countries[idx])
    sing_n   = sum(1 for v in cells.values() if len(v) == 1)
    qe_final = som.qe_history[-1][1] if som.qe_history else float("nan")

    # wellbeing axis gradient
    wb_idx = [variables.index(v) for v in WELLBEING if v in variables]
    bien   = som.weights[:, :, wb_idx].mean(axis=2)
    wb_r2, wb_ang, wb_mag, *_ = grid_gradient(bien, K)

    lines = [
        f"seed {seed}  —  k={K}, eta_0={ETA_0}, R0={RADIUS_0}, "
        f"n_iter={N_ITER}, init={INIT}, haykin",
        "=" * 60,
        f"Final QE:           {qe_final:.4f}",
        f"Dead neurons:       {dead_n} / {K*K}",
        f"Singletons:         {sing_n} / {K*K - dead_n} active",
        f"Max U-dist:         {u.max():.4f}",
        f"Max cluster:        {acts.max()}",
        f"Wellbeing axis R²:  {wb_r2:.4f}  angle={wb_ang:.1f}°  mag={wb_mag:.4f}",
        "",
        "R² spatial gradient per variable:",
    ]
    for var, r2 in sorted(zip(variables, r2s), key=lambda x: -x[1]):
        lines.append(f"  {var:<16} R² = {r2:.4f}")

    lines += ["", "Assignments:"]
    for cell in sorted(cells):
        tag     = "  [singleton]" if len(cells[cell]) == 1 else ""
        members = ", ".join(sorted(cells[cell]))
        lines.append(f"  neuron {cell}: {members}{tag}")

    lines += ["", "U-matrix:"]
    for i in range(K):
        lines.append("  " + "  ".join(f"{u[i,j]:.3f}" for j in range(K)))

    lines += ["", "Component planes (weight per neuron):"]
    for v_idx, var in enumerate(variables):
        lines.append(f"\n  {var}:")
        for i in range(K):
            lines.append("    " + "  ".join(
                f"{som.weights[i,j,v_idx]:>+7.3f}" for j in range(K)))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Gradiente espacial: regresion valor ~ fila + columna
# -----------------------------------------------------------------------------
def grid_gradient(plane, k):
    """plane: (k,k) -> (R2, angulo_deg, magnitud, b_fila, b_col)."""
    ii, jj = np.indices((k, k))
    A = np.column_stack([ii.ravel(), jj.ravel(), np.ones(k * k)])
    y = plane.ravel()
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    b_fila, b_col, _ = coef
    yhat = A @ coef
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    ang = np.degrees(np.arctan2(b_fila, b_col))
    mag = np.hypot(b_fila, b_col)
    return r2, ang, mag, b_fila, b_col


def run():
    countries, X, variables = load_europe()
    n_vars = len(variables)
    wb_idx = [variables.index(v) for v in WELLBEING]
    print(f"{len(countries)} countries, vars: {variables}")
    print(f"Bienestar = {WELLBEING} -> idx {wb_idx}\n")

    # acumuladores: R2 por variable y para bienestar, angulos para orientacion
    r2_var   = {v: [] for v in variables}
    r2_wb    = []
    ang_wb   = []
    orient_wb = []

    hm_dir       = OUT_DIR / "hitmaps"
    cp_arrows    = OUT_DIR / "component_planes_arrows"
    cp_no_arrows = OUT_DIR / "component_planes_no_arrows"
    sd_dir       = OUT_DIR / "per_seed_txt"
    for d in (hm_dir, cp_arrows, cp_no_arrows, sd_dir):
        d.mkdir(exist_ok=True)

    for s in SEEDS:
        som = Kohonen(
            k=K, input_dim=n_vars, eta_0=ETA_0, radius_0=RADIUS_0,
            weight_init=INIT, similarity="euclidean", sample_replace=True,
            eta_schedule="haykin", radius_schedule="haykin",
            n_iter=N_ITER, seed=s,
        )
        som.fit(X, n_iter=N_ITER)
        W = som.weights                       # (k,k,n_vars)

        # por variable
        seed_r2s = []
        for vi, v in enumerate(variables):
            r2, *_ = grid_gradient(W[:, :, vi], K)
            r2_var[v].append(r2)
            seed_r2s.append(r2)

        # eje de bienestar (promedio de planos estandarizados de las 3)
        bien = W[:, :, wb_idx].mean(axis=2)
        r2, ang, mag, bf, bc = grid_gradient(bien, K)
        r2_wb.append(r2)
        ang_wb.append(ang)
        orient_wb.append("horizontal" if abs(bc) > abs(bf) else "vertical")
        print(f"  seed {s:3d}  bienestar R2={r2:.3f}  ang={ang:6.1f}  {orient_wb[-1]}")

        _save_hitmap(som, s, countries, X, hm_dir / f"hitmap_seed{s}.png")
        _save_component_planes(som, s, list(variables),
                               cp_arrows    / f"planes_seed{s}.png",
                               cp_no_arrows / f"planes_seed{s}.png")
        _save_seed_txt(som, s, countries, X, list(variables), seed_r2s,
                       sd_dir / f"seed{s}.txt")

    r2_wb = np.array(r2_wb)
    ang_wb = np.array(ang_wb)
    ang_mod = ang_wb % 180   # el eje tiene simetria: +180 = mismo eje

    # ---- tabla por variable ----
    print("\n" + "=" * 60)
    print(f"R2 del gradiente espacial por variable  (media +- std, {len(SEEDS)} seeds)")
    print("=" * 60)
    order = sorted(variables, key=lambda v: -np.mean(r2_var[v]))
    for v in order:
        a = np.array(r2_var[v])
        print(f"  {v:<14} R2 = {a.mean():.3f} +- {a.std():.3f}   "
              f"(rango {a.min():.3f}-{a.max():.3f})")

    print("\n" + "=" * 60)
    print("EJE DE BIENESTAR (promedio GDP, Life.expect, Pop.growth)")
    print("=" * 60)
    print(f"  R2 (hay gradiente):  {r2_wb.mean():.3f} +- {r2_wb.std():.3f}  "
          f"(rango {r2_wb.min():.3f}-{r2_wb.max():.3f})")
    nh = orient_wb.count("horizontal"); nv = orient_wb.count("vertical")
    print(f"  Orientacion:         {nh} horizontal / {nv} vertical")
    print(f"  Angulo (mod 180):    media={ang_mod.mean():.1f}  std={ang_mod.std():.1f}  "
          f"(rango {ang_mod.min():.1f}-{ang_mod.max():.1f})")
    print("\n  => El eje EXISTE en toda corrida (R2 alto y estable);")
    print("     su ORIENTACION es arbitraria (reparto h/v, angulo disperso).")

    # ---- guardar txt ----
    lines = [
        "Spatial-gradient stability",
        f"k={K} eta_0={ETA_0} R0={RADIUS_0} n_iter={N_ITER} init={INIT} haykin",
        f"Seeds: {SEEDS}", "=" * 60,
        "R2 por variable (media +- std):",
    ]
    for v in order:
        a = np.array(r2_var[v])
        lines.append(f"  {v:<14} {a.mean():.3f} +- {a.std():.3f}")
    lines += [
        "", "Eje de bienestar:",
        f"  R2 = {r2_wb.mean():.3f} +- {r2_wb.std():.3f}",
        f"  Orientacion: {nh} horizontal / {nv} vertical",
        f"  Angulo (mod 180): {ang_mod.mean():.1f} +- {ang_mod.std():.1f}",
    ]
    (OUT_DIR / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved: {OUT_DIR / 'results.txt'}")

    _plot(variables, r2_var, r2_wb, ang_wb, order)


# -----------------------------------------------------------------------------
# Figura: R2 por variable (barras) + orientacion del eje (rosa de angulos)
# -----------------------------------------------------------------------------
def _plot(variables, r2_var, r2_wb, ang_wb, order):
    fig = plt.figure(figsize=(14, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1], wspace=0.3)

    # --- izquierda: R2 por variable, barras con error ---
    ax = fig.add_subplot(gs[0])
    means = [np.mean(r2_var[v]) for v in order]
    stds  = [np.std(r2_var[v]) for v in order]
    wb_m, wb_s = r2_wb.mean(), r2_wb.std()
    labels = order + ["EJE BIENESTAR"]
    vals   = means + [wb_m]
    errs   = stds + [wb_s]
    colors = ["#5588CC"] * len(order) + ["#CC4433"]
    ypos = np.arange(len(labels))[::-1]
    ax.barh(ypos, vals, xerr=errs, color=colors, alpha=0.85,
            error_kw=dict(ecolor="#333", lw=1, capsize=3))
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 1); ax.set_xlabel("R²  del gradiente espacial  (1 = eje perfecto)")
    ax.set_title("¿Cada variable forma un gradiente sobre el mapa?",
                 fontsize=11, fontweight="bold")
    ax.axvline(0.5, color="#999", ls=":", lw=1)
    for y, v, e in zip(ypos, vals, errs):
        ax.text(v + e + 0.02, y, f"{v:.2f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.2)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

    # --- derecha: orientacion del eje de bienestar (cada seed una flecha) ---
    ax2 = fig.add_subplot(gs[1], projection="polar")
    for ang in ang_wb:
        a = np.radians(ang % 180)
        ax2.plot([a, a], [0, 1], color="#CC4433", alpha=0.5, lw=2)
        ax2.plot([a + np.pi, a + np.pi], [0, 1], color="#CC4433", alpha=0.5, lw=2)
    ax2.set_yticks([]); ax2.set_thetagrids([0, 45, 90, 135],
                                            ["horizontal", "", "vertical", ""])
    ax2.set_title("Orientación del eje de bienestar\n(cada línea = una semilla)",
                  fontsize=11, fontweight="bold", pad=18)

    fig.suptitle(
        f"Gradiente espacial — k={K}, η₀={ETA_0}, R₀={RADIUS_0}, "
        f"n_iter={N_ITER}, init={INIT}, haykin, {len(SEEDS)} seeds",
        fontsize=12, fontweight="bold", y=1.02)
    out = OUT_DIR / "spatial_gradient_stability.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    run()
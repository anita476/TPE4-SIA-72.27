"""
Hopfield noise-tolerance analysis per stored pattern.

Outcomes (same as plot_hopfield / hopfield_analysis):
  exact     — recovered the original pattern
  inverse   — inverse of the original
  wrong     — converged to a different stored pattern (wrong pattern)
  spurious  — not any stored pattern

Cliff = first noise level where exact-recovery fraction drops below 50%.
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from utils.letters import load_patterns, add_noise, classify_recovery
from hopfield.HopfieldNetwork import HopfieldNetwork
from hopfield.ContinuousHopfieldNetwork import ContinuousHopfieldNetwork

BG_COLOR    = '#fff5ec'
C_EXACT     = '#2ecc71'
C_INVERSE   = '#e67e22'
C_WRONG = '#d35400'
C_SPURIOUS  = '#e74c3c'
C_THRESH    = '#76373b'

CLIFF_THRESHOLD = 0.5  # exact recovery becomes less likely than not


def run_trial(net, query_flat, stored, noise_pct, seed, max_iter=20, mode="sync",
              net_type="classic"):
    noisy = add_noise(
        query_flat.reshape(5, 5),
        noise_pct,
        seed=seed,
    ).flatten()

    if net_type == "modern":
        # Modern network: normalise query, run softmax updates, binarise result
        xi = noisy.astype(float)
        xi = xi / (np.linalg.norm(xi) + 1e-12)
        for _ in range(max_iter):
            xi_new = net._update(xi)
            if np.linalg.norm(xi_new - xi) < 1e-6:
                break
            xi = xi_new
        result = np.where(xi > 0, 1.0, -1.0)
    else:
        result = net.predict(noisy, mode=mode, max_iterations=max_iter,
                             verbose=False, seed=seed)

    return classify_recovery(result, query_flat, stored)


def _build_network(net_type, n, pattern_matrix, beta):
    """Instantiate and initialise the requested network type."""
    if net_type == "modern":
        net = ContinuousHopfieldNetwork(d=n, beta=beta)
        net.store_patterns(pattern_matrix)
    else:
        net = HopfieldNetwork(n=n)
        net.initialize_weights(pattern_matrix)
    return net


def analyse(patterns_file, noise_steps=20, trials=30, max_iter=20, seed=42,
            mode="sync", net_type="classic", beta=4.0):
    stored = load_patterns(patterns_file)
    if not stored:
        raise ValueError(f"No patterns found in {patterns_file}")

    pattern_matrix = np.array([p.flatten() for p in stored.values()])
    n = pattern_matrix.shape[1]
    net = _build_network(net_type, n, pattern_matrix, beta)

    noise_levels = np.linspace(0.0, 1.0, noise_steps + 1)

    results = {}
    for name, pat in stored.items():
        flat = pat.flatten()
        results[name] = []
        for ni, noise_pct in enumerate(noise_levels):
            counts = {"exact": 0, "inverse": 0, "wrong": 0, "spurious": 0}
            for t in range(trials):
                outcome = run_trial(
                    net, flat, stored, noise_pct,
                    seed=seed + ni * 1000 + t,
                    max_iter=max_iter, mode=mode,
                    net_type=net_type,
                )
                counts[outcome] += 1
            results[name].append(counts)

    return results, noise_levels


def cliff_level(pat_data, noise_levels):
    """First noise level where exact-recovery fraction drops below 50%."""
    for ni, counts in enumerate(pat_data):
        total = sum(counts.values()) or 1
        if counts["exact"] / total < CLIFF_THRESHOLD:
            return noise_levels[ni]
    return noise_levels[-1]


def plot_stacked_areas(results, noise_levels, trials, out_path):
    names = list(results.keys())
    n_patterns = len(names)

    fig, axes = plt.subplots(
        n_patterns, 1,
        figsize=(12, 3 * n_patterns),
        facecolor=BG_COLOR,
        sharex=True,
    )
    fig.set_dpi(150)
    if n_patterns == 1:
        axes = [axes]

    cliffs = {}
    for pi, (ax, name) in enumerate(zip(axes, names)):
        ax.set_facecolor(BG_COLOR)

        exact_pct     = np.array([c["exact"]      / trials * 100 for c in results[name]])
        inverse_pct   = np.array([c["inverse"]    / trials * 100 for c in results[name]])
        wrong_pct = np.array([c["wrong"] / trials * 100 for c in results[name]])
        spurious_pct  = np.array([c["spurious"]   / trials * 100 for c in results[name]])
        x = noise_levels * 100

        ax.stackplot(
            x,
            exact_pct, inverse_pct, wrong_pct, spurious_pct,
            labels=["Exact", "Inverse", "Wrong pattern", "Spurious"],
            colors=[C_EXACT, C_INVERSE, C_WRONG, C_SPURIOUS],
            alpha=0.88,
        )

        cliff = cliff_level(results[name], noise_levels)
        cliffs[name] = cliff
        cliff_idx = np.searchsorted(noise_levels, cliff)
        ax.axvline(cliff * 100, color=C_THRESH, lw=1.5, ls='--', alpha=0.85)
        ax.text(cliff * 100 + 1, 92, f"cliff {cliff*100:.0f}%",
                color=C_THRESH, fontsize=8, va='top')

        # label exact % at three key points: start, cliff, end
        key_indices = sorted({0, cliff_idx, len(noise_levels) - 1})
        for ki in key_indices:
            cp = exact_pct[ki]
            if cp < 8:
                continue
            ax.text(x[ki], cp / 2, f"{cp:.0f}%",
                    ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold',
                    clip_on=True)
        ax.axhline(CLIFF_THRESHOLD * 100, color='#888', lw=0.8, ls=':', alpha=0.6)

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_title(f"Pattern [{name}]", fontsize=10, fontweight='bold',
                     loc='left', pad=4, color='#444')
        ax.set_ylabel("% trials", fontsize=8, color='#666')
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.tick_params(labelsize=8, colors='#666')
        for spine in ax.spines.values():
            spine.set_edgecolor('#ddd')

        if pi == 0:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.6,
                      facecolor=BG_COLOR, edgecolor='#ddd',
                      ncol=4, handlelength=1.2)

    axes[-1].set_xlabel("Noise level", fontsize=8, color='#666')
    axes[-1].xaxis.set_major_formatter(mticker.PercentFormatter())

    fig.suptitle(
        f"Hopfield noise-tolerance  ·  {trials} trials/point  ·  "
        f"cliff = first noise level where exact recovery < 50%",
        fontsize=10, color='#444',
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_heatmap(results, noise_levels, trials, out_path):
    names = list(results.keys())
    n_patterns = len(names)
    n_noise = len(noise_levels)

    matrix = np.array([
        [results[name][ni]["exact"] / trials for ni in range(n_noise)]
        for name in names
    ])

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "tol", [C_SPURIOUS, '#FAC775', C_EXACT], N=256
    )

    cell_h = 1.2
    fig_h = max(3, n_patterns * cell_h + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_h), facecolor=BG_COLOR)
    fig.set_dpi(150)
    ax.set_facecolor(BG_COLOR)

    # pcolormesh draws individual cells with edges
    x_edges = np.arange(n_noise + 1) - 0.5
    y_edges = np.arange(n_patterns + 1) - 0.5
    mesh = ax.pcolormesh(
        x_edges, y_edges, matrix,
        cmap=cmap, vmin=0, vmax=1,
        edgecolors='black', linewidth=0.6,
    )

    # y-axis: pattern names
    ax.set_yticks(range(n_patterns))
    ax.set_yticklabels(names, fontsize=10, fontweight='bold', color='#333')
    ax.invert_yaxis()

    # x-axis: noise percentages, only label every ~10%
    tick_every = max(1, n_noise // 10)
    xtick_pos    = list(range(0, n_noise, tick_every))
    xtick_labels = [f"{noise_levels[i]*100:.0f}%" for i in xtick_pos]
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, fontsize=8, color='#666')
    ax.set_xlabel("Noise level", fontsize=9, color='#666')

    # cliff lines
    cliffs = {name: cliff_level(results[name], noise_levels) for name in names}
    for pi, name in enumerate(names):
        cliff_idx = np.searchsorted(noise_levels, cliffs[name])
        ax.axvline(cliff_idx - 0.5, color=C_THRESH, lw=1.5, ls='--', alpha=0.8)

    cbar = fig.colorbar(mesh, ax=ax, orientation='vertical',
                        fraction=0.015, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("exact recovery rate", fontsize=8)

    ax.set_title(
        f"Exact-recovery rate per pattern  ·  {trials} trials/point",
        fontsize=10, color='#444', pad=8,
    )
    for spine in ax.spines.values():
        spine.set_edgecolor('#ccc')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_retrieval_curves(results, noise_levels, trials, out_path):
    names = list(results.keys())
    x = noise_levels * 100
    exact_matrix = np.array([
        [c["exact"] / trials * 100 for c in results[name]]
        for name in names
    ])                                          # shape: (n_patterns, n_noise)

    mean_curve = exact_matrix.mean(axis=0)
    std_curve  = exact_matrix.std(axis=0)

    cliffs    = {name: cliff_level(results[name], noise_levels) for name in names}
    avg_cliff = float(np.mean(list(cliffs.values())))
    cmap   = plt.get_cmap("tab20")
    colors = [cmap(i / max(len(names) - 1, 1)) for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG_COLOR)
    fig.set_dpi(150)
    ax.set_facecolor(BG_COLOR)

    ax.fill_between(
        x,
        np.clip(mean_curve - std_curve, 0, 100),
        np.clip(mean_curve + std_curve, 0, 100),
        color="#aaaaaa", alpha=0.25, label="+/-1 std (across patterns)",
        zorder=1,
    )
    for i, name in enumerate(names):
        ax.plot(
            x, exact_matrix[i],
            color=colors[i], lw=1.4, alpha=0.82,
            label=name, zorder=3,
        )
        # thin faint cliff tick per pattern
        ax.axvline(cliffs[name] * 100, color=colors[i],
                   lw=0.8, ls=":", alpha=0.45, zorder=2)
    ax.plot(
        x, mean_curve,
        color="#333333", lw=2.2, ls="--", alpha=0.9,
        label="mean", zorder=4,
    )
    ax.axvline(avg_cliff * 100, color=C_THRESH, lw=2.0, ls="--", alpha=0.9, zorder=5)
    ax.text(
        avg_cliff * 100 + 0.8, 94,
        f"avg cliff  {avg_cliff*100:.0f}%",
        color=C_THRESH, fontsize=9, fontweight="bold", va="top", zorder=6,
    )
    ax.axhline(CLIFF_THRESHOLD * 100, color="#888", lw=0.9, ls=":", alpha=0.6, zorder=2)
    ax.text(0.5, CLIFF_THRESHOLD * 100 + 1.5,
            "50% threshold", color="#888", fontsize=7, va="bottom")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 103)
    ax.set_xlabel("Noise level", fontsize=9, color="#666")
    ax.set_ylabel("Exact recovery (%)", fontsize=9, color="#666")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.tick_params(labelsize=8, colors="#666")
    for spine in ax.spines.values():
        spine.set_edgecolor("#ddd")

    ax.set_title(
        f"Exact-recovery curves per pattern  ·  {trials} trials/point  ·  "
        f"avg cliff = {avg_cliff*100:.0f}%",
        fontsize=10, color="#444", pad=8,
    )
    ax.legend(
        loc="upper right", fontsize=7.5, framealpha=0.7,
        facecolor=BG_COLOR, edgecolor="#ddd",
        ncol=2, handlelength=1.4,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Hopfield noise-tolerance analysis")
    parser.add_argument("patterns_file")
    parser.add_argument("--noise-steps", type=int, default=20)
    parser.add_argument("--trials",      type=int, default=30)
    parser.add_argument("--max-iter",    type=int, default=20)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--out-dir",     type=str, default=".",
                        help="Directory to write PNGs into (default: current dir)")
    parser.add_argument(
        "--mode", choices=["sync", "async"], default="sync",
        help="Hopfield update mode for the classic network (default: sync)",
    )
    parser.add_argument(
        "--type", dest="net_type", choices=["classic", "modern"], default="classic",
        help="Network type: 'classic' (binary Hopfield) or 'modern' "
             "(continuous, Ramsauer et al. 2021). Default: classic",
    )
    parser.add_argument(
        "--beta", type=float, default=4.0,
        help="Inverse temperature β for the modern network (ignored for classic). "
             "Default: 4.0",
    )
    args = parser.parse_args()

    if args.net_type == "modern" and args.mode != "sync":
        print("Note: --mode is ignored for the modern network (always uses softmax update).")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running analysis  [type={args.net_type}"
          + (f", beta={args.beta}" if args.net_type == "modern" else f", mode={args.mode}")
          + "] ...")
    results, noise_levels = analyse(
        args.patterns_file,
        noise_steps=args.noise_steps,
        trials=args.trials,
        max_iter=args.max_iter,
        seed=args.seed,
        mode=args.mode,
        net_type=args.net_type,
        beta=args.beta,
    )

    plot_stacked_areas(results, noise_levels, args.trials,
                       out_dir / "stacked_areas.png")
    plot_heatmap(results, noise_levels, args.trials,
                 out_dir / "heatmap.png")
    plot_retrieval_curves(results, noise_levels, args.trials,
                          out_dir / "retrieval_curves.png")


if __name__ == "__main__":
    main()
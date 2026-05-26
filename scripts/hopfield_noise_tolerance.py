"""
    Each individual trial
    is always a binary outcome, exact recovery or not.
    The cliff is therefore
    defined as the noise level where the fraction of correct trials first drops
    below 50%, ie:
     correct recovery becomes less likely than not
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.letters import load_patterns, add_noise, best_match
from hopfield.HopfieldNetwork import HopfieldNetwork

BG_COLOR = '#fff5ec'
C_CORRECT  = '#1D9E75'
C_INVERSE  = '#378ADD'
C_SPURIOUS = '#D85A30'
C_THRESH   = '#76373b'

CLIFF_THRESHOLD = 0.5  # crossover: correct recovery becomes less likely than not


def run_trial(net, query_flat, noise_pct, seed):
    noisy = add_noise(
        query_flat.reshape(5, 5),
        noise_pct,
        seed=seed
    ).flatten()
    result = net.predict(noisy, max_iterations=20, verbose=False)
    if np.array_equal(result, query_flat):
        return "correct"
    if np.array_equal(result, -query_flat):
        return "inverse"
    return "spurious"

def analyse(patterns_file, noise_steps=20, trials=30, max_iter=20, seed=42):
    stored = load_patterns(patterns_file)
    if not stored:
        raise ValueError(f"No patterns found in {patterns_file}")

    pattern_matrix = np.array([p.flatten() for p in stored.values()])
    net = HopfieldNetwork(n=25)
    net.initialize_weights(pattern_matrix)

    noise_levels = np.linspace(0.0, 1.0, noise_steps + 1)

    results = {}
    for name, pat in stored.items():
        flat = pat.flatten()
        results[name] = []
        for ni, noise_pct in enumerate(noise_levels):
            counts = {"correct": 0, "inverse": 0, "spurious": 0}
            for t in range(trials):
                outcome = run_trial(net, flat, noise_pct,
                                    seed=seed + ni * 1000 + t)
                counts[outcome] += 1
            results[name].append(counts)

    return results, noise_levels


def cliff_level(pat_data, noise_levels):
    """First noise level where correct-trial fraction drops below 50%."""
    for ni, counts in enumerate(pat_data):
        total = sum(counts.values()) or 1
        if counts["correct"] / total < CLIFF_THRESHOLD:
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

        correct_pct  = np.array([c["correct"]  / trials * 100 for c in results[name]])
        inverse_pct  = np.array([c["inverse"]  / trials * 100 for c in results[name]])
        spurious_pct = np.array([c["spurious"] / trials * 100 for c in results[name]])
        x = noise_levels * 100

        ax.stackplot(
            x,
            correct_pct, inverse_pct, spurious_pct,
            labels=["Correct", "Inverse", "Spurious"],
            colors=[C_CORRECT, C_INVERSE, C_SPURIOUS],
            alpha=0.88,
        )

        cliff = cliff_level(results[name], noise_levels)
        cliffs[name] = cliff
        cliff_idx = np.searchsorted(noise_levels, cliff)
        ax.axvline(cliff * 100, color=C_THRESH, lw=1.5, ls='--', alpha=0.85)
        ax.text(cliff * 100 + 1, 92, f"cliff {cliff*100:.0f}%",
                color=C_THRESH, fontsize=8, va='top')

        # label correct % at three key points: start, cliff, end
        key_indices = sorted({0, cliff_idx, len(noise_levels) - 1})
        for ki in key_indices:
            cp = correct_pct[ki]
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
                      ncol=3, handlelength=1.2)

    axes[-1].set_xlabel("Noise level", fontsize=8, color='#666')
    axes[-1].xaxis.set_major_formatter(mticker.PercentFormatter())

    fig.suptitle(
        f"Hopfield noise-tolerance  ·  {trials} trials/point  ·  "
        f"cliff = first noise level where correct recovery < 50%",
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
        [results[name][ni]["correct"] / trials for ni in range(n_noise)]
        for name in names
    ])

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "tol", [C_SPURIOUS, '#FAC775', C_CORRECT], N=256
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
    cbar.set_label("correct recovery rate", fontsize=8)

    ax.set_title(
        f"Correct-recovery rate per pattern  ·  {trials} trials/point",
        fontsize=10, color='#444', pad=8,
    )
    for spine in ax.spines.values():
        spine.set_edgecolor('#ccc')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
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
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running analysis...")
    results, noise_levels = analyse(
        args.patterns_file,
        noise_steps=args.noise_steps,
        trials=args.trials,
        max_iter=args.max_iter,
        seed=args.seed,
    )

    plot_stacked_areas(results, noise_levels, args.trials,
                       out_dir / "stacked_areas.png")
    plot_heatmap(results, noise_levels, args.trials,
                 out_dir / "heatmap.png")


if __name__ == "__main__":
    main()
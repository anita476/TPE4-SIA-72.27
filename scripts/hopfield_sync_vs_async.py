"""
Hopfield network — sync vs async comparison plots for TPE4.

Plots generated
───────────────
1. energy_overlay     — energy trajectories of both modes on the same axes,
                        averaged over seeds, for several noise levels
2. steps_distribution — distribution of iterations to convergence per mode
                        (boxplots side by side, per noise level)
3. trajectory_compare — same noisy input recovered in both modes,
                        shown frame by frame as pixel grids
4. wallclock_compare  — wall-clock time to convergence per mode, per noise
                        level (boxplots side by side)
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib.pyplot as plt

from utils.letters import load_patterns, classify_recovery
from hopfield.HopfieldNetwork import HopfieldNetwork

# Reuse styling + helpers from the single-mode analysis script
from hopfield_analysis import (
    apply_style, save_fig, draw_pattern, build_net, run_steps,
    flip_noise, STYLE, FIG_DPI, FIG_SIZE, OUTCOME_COLOR, OUTCOME_LS,
)

C_SYNC  = "#4a90d9"
C_ASYNC = "#d35400"


# ── Plot 1: energy convergence overlay ───────────────────────────────────────

def plot_energy_overlay(stored: dict, query_name: str, out: Path,
                         n_seeds: int = 15,
                         noise_levels: list = (0.1, 0.3, 0.5, 0.7)):
    """
    Overlay mean energy trajectory for sync and async at several noise levels.
    Sync is solid, async is dashed; colour encodes noise level.
    """
    apply_style()
    net   = build_net(stored)
    query = stored[query_name]

    cmap   = plt.cm.plasma
    colors = [cmap(i / max(1, len(noise_levels) - 1)) for i in range(len(noise_levels))]

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI,
                            facecolor=STYLE["figure_bg"])
    ax.set_facecolor(STYLE["axes_bg"])

    for noise, color in zip(noise_levels, colors):
        for mode, ls, marker in [("sync", "-", "o"), ("async", "--", "s")]:
            all_curves = []
            for seed in range(n_seeds):
                base_seed = seed * 100 + int(noise * 10)
                noisy  = flip_noise(query.flatten(), noise,
                                    np.random.default_rng(base_seed))
                states = run_steps(net, noisy, mode=mode, seed=base_seed)
                all_curves.append([net.energy(s) for s in states])

            max_len = max(len(c) for c in all_curves)
            padded  = [c + [c[-1]] * (max_len - len(c)) for c in all_curves]
            arr     = np.array(padded)
            mean    = arr.mean(axis=0)
            xs      = list(range(max_len))

            label = f"{noise:.0%}  {mode}"
            ax.plot(xs, mean, linestyle=ls, color=color,
                    linewidth=1.8, marker=marker, markersize=4,
                    label=label, alpha=0.9)

    ax.set_xlabel("Iteration (sync step / async sweep)",
                   color=STYLE["text_axis"])
    ax.set_ylabel("Energy  E(s)", color=STYLE["text_axis"])
    ax.grid(color=STYLE["grid"], linestyle="--", linewidth=0.7)
    ax.legend(title="Noise / Mode", fontsize=8, title_fontsize=8,
               framealpha=0.85, loc="lower right", ncol=len(noise_levels))
    plt.tight_layout()
    save_fig(fig, out)


# ── Plot 2: distribution of iterations to convergence ────────────────────────

def plot_steps_distribution(stored: dict, query_name: str, out: Path,
                             n_seeds: int = 80,
                             noise_levels: list = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)):
    """
    Boxplot of #iterations to convergence per (mode, noise level).
    """
    apply_style()
    net   = build_net(stored)
    query = stored[query_name]

    steps = {"sync": [], "async": []}
    for noise in noise_levels:
        for mode in ("sync", "async"):
            counts = []
            for seed in range(n_seeds):
                base_seed = seed * 100 + int(noise * 10)
                noisy  = flip_noise(query.flatten(), noise,
                                    np.random.default_rng(base_seed))
                states = run_steps(net, noisy, mode=mode, seed=base_seed)
                counts.append(len(states) - 1)
            steps[mode].append(counts)

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI,
                            facecolor=STYLE["figure_bg"])
    ax.set_facecolor(STYLE["axes_bg"])

    n_levels = len(noise_levels)
    positions = np.arange(n_levels) * 3
    width = 0.9

    bp_sync = ax.boxplot(
        steps["sync"], positions=positions - width / 1.6, widths=width,
        patch_artist=True, manage_ticks=False,
        boxprops=dict(facecolor=C_SYNC, alpha=0.75, edgecolor=C_SYNC),
        medianprops=dict(color="white", linewidth=1.6),
        whiskerprops=dict(color=C_SYNC), capprops=dict(color=C_SYNC),
        flierprops=dict(marker="o", markerfacecolor=C_SYNC,
                        markeredgecolor=C_SYNC, markersize=3, alpha=0.5),
    )
    bp_async = ax.boxplot(
        steps["async"], positions=positions + width / 1.6, widths=width,
        patch_artist=True, manage_ticks=False,
        boxprops=dict(facecolor=C_ASYNC, alpha=0.75, edgecolor=C_ASYNC),
        medianprops=dict(color="white", linewidth=1.6),
        whiskerprops=dict(color=C_ASYNC), capprops=dict(color=C_ASYNC),
        flierprops=dict(marker="o", markerfacecolor=C_ASYNC,
                        markeredgecolor=C_ASYNC, markersize=3, alpha=0.5),
    )

    # mean markers
    for i, noise in enumerate(noise_levels):
        ax.plot(positions[i] - width / 1.6, np.mean(steps["sync"][i]),
                marker="D", color="white", markeredgecolor=C_SYNC,
                markersize=6, zorder=5)
        ax.plot(positions[i] + width / 1.6, np.mean(steps["async"][i]),
                marker="D", color="white", markeredgecolor=C_ASYNC,
                markersize=6, zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{n:.0%}" for n in noise_levels])
    ax.set_xlabel("Noise level", color=STYLE["text_axis"])
    ax.set_ylabel("Iterations to convergence",
                   color=STYLE["text_axis"])
    ax.grid(color=STYLE["grid"], linestyle="--", linewidth=0.7, axis="y")
    ax.legend(
        [bp_sync["boxes"][0], bp_async["boxes"][0]],
        ["sync", "async"],
        loc="upper left", fontsize=9, framealpha=0.85,
    )
    plt.tight_layout()
    save_fig(fig, out)


# ── Plot 3: side-by-side pixel-by-pixel trajectory comparison ────────────────

def plot_trajectory_compare(stored: dict, query_name: str,
                             noise: float, seed: int, out: Path):
    """
    Same noisy input recovered in both modes, shown frame by frame
    as pixel grids on two rows (top: sync, bottom: async).
    """
    apply_style()
    net   = build_net(stored)
    query = stored[query_name]
    rng   = np.random.default_rng(seed)
    noisy = flip_noise(query.flatten(), noise, rng)

    states_sync  = run_steps(net, noisy, mode="sync",  seed=seed)
    states_async = run_steps(net, noisy, mode="async", seed=seed)

    n_sync  = len(states_sync)
    n_async = len(states_async)
    ncols   = 1 + max(n_sync, n_async)   # +1 for the "original" column

    fig, axes = plt.subplots(2, ncols,
                              figsize=(ncols * 1.5, 4.2),
                              dpi=FIG_DPI, facecolor=STYLE["figure_bg"])

    out_sync  = classify_recovery(states_sync[-1],  query, stored)
    out_async = classify_recovery(states_async[-1], query, stored)
    bord_s    = OUTCOME_COLOR[out_sync]
    bord_a    = OUTCOME_COLOR[out_async]

    # Column 0: the original target pattern (for reference)
    draw_pattern(axes[0, 0], query, title="target", title_fontsize=9)
    draw_pattern(axes[1, 0], query, title="target", title_fontsize=9)

    # Rows: trajectories. Column 1 = noisy input (t=0), then steps.
    for row, (states, n_steps, border, mode) in enumerate([
        (states_sync,  n_sync,  bord_s, "sync"),
        (states_async, n_async, bord_a, "async"),
    ]):
        for k in range(ncols - 1):
            ax = axes[row, k + 1]
            if k < n_steps:
                is_final = (k == n_steps - 1)
                bc       = border if is_final else None
                title    = "noisy" if k == 0 else f"t={k}"
                draw_pattern(ax, states[k], title=title,
                              energy=net.energy(states[k]),
                              border_color=bc, title_fontsize=9)
            else:
                ax.axis("off")

        # row label on the far left
        axes[row, 0].set_ylabel(mode, fontsize=11, fontweight="bold",
                                 color=C_SYNC if mode == "sync" else C_ASYNC,
                                 rotation=0, labelpad=28, va="center")

    plt.tight_layout()
    save_fig(fig, out)


# ── Plot 4: wall-clock time to convergence ───────────────────────────────────

def _time_convergence(net: HopfieldNetwork, noisy: np.ndarray,
                       mode: str, seed: int, repeats: int = 50) -> float:
    """
    Wall-clock seconds to converge from *noisy*, averaged over *repeats*
    identical runs (to reduce timer noise on very fast iterations).
    """
    t0 = time.perf_counter()
    for _ in range(repeats):
        net.predict(noisy, mode=mode, seed=seed, verbose=False)
    return (time.perf_counter() - t0) / repeats


def plot_wallclock_compare(stored: dict, query_name: str, out: Path,
                            n_seeds: int = 60,
                            noise_levels: list = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
                            repeats: int = 50):
    """
    Boxplot of wall-clock time to convergence per (mode, noise level).
    Each measurement averages over `repeats` identical runs to reduce noise.
    """
    apply_style()
    net   = build_net(stored)
    query = stored[query_name]

    # warm-up to stabilise JIT / cache before measuring
    warmup_noisy = flip_noise(query.flatten(), 0.2, np.random.default_rng(0))
    for _ in range(20):
        net.predict(warmup_noisy, mode="sync", seed=0, verbose=False)
        net.predict(warmup_noisy, mode="async", seed=0, verbose=False)

    times_us = {"sync": [], "async": []}
    for noise in noise_levels:
        for mode in ("sync", "async"):
            seconds = []
            for seed in range(n_seeds):
                base_seed = seed * 100 + int(noise * 10)
                noisy = flip_noise(query.flatten(), noise,
                                    np.random.default_rng(base_seed))
                t = _time_convergence(net, noisy, mode, base_seed, repeats)
                seconds.append(t * 1e6)   # convert to microseconds
            times_us[mode].append(seconds)

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI,
                            facecolor=STYLE["figure_bg"])
    ax.set_facecolor(STYLE["axes_bg"])

    n_levels  = len(noise_levels)
    positions = np.arange(n_levels) * 3
    width     = 0.9

    bp_sync = ax.boxplot(
        times_us["sync"], positions=positions - width / 1.6, widths=width,
        patch_artist=True, manage_ticks=False,
        boxprops=dict(facecolor=C_SYNC, alpha=0.75, edgecolor=C_SYNC),
        medianprops=dict(color="white", linewidth=1.6),
        whiskerprops=dict(color=C_SYNC), capprops=dict(color=C_SYNC),
        flierprops=dict(marker="o", markerfacecolor=C_SYNC,
                        markeredgecolor=C_SYNC, markersize=3, alpha=0.5),
    )
    bp_async = ax.boxplot(
        times_us["async"], positions=positions + width / 1.6, widths=width,
        patch_artist=True, manage_ticks=False,
        boxprops=dict(facecolor=C_ASYNC, alpha=0.75, edgecolor=C_ASYNC),
        medianprops=dict(color="white", linewidth=1.6),
        whiskerprops=dict(color=C_ASYNC), capprops=dict(color=C_ASYNC),
        flierprops=dict(marker="o", markerfacecolor=C_ASYNC,
                        markeredgecolor=C_ASYNC, markersize=3, alpha=0.5),
    )

    # mean markers
    for i, noise in enumerate(noise_levels):
        ax.plot(positions[i] - width / 1.6, np.mean(times_us["sync"][i]),
                marker="D", color="white", markeredgecolor=C_SYNC,
                markersize=6, zorder=5)
        ax.plot(positions[i] + width / 1.6, np.mean(times_us["async"][i]),
                marker="D", color="white", markeredgecolor=C_ASYNC,
                markersize=6, zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{n:.0%}" for n in noise_levels])
    ax.set_xlabel("Noise level", color=STYLE["text_axis"])
    ax.set_ylabel("Wall-clock time to convergence  (µs)",
                   color=STYLE["text_axis"])
    ax.grid(color=STYLE["grid"], linestyle="--", linewidth=0.7, axis="y")
    ax.legend(
        [bp_sync["boxes"][0], bp_async["boxes"][0]],
        ["sync", "async"],
        loc="upper left", fontsize=9, framealpha=0.85,
    )
    plt.tight_layout()
    save_fig(fig, out)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    apply_style()

    patterns_file = ROOT / "data" / "patterns.txt"
    out_dir       = ROOT / "results" / "hopfield" / "comparison"

    stored     = load_patterns(str(patterns_file))
    query_name = list(stored.keys())[2]   # W, same as hopfield_analysis.py

    print(f"Stored patterns : {list(stored.keys())}")
    print(f"Query           : {query_name}")
    print(f"Output          : {out_dir}\n")

    plot_energy_overlay(stored, query_name,
        out=out_dir / "1_energy_overlay.png")

    plot_steps_distribution(stored, query_name,
        out=out_dir / "2_steps_distribution.png")

    # one "clean" recovery and one "interesting" mid-noise case
    plot_trajectory_compare(stored, query_name, noise=0.20, seed=7,
        out=out_dir / "3_trajectory_low_noise.png")
    plot_trajectory_compare(stored, query_name, noise=0.40, seed=14,
        out=out_dir / "3_trajectory_mid_noise.png")

    plot_wallclock_compare(stored, query_name,
        out=out_dir / "4_wallclock_compare.png")

    print("\nDone.")


if __name__ == "__main__":
    main()

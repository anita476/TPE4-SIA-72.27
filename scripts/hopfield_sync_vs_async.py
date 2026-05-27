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
5. outcomes_paired    — paired stacked bars per noise level, breaking down
                        results by outcome type (exact / inverse / wrong /
                        spurious) for sync and async side by side
6. outcomes_table     — same data as plot 5, rendered as an exact numeric
                        table (per noise × mode × outcome)
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


# ── Plot 5: paired outcome breakdown per noise level ─────────────────────────

def _compute_outcome_fractions(stored: dict, n_trials: int,
                                noise_levels: list) -> dict:
    """
    For each (mode, noise level), run n_trials and return the fraction of
    trials that ended in each outcome category.

    Returns a dict like:
        { "sync":  {"exact": [...], "wrong": [...], "inverse": [...], "spurious": [...]},
          "async": {...} }
    """
    pattern_list = list(stored.values())
    pats         = np.array([p.flatten() for p in pattern_list])
    net          = HopfieldNetwork(n=25)
    net.initialize_weights(pats)

    fracs = {"sync":  {k: [] for k in ["exact", "inverse", "wrong", "spurious"]},
             "async": {k: [] for k in ["exact", "inverse", "wrong", "spurious"]}}

    for noise in noise_levels:
        for mode in ("sync", "async"):
            counts = {"exact": 0, "inverse": 0, "wrong": 0, "spurious": 0}
            rng = np.random.default_rng(0)
            for trial in range(n_trials):
                q     = pattern_list[trial % len(pattern_list)]
                noisy = flip_noise(q.flatten(), noise, rng)
                r     = net.predict(noisy, mode=mode, seed=trial, verbose=False)
                counts[classify_recovery(r, q, stored)] += 1
            for k in counts:
                fracs[mode][k].append(counts[k] / n_trials)
    return fracs


def plot_outcomes_paired(stored: dict, out: Path,
                          n_trials: int = 1000,
                          noise_levels: list = (0.04, 0.12, 0.20, 0.28,
                                                 0.36, 0.44, 0.52, 0.60)):
    """
    For each noise level draw a pair of stacked bars (sync | async)
    showing the fraction of trials that ended in each outcome category.
    """
    apply_style()
    fracs = _compute_outcome_fractions(stored, n_trials, noise_levels)

    fig, ax = plt.subplots(figsize=(FIG_SIZE[0], FIG_SIZE[1] + 0.5),
                            dpi=FIG_DPI, facecolor=STYLE["figure_bg"])
    ax.set_facecolor(STYLE["axes_bg"])

    n_levels  = len(noise_levels)
    positions = np.arange(n_levels) * 3
    width     = 1.05

    outcome_order  = ["exact", "wrong", "inverse", "spurious"]
    outcome_label  = {"exact": "Exact", "wrong": "Wrong pattern",
                       "inverse": "Inverse", "spurious": "Spurious"}
    # local palette: override "inverse" to a distinctly different hue
    # so it doesn't blur with "wrong pattern" in stacked bars
    local_color = {
        "exact":    OUTCOME_COLOR["exact"],
        "wrong":    OUTCOME_COLOR["wrong"],
        "inverse":  "#7b5ea7",   # muted purple
        "spurious": OUTCOME_COLOR["spurious"],
    }

    for offset, mode in [(-width / 1.6, "sync"), (+width / 1.6, "async")]:
        bottoms = np.zeros(n_levels)
        for outcome in outcome_order:
            heights = np.array(fracs[mode][outcome])
            ax.bar(positions + offset, heights, width=width,
                   bottom=bottoms, color=local_color[outcome],
                   edgecolor="white", linewidth=0.6,
                   label=outcome_label[outcome] if mode == "sync" else None,
                   alpha=0.88)
            bottoms += heights

    # group labels under each pair (sync / async)
    for i, noise in enumerate(noise_levels):
        ax.text(positions[i] - width / 1.6, -0.012, "s", ha="center", va="top",
                fontsize=8, color=C_SYNC, fontweight="bold",
                transform=ax.get_xaxis_transform())
        ax.text(positions[i] + width / 1.6, -0.012, "a", ha="center", va="top",
                fontsize=8, color=C_ASYNC, fontweight="bold",
                transform=ax.get_xaxis_transform())

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{n:.0%}" for n in noise_levels])
    ax.tick_params(axis="x", pad=18)
    ax.set_xlabel("Noise level   (s = sync, a = async)",
                   color=STYLE["text_axis"])
    ax.set_ylabel("Fraction of trials", color=STYLE["text_axis"])
    ax.set_ylim(0, 1)
    ax.grid(color=STYLE["grid"], linestyle="--", linewidth=0.7, axis="y")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=9, framealpha=0.9, title="Outcome", title_fontsize=9)
    plt.tight_layout()
    save_fig(fig, out)


# ── Plot 6: outcome breakdown as a numeric table ─────────────────────────────

def plot_outcomes_table(stored: dict, out: Path,
                         n_trials: int = 1000,
                         noise_levels: list = (0.04, 0.12, 0.20, 0.28,
                                                0.36, 0.44, 0.52, 0.60)):
    """
    Same data as plot_outcomes_paired, rendered as a numeric table.
    Rows  = noise levels.
    Cols  = (sync exact / wrong / inv / spur, async exact / wrong / inv / spur).
    Cell colour intensity tracks the value, with green for exact and a
    sequential warm palette for the failure modes.
    """
    apply_style()
    fracs = _compute_outcome_fractions(stored, n_trials, noise_levels)

    outcomes = ["exact", "wrong", "inverse", "spurious"]
    out_lbl  = {"exact": "Exact", "wrong": "Wrong",
                 "inverse": "Inverse", "spurious": "Spurious"}

    # Build (n_rows, n_cols) matrix of percentages
    n_rows = len(noise_levels)
    n_cols = 2 * len(outcomes)
    data   = np.zeros((n_rows, n_cols))
    for j, mode in enumerate(("sync", "async")):
        for k, oc in enumerate(outcomes):
            data[:, j * len(outcomes) + k] = np.array(fracs[mode][oc]) * 100

    fig, ax = plt.subplots(figsize=(11.5, 0.55 * n_rows + 1.6),
                            dpi=FIG_DPI, facecolor=STYLE["figure_bg"])
    ax.set_facecolor(STYLE["axes_bg"])
    ax.set_xlim(0, n_cols + 1)
    ax.set_ylim(0, n_rows + 2)
    ax.invert_yaxis()
    ax.axis("off")

    # Local palette (matches plot_outcomes_paired)
    local_color = {
        "exact":    OUTCOME_COLOR["exact"],
        "wrong":    OUTCOME_COLOR["wrong"],
        "inverse":  "#7b5ea7",
        "spurious": OUTCOME_COLOR["spurious"],
    }

    # ── Header row 0: mode group labels (sync / async)
    sync_color  = C_SYNC
    async_color = C_ASYNC
    ax.add_patch(plt.Rectangle((1, 0), len(outcomes), 1,
                                facecolor=sync_color, alpha=0.18,
                                edgecolor="white"))
    ax.add_patch(plt.Rectangle((1 + len(outcomes), 0), len(outcomes), 1,
                                facecolor=async_color, alpha=0.18,
                                edgecolor="white"))
    ax.text(1 + len(outcomes) / 2, 0.5, "sync",
             ha="center", va="center", fontsize=12, fontweight="bold",
             color=sync_color)
    ax.text(1 + len(outcomes) + len(outcomes) / 2, 0.5, "async",
             ha="center", va="center", fontsize=12, fontweight="bold",
             color=async_color)

    # ── Header row 1: outcome labels
    for j, mode in enumerate(("sync", "async")):
        for k, oc in enumerate(outcomes):
            col = 1 + j * len(outcomes) + k
            ax.add_patch(plt.Rectangle((col, 1), 1, 1,
                                        facecolor=local_color[oc], alpha=0.25,
                                        edgecolor="white"))
            ax.text(col + 0.5, 1.5, out_lbl[oc],
                     ha="center", va="center", fontsize=9, fontweight="bold",
                     color=STYLE["text_title"])

    # ── Left column header
    ax.text(0.5, 1.5, "Noise", ha="center", va="center",
             fontsize=10, fontweight="bold", color=STYLE["text_title"])

    # ── Data rows
    for i, noise in enumerate(noise_levels):
        row_y = 2 + i
        # noise label
        ax.add_patch(plt.Rectangle((0, row_y), 1, 1,
                                    facecolor="#e8dcd0", alpha=0.45,
                                    edgecolor="white"))
        ax.text(0.5, row_y + 0.5, f"{noise:.0%}",
                 ha="center", va="center", fontsize=10, fontweight="bold",
                 color=STYLE["text_title"])

        # cells: tint by mode-group, with bold text for exact > 50%
        for j, mode in enumerate(("sync", "async")):
            for k, oc in enumerate(outcomes):
                col = 1 + j * len(outcomes) + k
                val = data[i, j * len(outcomes) + k]
                # cell tint: per-outcome colour, intensity from value
                intensity = min(0.55, 0.05 + val / 100 * 0.55)
                ax.add_patch(plt.Rectangle(
                    (col, row_y), 1, 1,
                    facecolor=local_color[oc], alpha=intensity,
                    edgecolor="white",
                ))
                weight = "bold" if (oc == "exact" and val >= 50) else "normal"
                ax.text(col + 0.5, row_y + 0.5,
                         f"{val:.1f}" if val >= 0.05 else "—",
                         ha="center", va="center", fontsize=10,
                         color=STYLE["text_title"], fontweight=weight)

    # Footer note: total trials
    ax.text(n_cols / 2 + 0.5, n_rows + 2.5,
             f"values in %  ·  {n_trials} trials per cell",
             ha="center", va="top", fontsize=8,
             color=STYLE["stats_text"], style="italic")

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

    plot_outcomes_paired(stored,
        out=out_dir / "5_outcomes_paired.png")

    plot_outcomes_table(stored,
        out=out_dir / "6_outcomes_table.png")

    print("\nDone.")


if __name__ == "__main__":
    main()

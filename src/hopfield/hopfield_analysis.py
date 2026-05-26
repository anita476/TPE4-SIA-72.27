"""
Hopfield network — complete analysis plots for TPE4.

Plots generated
───────────────
1. stored_patterns      — the 4 stored letter patterns
2. recovery_grid        — all 4 patterns × (original / noisy / recovered)   [req. a]
3. recovery_steps       — step-by-step evolution at 3 noise levels          [req. a]
4. spurious_state       — local minimum reached from very noisy input        [req. b]
5. energy_convergence   — energy vs iteration, averaged over seeds
6. noise_robustness     — stacked outcome fractions vs noise (multi-seed avg)
7. overlap_matrix       — normalised pattern dot-product (orthogonality)
8. basin_by_pattern     — per-pattern recovery rate vs noise (mean ± std)
9. capacity_experiment  — recovery rate vs number of stored patterns
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from sklearn.decomposition import PCA

from utils.letters import load_patterns, load_letters, best_match
from HopfieldNetwork import HopfieldNetwork

# ── Style ─────────────────────────────────────────────────────────────────────

FIG_DPI         = 144
FIG_SIZE        = (11.0, 6.2)
SAVE_PAD_INCHES = 0.2

STYLE = {
    "figure_bg":  "#fff5ec",
    "axes_bg":    "#fff5ec",
    "text_title": "#343434",
    "text_axis":  "#343434",
    "grid":       "#e8dcd0",
    "grid_minor": "#d4c8bc",
    "stats_text": "#555555",
}

PLOT_RC = {
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.spines.top":   True,
    "axes.spines.right": True,
    "axes.edgecolor":    "#343434",
    "axes.labelcolor":   "#343434",
    "axes.linewidth":    0.8,
    "xtick.color":       "#343434",
    "ytick.color":       "#343434",
}

C_EXACT    = "#2ecc71"
C_INVERSE  = "#e67e22"
C_SPURIOUS = "#e74c3c"
C_BLUE     = "#4a90d9"


# ── Shared utilities ──────────────────────────────────────────────────────────

def apply_style():
    plt.rcParams.update(PLOT_RC)


def save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight",
                pad_inches=SAVE_PAD_INCHES, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  done  {path.name}")


def draw_pattern(ax, pattern: np.ndarray, title: str = "",
                 energy: float = None, border_color: str = None,
                 title_fontsize: int = 9, steps: int = None):
    """Draw a +-1 pattern (flat or 5x5) as a pixel grid on *ax*."""
    grid = np.array(pattern).flatten().reshape(5, 5)
    ax.set_xlim(0, 5);  ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.set_xticks([]);  ax.set_yticks([])
    ax.set_facecolor(STYLE["axes_bg"])

    for r in range(5):
        for c in range(5):
            face = "#343434" if grid[r, c] == 1 else STYLE["figure_bg"]
            ax.add_patch(plt.Rectangle(
                (c, 4 - r), 1, 1,
                facecolor=face, edgecolor="#aaaaaa", linewidth=0.5,
            ))

    for spine in ax.spines.values():
        if border_color:
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(2.5)
        else:
            spine.set_visible(False)

    label = title
    if energy is not None:
        label += f"\nE={energy:.2f}"
    if steps is not None:
        label += f"\n{steps} step{'s' if steps != 1 else ''}"
    if label.strip():
        ax.set_title(label, fontsize=title_fontsize,
                     color=STYLE["text_title"], pad=3)


def build_net(stored: dict) -> HopfieldNetwork:
    pattern_matrix = np.array([p.flatten() for p in stored.values()])
    net = HopfieldNetwork(n=25)
    net.initialize_weights(pattern_matrix)
    return net


def classify(result: np.ndarray, stored: dict) -> str:
    _, sim, is_inv = best_match(result, stored)
    if sim == 100.0 and not is_inv:   return "exact"
    if sim == 100.0 and is_inv:       return "inverse"
    return "spurious"


def run_steps(net: HopfieldNetwork, initial: np.ndarray,
              mode: str = "sync", max_iter: int = 20,
              seed: int = None) -> list:
    """
    Return [s(0), s(1), s(2), ...] capturing the state after each step
    (synchronous step or async sweep) until convergence.
    """
    states = [initial.copy().astype(float)]
    s = states[0].copy()

    if mode == "sync":
        s_pprev = None
        for _ in range(max_iter):
            s_prev = s.copy()
            s = np.where(net.W @ s > 0, 1.0, -1.0)
            states.append(s.copy())
            if np.array_equal(s, s_prev):
                break
            if s_pprev is not None and np.array_equal(s, s_pprev):
                break
            s_pprev = s_prev
        return states

    if mode == "async":
        rng = np.random.default_rng(seed)
        for _ in range(max_iter):
            order   = rng.permutation(net.n)
            changes = 0
            for i in order:
                h_i   = float(net.W[i] @ s)
                new_i = 1.0 if h_i > 0 else -1.0
                if new_i != s[i]:
                    s[i] = new_i
                    changes += 1
            states.append(s.copy())
            if changes == 0:
                break
        return states

    raise ValueError(f"Unknown mode '{mode}'.")


def mode_tag(mode: str) -> str:
    """Small badge used in plot titles."""
    return f"[mode: {mode}]"


def flip_noise(flat: np.ndarray, noise: float,
               rng: np.random.Generator) -> np.ndarray:
    """Return copy of *flat* with *noise* fraction of pixels flipped."""
    out = flat.copy()
    n_flip = int(len(out) * noise)
    if n_flip > 0:
        out[rng.choice(len(out), size=n_flip, replace=False)] *= -1
    return out


OUTCOME_COLOR = {"exact": C_EXACT, "inverse": C_INVERSE, "spurious": C_SPURIOUS}
OUTCOME_LS    = {"exact": "-",     "inverse": "--",       "spurious": ":"}


# ── Plot 1: stored patterns ───────────────────────────────────────────────────

def plot_stored_patterns(stored: dict, out: Path):
    apply_style()
    n = len(stored)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.2, 2.8),
                              dpi=FIG_DPI, facecolor=STYLE["figure_bg"])
    axes = np.atleast_1d(axes)
    for ax, (name, pat) in zip(axes, stored.items()):
        draw_pattern(ax, pat, title=name, title_fontsize=11)
    fig.suptitle("Stored Patterns", fontsize=14, fontweight="bold",
                 color=STYLE["text_title"], y=1.06)
    plt.tight_layout()
    save_fig(fig, out)


# ── Plot 2: recovery grid — all patterns × (original / noisy / recovered) ────

def plot_recovery_grid(stored: dict, noise: float, seeds: list, out: Path,
                        mode: str = "sync"):
    """
    Grid with one row per stored pattern.
    Columns: original  |  noisy  |  recovered
    """
    apply_style()
    net = build_net(stored)
    names = list(stored.keys())
    n = len(names)

    fig, axes = plt.subplots(n, 3, figsize=(6.5, n * 1.9 + 0.6),
                              dpi=FIG_DPI, facecolor=STYLE["figure_bg"])
    axes = np.atleast_2d(axes)

    col_headers = [f"Original", f"Noisy  ({noise:.0%})", "Recovered"]
    for j, header in enumerate(col_headers):
        axes[0, j].set_title(header, fontsize=10, fontweight="bold",
                              color=STYLE["text_title"], pad=6)

    for i, (name, pat) in enumerate(stored.items()):
        rng    = np.random.default_rng(seeds[i])
        noisy  = flip_noise(pat.flatten(), noise, rng)
        states = run_steps(net, noisy, mode=mode, seed=seeds[i])
        result = states[-1]
        n_steps = len(states) - 1
        outcome = classify(result, stored)
        border  = OUTCOME_COLOR[outcome]

        draw_pattern(axes[i, 0], pat,    title="")
        draw_pattern(axes[i, 1], noisy,  title="", energy=net.energy(noisy))
        draw_pattern(axes[i, 2], result, title="", energy=net.energy(result),
                     border_color=border, steps=n_steps)

        # row label
        axes[i, 0].text(-0.35, 2.5, name, fontsize=12, fontweight="bold",
                         color=STYLE["text_title"], va="center",
                         transform=axes[i, 0].transData)

    # legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_EXACT,    label="Exact match"),
        Patch(facecolor=C_INVERSE,  label="Inverse attractor"),
        Patch(facecolor=C_SPURIOUS, label="Spurious state"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=8, framealpha=0.85,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Recovery from noisy inputs  (noise = {noise:.0%})   {mode_tag(mode)}",
                 fontsize=13, fontweight="bold",
                 color=STYLE["text_title"], y=1.02)
    plt.tight_layout()
    save_fig(fig, out)


# ── Plot 3: step-by-step recovery at 3 noise levels ──────────────────────────

def plot_recovery_steps(stored: dict, query_name: str,
                         noise_levels: list, seeds: list, out: Path,
                         mode: str = "sync"):
    apply_style()
    net   = build_net(stored)
    query = stored[query_name]

    # collect trajectories
    rows = []
    for noise, seed in zip(noise_levels, seeds):
        rng    = np.random.default_rng(seed)
        noisy  = flip_noise(query.flatten(), noise, rng)
        states = run_steps(net, noisy, mode=mode, seed=seed)
        rows.append((noise, states, noisy))

    # fixed column count: original + noisy + up to MAX_EXTRA steps
    MAX_EXTRA = max(4, max(len(s) - 1 for _, s, _ in rows))
    ncols = 2 + MAX_EXTRA  # original | noisy | t=1 … t=MAX_EXTRA

    nrows = len(rows)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 1.75, nrows * 2.0 + 0.5),
                              dpi=FIG_DPI, facecolor=STYLE["figure_bg"])
    axes = np.atleast_2d(axes)

    # column headers (top row only)
    col_labels = ["original", "noisy"] + [f"t={k}" for k in range(1, MAX_EXTRA + 1)]
    for j, lbl in enumerate(col_labels):
        axes[0, j].set_title(lbl, fontsize=9, fontweight="bold",
                              color=STYLE["text_title"], pad=4)

    for i, (noise, states, noisy) in enumerate(rows):
        outcome = classify(states[-1], stored)
        border  = OUTCOME_COLOR[outcome]

        # col 0: original
        draw_pattern(axes[i, 0], query, title="")
        # col 1: noisy
        draw_pattern(axes[i, 1], noisy, title="", energy=net.energy(noisy))
        # cols 2 …: steps (pad with final if trajectory shorter)
        for k in range(MAX_EXTRA):
            idx = min(k + 1, len(states) - 1)   # clamp to last known state
            is_final = (k + 1) >= len(states)
            bc = border if is_final else None
            draw_pattern(axes[i, 2 + k], states[idx],
                         title="", energy=net.energy(states[idx]),
                         border_color=bc)

        # row label on the left: noise% + step count
        n_steps = len(states) - 1
        step_label = f"{noise:.0%}\n({n_steps} step{'s' if n_steps != 1 else ''})"
        axes[i, 0].text(-0.35, 2.5, step_label, fontsize=9,
                         fontweight="bold", color=STYLE["text_title"],
                         va="center", ha="right", transform=axes[i, 0].transData)

    fig.suptitle(f"Step-by-step recovery — query: {query_name}   {mode_tag(mode)}",
                 fontsize=13, fontweight="bold",
                 color=STYLE["text_title"], y=1.02)
    plt.tight_layout()
    save_fig(fig, out)


# ── Plot 4: spurious state ────────────────────────────────────────────────────

def plot_spurious(stored: dict, query_name: str,
                   noise: float, seed: int, out: Path,
                   mode: str = "sync"):
    apply_style()
    net    = build_net(stored)
    query  = stored[query_name]
    rng    = np.random.default_rng(seed)
    noisy  = flip_noise(query.flatten(), noise, rng)
    result = net.predict(noisy, mode=mode, seed=seed, verbose=False)
    match_name, sim, is_inv = best_match(result, stored)

    panels = [
        (query,              f"Original\n({query_name})",               None,                None),
        (noisy,              f"Noisy input\n({noise:.0%} noise)",        net.energy(noisy),   None),
        (result,             "Network output\n(spurious state)",          net.energy(result),  C_SPURIOUS),
        (stored[match_name], f"Closest stored\n({match_name}, {sim:.0f}% match)", None,        None),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(8.5, 3.0),
                              dpi=FIG_DPI, facecolor=STYLE["figure_bg"])
    for ax, (pat, label, energy, border) in zip(axes, panels):
        draw_pattern(ax, pat, title=label, energy=energy, border_color=border)

    fig.suptitle(f"Spurious State — local energy minimum != any stored pattern   {mode_tag(mode)}",
                 fontsize=12, fontweight="bold", color=STYLE["text_title"], y=1.06)
    plt.tight_layout()
    save_fig(fig, out)


# ── Plot 5: energy convergence (multi-seed, mean +- std) ─────────────────────

def plot_energy_convergence(stored: dict, query_name: str, out: Path,
                             n_seeds: int = 15, mode: str = "sync"):
    apply_style()
    net   = build_net(stored)
    query = stored[query_name]

    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    cmap   = plt.cm.plasma
    colors = [cmap(i / (len(noise_levels) - 1)) for i in range(len(noise_levels))]

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI,
                            facecolor=STYLE["figure_bg"])
    ax.set_facecolor(STYLE["axes_bg"])

    for noise, color in zip(noise_levels, colors):
        all_curves = []
        outcomes   = []
        for seed in range(n_seeds):
            noisy_seed = seed * 100 + int(noise * 10)
            noisy  = flip_noise(query.flatten(), noise,
                                np.random.default_rng(noisy_seed))
            states = run_steps(net, noisy, mode=mode, seed=noisy_seed)
            all_curves.append([net.energy(s) for s in states])
            outcomes.append(classify(states[-1], stored))

        # pad to max length
        raw_steps = [len(c) - 1 for c in all_curves]   # steps per seed
        avg_steps = float(np.mean(raw_steps))
        max_len = max(len(c) for c in all_curves)
        padded  = [c + [c[-1]] * (max_len - len(c)) for c in all_curves]
        arr     = np.array(padded)
        mean    = arr.mean(axis=0)
        std     = arr.std(axis=0)

        # outcome for color of line style: use majority vote
        majority = max(set(outcomes), key=outcomes.count)
        ls = OUTCOME_LS[majority]
        xs = list(range(max_len))

        ax.plot(xs, mean, linestyle=ls, color=color, linewidth=1.8,
                marker="o", markersize=3, label=f"{noise:.0%}")
        ax.fill_between(xs, mean - std, mean + std,
                        color=color, alpha=0.15)

        # annotate mean step count at the end of the curve
        ax.annotate(
            f"  {avg_steps:.1f}",
            xy=(xs[-1], mean[-1]),
            fontsize=7, color=color, va="center",
        )

    ax.set_xlabel("Iteration" if mode == "sync" else "Sweep",
                  color=STYLE["text_axis"])
    ax.set_ylabel("Energy  E(s)", color=STYLE["text_axis"])
    ax.set_title(
        f"Energy convergence — query: {query_name}   {mode_tag(mode)}  "
        f"(mean +- std over {n_seeds} seeds)\n"
        "line style:  solid = exact   dashed = inverse   dotted = spurious",
        fontsize=11, fontweight="bold", color=STYLE["text_title"],
    )
    ax.grid(color=STYLE["grid"], linestyle="--", linewidth=0.7)
    ax.legend(title="Noise", fontsize=8, title_fontsize=8,
              framealpha=0.85, loc="lower right")
    plt.tight_layout()
    save_fig(fig, out)


# ── Plot 6: noise robustness (stacked, multi-seed avg) ───────────────────────

def plot_noise_robustness(stored: dict, out: Path,
                           n_trials: int = 250, mode: str = "sync"):
    apply_style()
    net          = build_net(stored)
    noise_levels = np.arange(0.0, 0.96, 0.04)
    pattern_list = list(stored.values())
    rng          = np.random.default_rng(0)

    exact_f, inverse_f, spurious_f = [], [], []

    for noise in noise_levels:
        counts = {"exact": 0, "inverse": 0, "spurious": 0}
        for trial in range(n_trials):
            query  = pattern_list[trial % len(pattern_list)]
            flat   = flip_noise(query.flatten(), noise, rng)
            result = net.predict(flat, mode=mode, seed=trial, verbose=False)
            counts[classify(result, stored)] += 1
        total = sum(counts.values())
        exact_f.append(counts["exact"]    / total)
        inverse_f.append(counts["inverse"] / total)
        spurious_f.append(counts["spurious"] / total)

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI,
                            facecolor=STYLE["figure_bg"])
    ax.set_facecolor(STYLE["axes_bg"])
    x = noise_levels * 100
    ax.stackplot(x, [exact_f, inverse_f, spurious_f],
                 labels=["Exact match", "Inverse attractor", "Spurious state"],
                 colors=[C_EXACT, C_INVERSE, C_SPURIOUS], alpha=0.85)
    ax.set_xlabel("Noise level (%)", color=STYLE["text_axis"])
    ax.set_ylabel("Fraction of trials", color=STYLE["text_axis"])
    ax.set_title(f"Noise robustness — outcome distribution  "
                 f"({n_trials} trials per level)   {mode_tag(mode)}",
                 fontsize=13, fontweight="bold", color=STYLE["text_title"])
    ax.set_xlim(0, x[-1])
    ax.set_ylim(0, 1)
    ax.grid(color=STYLE["grid"], linestyle="--", linewidth=0.7,
            axis="y", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    plt.tight_layout()
    save_fig(fig, out)


# ── Plot 7: pattern overlap matrix ────────────────────────────────────────────

def plot_overlap_matrix(stored: dict, out: Path):
    apply_style()
    names   = list(stored.keys())
    n       = len(names)
    P       = np.array([stored[nm].flatten() for nm in names])
    overlap = (P @ P.T) / 25.0

    fig, ax = plt.subplots(figsize=(5.0, 4.5), dpi=FIG_DPI,
                            facecolor=STYLE["figure_bg"])
    ax.set_facecolor(STYLE["axes_bg"])

    im = ax.imshow(overlap, cmap="RdBu_r",
                   norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
    for i in range(n):
        for j in range(n):
            c = "white" if abs(overlap[i, j]) > 0.5 else STYLE["text_title"]
            ax.text(j, i, f"{overlap[i, j]:.2f}",
                    ha="center", va="center", fontsize=11, color=c)

    ax.set_xticks(range(n));  ax.set_xticklabels(names, fontsize=10)
    ax.set_yticks(range(n));  ax.set_yticklabels(names, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
        "xi_u · xi_v / N", color=STYLE["text_axis"])
    ax.set_title("Pattern overlap matrix\n"
                 "low off-diagonal = nearly orthogonal patterns",
                 fontsize=12, fontweight="bold", color=STYLE["text_title"])
    plt.tight_layout()
    save_fig(fig, out)


# ── Plot 8: basin of attraction per pattern (mean +- std) ────────────────────

def plot_basin_by_pattern(stored: dict, out: Path,
                           n_seeds: int = 10, n_trials: int = 60,
                           mode: str = "sync"):
    apply_style()
    net          = build_net(stored)
    noise_levels = np.arange(0.0, 0.96, 0.05)
    colors       = plt.cm.tab10(np.linspace(0, 0.4, len(stored)))

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI,
                            facecolor=STYLE["figure_bg"])
    ax.set_facecolor(STYLE["axes_bg"])

    for (name, query), color in zip(stored.items(), colors):
        seed_rates = []
        for seed in range(n_seeds):
            rng       = np.random.default_rng(seed * 7 + 13)
            async_sd  = seed * 7 + 13
            rates = []
            for noise in noise_levels:
                correct = 0
                for trial in range(n_trials):
                    flat   = flip_noise(query.flatten(), noise, rng)
                    result = net.predict(flat, mode=mode,
                                          seed=async_sd + trial, verbose=False)
                    mn, sim, is_inv = best_match(result, stored)
                    if sim == 100.0 and not is_inv and mn == name:
                        correct += 1
                rates.append(correct / n_trials)
            seed_rates.append(rates)

        arr  = np.array(seed_rates)          # (n_seeds, n_noise_levels)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        xs   = noise_levels * 100

        ax.plot(xs, mean, marker="o", markersize=3,
                label=name, color=color, linewidth=2)
        ax.fill_between(xs, mean - std, mean + std,
                        color=color, alpha=0.18)

    ax.axhline(0.5, color=STYLE["grid_minor"], linewidth=1,
               linestyle="--", alpha=0.7, label="50% threshold")
    ax.set_xlabel("Noise level (%)", color=STYLE["text_axis"])
    ax.set_ylabel("Exact recovery rate", color=STYLE["text_axis"])
    ax.set_title(
        f"Basin of attraction per pattern  "
        f"(mean +- std, {n_seeds} seeds x {n_trials} trials)   {mode_tag(mode)}",
        fontsize=13, fontweight="bold", color=STYLE["text_title"],
    )
    ax.set_xlim(0, 95);  ax.set_ylim(-0.05, 1.05)
    ax.grid(color=STYLE["grid"], linestyle="--", linewidth=0.7)
    ax.legend(title="Pattern", fontsize=9, title_fontsize=9, framealpha=0.85)
    plt.tight_layout()
    save_fig(fig, out)


# ── Plot 9: capacity experiment ───────────────────────────────────────────────

def plot_capacity_experiment(all_letters: dict, out: Path,
                              n_subsets: int = 20, n_trials: int = 60,
                              noise: float = 0.2, mode: str = "sync"):
    """
    For each P in 1..max_P, randomly draw P letters, train Hopfield,
    measure recovery rate. Repeat n_subsets times and plot mean +- std.
    Marks the theoretical capacity limit 0.138 * N.
    """
    apply_style()
    letter_names   = list(all_letters.keys())
    letter_patterns = all_letters
    max_P = min(14, len(letter_names))
    sizes = list(range(1, max_P + 1))

    theoretical_limit = 0.138 * 25   # ~3.45 for N=25

    mean_rates, std_rates = [], []
    rng = np.random.default_rng(99)
    trial_counter = 0

    for P in sizes:
        subset_rates = []
        for _ in range(n_subsets):
            chosen  = list(rng.choice(letter_names, size=P, replace=False))
            stored  = {k: letter_patterns[k] for k in chosen}
            net     = build_net(stored)
            correct = 0
            for __ in range(n_trials):
                name   = chosen[rng.integers(P)]
                flat   = flip_noise(stored[name].flatten(), noise, rng)
                result = net.predict(flat, mode=mode,
                                     seed=trial_counter, verbose=False)
                trial_counter += 1
                mn, sim, is_inv = best_match(result, stored)
                if sim == 100.0 and not is_inv and mn == name:
                    correct += 1
            subset_rates.append(correct / n_trials)

        mean_rates.append(np.mean(subset_rates))
        std_rates.append(np.std(subset_rates))

    mean_rates = np.array(mean_rates)
    std_rates  = np.array(std_rates)

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI,
                            facecolor=STYLE["figure_bg"])
    ax.set_facecolor(STYLE["axes_bg"])

    ax.plot(sizes, mean_rates, marker="o", markersize=5,
            color=C_BLUE, linewidth=2, label="Exact recovery rate (mean)")
    ax.fill_between(sizes, mean_rates - std_rates, mean_rates + std_rates,
                    color=C_BLUE, alpha=0.2, label="+-1 std")

    ax.axvline(theoretical_limit, color=C_SPURIOUS, linewidth=1.5,
               linestyle="--",
               label=f"Theoretical limit  0.138 x N = {theoretical_limit:.1f}")

    ax.set_xlabel("Number of stored patterns  (P)", color=STYLE["text_axis"])
    ax.set_ylabel("Exact recovery rate", color=STYLE["text_axis"])
    ax.set_title(
        f"Capacity experiment  (N=25 neurons, noise={noise:.0%}, "
        f"{n_subsets} random subsets x {n_trials} trials each)   {mode_tag(mode)}",
        fontsize=12, fontweight="bold", color=STYLE["text_title"],
    )
    ax.set_xlim(0.5, max_P + 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(sizes)
    ax.grid(color=STYLE["grid"], linestyle="--", linewidth=0.7)
    ax.legend(fontsize=9, framealpha=0.85)
    plt.tight_layout()
    save_fig(fig, out)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate Hopfield network analysis plots."
    )
    parser.add_argument(
        "--mode", choices=["sync", "async"], default="sync",
        help="Update mode: 'sync' (default) or 'async'. "
             "Outputs go to results/hopfield/<mode>/",
    )
    args = parser.parse_args()
    mode = args.mode

    apply_style()

    patterns_file = ROOT / "data" / "patterns_worst.txt"
    letters_file  = ROOT / "data" / "letters.txt"
    out_dir       = ROOT / "results" / "hopfield" / mode

    stored      = load_patterns(str(patterns_file))
    all_letters = load_letters(str(letters_file))
    query_name  = list(stored.keys())[2]   # W

    print(f"Stored patterns : {list(stored.keys())}")
    print(f"Query           : {query_name}")
    print(f"All letters     : {len(all_letters)}")
    print(f"Mode            : {mode}")
    print(f"Output          : {out_dir}\n")

    plot_stored_patterns(stored,
        out=out_dir / "1_stored_patterns.png")

    plot_recovery_grid(stored, noise=0.2,
        seeds=[7, 14, 21, 28],
        out=out_dir / "2_recovery_grid.png", mode=mode)

    plot_recovery_steps(stored, query_name,
        noise_levels=[0.2, 0.4, 0.6],
        seeds=[7, 14, 42],
        out=out_dir / "3_recovery_steps.png", mode=mode)

    plot_spurious(stored, query_name, noise=0.6, seed=42,
        out=out_dir / "4_spurious_state.png", mode=mode)

    plot_energy_convergence(stored, query_name,
        out=out_dir / "5_energy_convergence.png", mode=mode)

    plot_noise_robustness(stored,
        out=out_dir / "6_noise_robustness.png", mode=mode)

    plot_overlap_matrix(stored,
        out=out_dir / "7_overlap_matrix.png")

    plot_basin_by_pattern(stored,
        out=out_dir / "8_basin_by_pattern.png", mode=mode)

    plot_capacity_experiment(all_letters,
        out=out_dir / "9_capacity_experiment.png", mode=mode)

    print("\nDone.")


if __name__ == "__main__":
    main()

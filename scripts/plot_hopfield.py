"""
hopfield_convergence_plot.py
────────────────────────────
Visualises every step of Hopfield network convergence for a single query,
showing the board state and energy at each stage:

    original → noisy (t=0) → t=1 → … → converged

Usage
─────
    python hopfield_convergence_plot.py <patterns_file> <query_letter> \
        [--noise 0.2] [--seed 42] [--max-iter 20] [--out convergence.png]

    <query_letter>   name of the letter to query (must be in patterns_file)
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from utils.letters import load_patterns, load_query, best_match, add_noise, classify_recovery
from hopfield.HopfieldNetwork import HopfieldNetwork

BG        = "#fff5ec"
CELL_ON   = "#343434"
CELL_OFF  = BG
GRID_COL  = "#aaaaaa"

C_EXACT     = "#2ecc71"
C_INVERSE   = "#e67e22"
C_WRONG    = "#d35400"
C_SPURIOUS = "#e74c3c"
C_NEUTRAL  = "#4a90d9"   # original & noisy frames

OUTCOME_COLOR = {
    "exact": C_EXACT, "inverse": C_INVERSE,
    "wrong": C_WRONG, "spurious": C_SPURIOUS,
}
OUTCOME_LABEL = {
    "exact": "exact", "inverse": "inverse",
    "wrong": "wrong pattern", "spurious": "spurious",
}

TITLE_COLOR = "#343434"
SUB_COLOR   = "#555555"

FIG_DPI = 144



def build_net(stored: dict) -> HopfieldNetwork:
    pm = np.array([p.flatten() for p in stored.values()])
    net = HopfieldNetwork(n=25)
    net.initialize_weights(pm)
    return net


def run_steps(net: HopfieldNetwork, initial: np.ndarray,
              max_iter: int = 20) -> list[np.ndarray]:
    """Return [initial, s1, s2, …] up to convergence."""
    states = [initial.copy().astype(float)]
    s = states[0].copy()
    s_pprev = None
    for _ in range(max_iter):
        s_prev = s.copy()
        s = np.where(net.W @ s > 0, 1.0, -1.0)
        if np.array_equal(s, s_prev):          # fixed point (already in states)
            break
        states.append(s.copy())
        if s_pprev is not None and np.array_equal(s, s_pprev):  # 2-cycle
            break
        s_pprev = s_prev
    return states


def draw_cell(ax, state: np.ndarray, border_color: str = None,
              lw: float = 2.5):
    """Draw a ±1 5×5 pattern on *ax*."""
    grid = np.array(state).flatten().reshape(5, 5)
    ax.set_xlim(0, 5);  ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.set_xticks([]);  ax.set_yticks([])
    ax.set_facecolor(BG)

    for r in range(5):
        for c in range(5):
            face = CELL_ON if grid[r, c] == 1 else CELL_OFF
            ax.add_patch(plt.Rectangle(
                (c, 4 - r), 1, 1,
                facecolor=face, edgecolor=GRID_COL, linewidth=0.5,
            ))

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(border_color if border_color else GRID_COL)
        spine.set_linewidth(lw if border_color else 0.8)


def plot_convergence(patterns_file: str, query_file: str,
                     noise: float, seed: int, max_iter: int,
                     out_path: Path):

    stored = load_patterns(patterns_file)
    query  = load_query(query_file)

    # identify closest stored pattern for labelling purposes only
    query_letter, sim, is_inv = best_match(query.flatten(), stored)
    if sim == 100.0 and not is_inv:
        label_str = query_letter
    else:
        label_str = f"query ({sim:.0f}%~{query_letter})"

    net   = build_net(stored)
    orig  = query.flatten()
    noisy = add_noise(query, noise, seed=seed).flatten()

    # collect all update states
    update_states = run_steps(net, noisy, max_iter)
    n_updates = len(update_states) - 1
    # duplicate final state to show fixed-point convergence (display only)
    if n_updates >= 1:
        update_states.append(update_states[-1].copy())
    outcome = classify_recovery(update_states[-1], orig, stored)
    border  = OUTCOME_COLOR[outcome]

    # columns: original | t=0 (noisy, fed in) | t=1 | … | t=N
    n_cols       = 1 + len(update_states)
    cell_w       = 1.4                         # inches per cell
    cell_h       = 1.8
    fig_w        = max(8, n_cols * cell_w + 1.0)

    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(fig_w, cell_h + 1.6),
        facecolor=BG,
        dpi=FIG_DPI,
    )
    fig.set_dpi(FIG_DPI)
    axes = np.atleast_1d(axes)

    n_flipped = int(np.sum(orig != noisy))

    def _annotate_panel(ax, state, title, title_color, title_weight, border_c, lw,
                        energy, energy_color):
        draw_cell(ax, state, border_color=border_c, lw=lw)
        ax.set_title(title, fontsize=8, color=title_color, fontweight=title_weight,
                     pad=6, ha="center")
        ax.text(0.5, -0.10, f"E = {energy:.2f}", transform=ax.transAxes,
                ha="center", va="top", fontsize=7.5, color=energy_color)

    _annotate_panel(
        axes[0], orig,
        f"Original\n[{label_str}]",
        TITLE_COLOR, "normal", C_NEUTRAL, 1.8,
        net.energy(orig), SUB_COLOR,
    )

    n_states = len(update_states)
    for si, state in enumerate(update_states):
        ax = axes[1 + si]
        is_converged_copy = n_updates >= 1 and si == n_states - 1
        e = net.energy(state)

        if si == 0:
            title = f"Noisy input\n(t = 0, {noise:.0%} / {n_flipped} px)"
            bc, lw = C_NEUTRAL, 1.8
            tc, tw, ec = TITLE_COLOR, "normal", SUB_COLOR
        elif is_converged_copy:
            title = f"t = {si}\n({OUTCOME_LABEL[outcome]})"
            bc, lw = border, 3.0
            tc, tw, ec = border, "bold", border
        else:
            title = f"t = {si}"
            bc, lw = "#888888", 1.8
            tc, tw, ec = TITLE_COLOR, "normal", SUB_COLOR

        _annotate_panel(ax, state, title, tc, tw, bc, lw, e, ec)

    fig.subplots_adjust(wspace=0.35, bottom=0.14, top=0.92)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight",
                pad_inches=0.25, facecolor=BG)
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Hopfield step-by-step convergence with energy."
    )
    parser.add_argument("patterns_file",
                        help="Patterns file (same format as hopfield.py)")
    parser.add_argument("query_file",
                        help="Query file (same format as hopfield.py)")
    parser.add_argument("--noise",    type=float, default=0.2,
                        help="Noise fraction (default 0.2)")
    parser.add_argument("--seed",     type=int,   default=42,
                        help="RNG seed (default 42)")
    parser.add_argument("--max-iter", type=int,   default=20,
                        help="Max update iterations (default 20)")
    parser.add_argument("--out",      type=str,   default="convergence.png",
                        help="Output PNG path (default convergence.png)")
    args = parser.parse_args()

    plot_convergence(
        patterns_file=args.patterns_file,
        query_file=args.query_file,
        noise=args.noise,
        seed=args.seed,
        max_iter=args.max_iter,
        out_path=Path(args.out),
    )


if __name__ == "__main__":
    main()
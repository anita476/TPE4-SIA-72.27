"""
hopfield_convergence_plot.py
────────────────────────────
Visualises every step of Hopfield network convergence for a single query,
showing the board state and energy at each stage:

    original → noisy query → step 1 → step 2 → … → converged

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
import matplotlib.patches as mpatches
from pathlib import Path

# ── path setup (mirrors hopfield_analysis.py) ─────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from utils.letters import load_patterns, load_query, best_match, add_noise
from hopfield.HopfieldNetwork import HopfieldNetwork

BG        = "#fff5ec"
CELL_ON   = "#343434"
CELL_OFF  = BG
GRID_COL  = "#aaaaaa"

C_EXACT    = "#2ecc71"
C_INVERSE  = "#e67e22"
C_SPURIOUS = "#e74c3c"
C_NEUTRAL  = "#4a90d9"   # original & noisy frames

OUTCOME_COLOR = {"exact": C_EXACT, "inverse": C_INVERSE, "spurious": C_SPURIOUS}

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
        states.append(s.copy())
        if np.array_equal(s, s_prev):          # fixed point
            break
        if s_pprev is not None and np.array_equal(s, s_pprev):  # 2-cycle
            break
        s_pprev = s_prev
    return states


def classify(result: np.ndarray, stored: dict) -> str:
    _, sim, is_inv = best_match(result, stored)
    if sim == 100.0 and not is_inv:  return "exact"
    if sim == 100.0 and is_inv:      return "inverse"
    return "spurious"


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
    outcome       = classify(update_states[-1], stored)
    border        = OUTCOME_COLOR[outcome]

    # columns: original | noisy | step-0 | step-1 | … | step-N
    # step-0 is the noisy input seen from the network's perspective (= noisy)
    # step-1 … step-N are the update states
    n_steps      = len(update_states)          # includes the initial noisy
    n_cols       = 2 + n_steps                 # original + noisy + all steps
    cell_w       = 1.4                         # inches per cell
    cell_h       = 1.8
    fig_w        = max(8, n_cols * cell_w + 1.0)

    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(fig_w, cell_h + 1.0),
        facecolor=BG,
        dpi=FIG_DPI,
    )
    fig.set_dpi(FIG_DPI)

    ax = axes[0]
    draw_cell(ax, orig, border_color=C_NEUTRAL)
    e_orig = net.energy(orig)
    ax.set_title(f"Original\n[{label_str}]",
                 fontsize=8, color=TITLE_COLOR, pad=4)
    ax.text(2.5, -0.55, f"E = {e_orig:.2f}",
            ha="center", va="top", fontsize=7.5, color=SUB_COLOR,
            transform=ax.transData)

    ax = axes[1]
    draw_cell(ax, noisy, border_color=C_NEUTRAL)
    e_noisy = net.energy(noisy)
    n_flipped = int(np.sum(orig != noisy))
    ax.set_title(f"Noisy input\n({noise:.0%} / {n_flipped} px)",
                 fontsize=8, color=TITLE_COLOR, pad=4)
    ax.text(2.5, -0.55, f"E = {e_noisy:.2f}",
            ha="center", va="top", fontsize=7.5, color=SUB_COLOR,
            transform=ax.transData)



    for si, state in enumerate(update_states):
        ax = axes[2 + si]
        is_last = (si == len(update_states) - 1)

        if si == 0:
            label = "t = 0\n(fed in)"
            bc    = C_NEUTRAL
        else:
            label = f"t = {si}"
            bc    = border if is_last else "#888888"

        draw_cell(ax, state, border_color=bc,
                  lw=3.0 if is_last else 1.8)

        e = net.energy(state)
        ax.set_title(label, fontsize=8,
                     color=border if is_last else TITLE_COLOR, pad=4)
        ax.text(2.5, -0.55, f"E = {e:.2f}",
                ha="center", va="top", fontsize=7.5,
                color=border if is_last else SUB_COLOR,
                transform=ax.transData)

        if is_last:
            ax.set_title(f"t = {si}  \n({outcome})",
                         fontsize=8, color=border,
                         fontweight="bold", pad=4)

    fig.subplots_adjust(wspace=0.35)

    _, sim, is_inv = best_match(update_states[-1], stored)
    match_name, _, _ = best_match(update_states[-1], stored)
    outcome_str = {
        "exact":    f"exact recovery of [{match_name}]",
        "inverse":  f"inverse of [{match_name}]",
        "spurious": f"spurious state ({sim:.0f}% sim. to [{match_name}])",
    }[outcome]

    fig.suptitle(
        f"Hopfield convergence  ·  query [{label_str}]  ·  "
        f"noise {noise:.0%}  ·  seed {seed}  ·  "
        f"converged in {len(update_states)-1} step(s)  →  {outcome_str}",
        fontsize=9, color=TITLE_COLOR, y=1.04,
    )

    legend_patches = [
        mpatches.Patch(facecolor=C_NEUTRAL,  label="Original / fed-in state"),
        mpatches.Patch(facecolor=C_EXACT,    label="Exact recovery"),
        mpatches.Patch(facecolor=C_INVERSE,  label="Inverse attractor"),
        mpatches.Patch(facecolor=C_SPURIOUS, label="Spurious state"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=4,
        fontsize=8,
        framealpha=0.7,
        facecolor=BG,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, -0.12),
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight",
                pad_inches=0.2, facecolor=BG)
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
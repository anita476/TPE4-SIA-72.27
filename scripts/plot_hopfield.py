"""
hopfield_convergence_plot.py
────────────────────────────
Visualises every step of Hopfield network convergence for a single query,
showing the board state and energy at each stage:

    original → noisy (t=0) → t=1 → … → converged

Usage
─────
    python plot_hopfield.py <patterns_file> <query_file> \
        [--noise 0.2] [--seed 42] [--max-iter 20] [--mode sync|async] [--out convergence.png]

    <query_letter>   name of the letter to query (must be in patterns_file)
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
              mode: str = "sync", max_iter: int = 20,
              seed: int = None) -> list[np.ndarray]:
    """Return [s(0), s(1), …] after each sync step or async sweep until convergence."""
    states = [initial.copy().astype(float)]
    s = states[0].copy()

    if mode == "sync":
        s_pprev = None
        for _ in range(max_iter):
            s_prev = s.copy()
            s = np.where(net.W @ s > 0, 1.0, -1.0)
            states.append(s.copy())          # always record, even the converged copy
            if np.array_equal(s, s_prev):    # period-1 fixed point
                break
            if s_pprev is not None and np.array_equal(s, s_pprev):  # period-2
                break
            s_pprev = s_prev
        return states

    if mode == "async":
        rng = np.random.default_rng(seed)
        for _ in range(max_iter):
            order = rng.permutation(net.n)
            changes = 0
            for i in order:
                h_i = float(net.W[i] @ s)
                new_i = 1.0 if h_i > 0 else -1.0
                if new_i != s[i]:
                    s[i] = new_i
                    changes += 1
            states.append(s.copy())          # always record, even the 0-change sweep
            if changes == 0:                 # fixed point confirmed
                break
        return states

    raise ValueError(f"Unknown mode '{mode}'. Use 'sync' or 'async'.")


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


FIELD_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "field", ["#378ADD", BG, CELL_ON], N=256
)


def draw_field(ax, h_flat: np.ndarray, title: str,
               title_color: str = TITLE_COLOR, border_color: str = None,
               lw: float = 1.2):
    """Draw the 25-element local field h = W·s as a coloured 5×5 grid with values."""
    h = np.array(h_flat).flatten().reshape(5, 5)
    abs_max = max(float(np.abs(h).max()), 1e-9)
    norm_fn = mcolors.Normalize(vmin=-abs_max, vmax=abs_max)
    cmap_fn = plt.get_cmap(FIELD_CMAP)

    ax.set_facecolor(BG)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    for r in range(5):
        for c in range(5):
            val = h[r, c]
            face = cmap_fn(norm_fn(val))
            ax.add_patch(plt.Rectangle(
                (c, 4 - r), 1, 1,
                facecolor=face, edgecolor=GRID_COL, linewidth=0.5,
            ))
            norm_abs = abs(val) / abs_max
            txt_col = "white" if norm_abs > 0.55 else "#333333"
            ax.text(c + 0.5, 4.5 - r, f"{val:.1f}",
                    ha="center", va="center",
                    fontsize=6, color=txt_col, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(border_color if border_color else GRID_COL)
        spine.set_linewidth(lw if border_color else 0.8)

    ax.set_title(title, fontsize=7.5, color=title_color, pad=5, ha="center")


def plot_fields(net, orig: np.ndarray, update_states: list,
                noise: float, n_flipped: int, outcome: str,
                label_str: str, mode: str, out_path: Path):
    """
    Save a separate figure showing h(t) = W·s(t) at every step.

    Layout (one column per step):
      col 0 : W · ξ_orig   — confirms the original is a fixed point
      col 1 : h(0) = W·s(0) — field that produced s(1)
      col 2 : h(1) = W·s(1) — field that produced s(2)
      …
      col T : h(T) = W·s(T) — field of the converged state (confirms stability)
    """
    border = OUTCOME_COLOR[outcome]
    n_states = len(update_states)          # s(0) … s(T)
    n_cols   = 1 + n_states               # orig + every update state

    cell_w, cell_h = 1.4, 1.6
    fig_w = max(8, n_cols * cell_w + 1.0)

    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(fig_w, cell_h + 1.8),
        facecolor=BG,
        dpi=FIG_DPI,
    )
    axes = np.atleast_1d(axes)

    # ── col 0 : field of the original pattern ────────────────────────────────
    h_orig = net.W @ orig
    draw_field(axes[0], h_orig,
               title=f"campo de ξ\n(original [{label_str}])",
               title_color=TITLE_COLOR, border_color=C_NEUTRAL, lw=1.8)

    # ── cols 1 … n_states : h(t) = W·s(t) ───────────────────────────────────
    for si, state in enumerate(update_states):
        ax = axes[1 + si]
        h = net.W @ state
        is_last = (si == n_states - 1)
        step_lbl = f"sweep {si}" if mode == "async" else f"t = {si}"

        if si == 0:
            title  = f"h(0) = W·s(0)\n(noisy, {noise:.0%} / {n_flipped} px)"
            bc, lw = C_NEUTRAL, 1.8
            tc     = TITLE_COLOR
        elif is_last:
            title  = f"h({si}) = W·s({si})\n({OUTCOME_LABEL[outcome]}, {step_lbl})"
            bc, lw = border, 2.8
            tc     = border
        else:
            title  = f"h({si}) = W·s({si})\n({step_lbl})"
            bc, lw = "#888888", 1.4
            tc     = TITLE_COLOR

        draw_field(ax, h, title=title, title_color=tc, border_color=bc, lw=lw)

    fig.subplots_adjust(wspace=0.35, bottom=0.08, top=0.92)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight",
                pad_inches=0.25, facecolor=BG)
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_convergence(patterns_file: str, query_file: str,
                     noise: float, seed: int, max_iter: int,
                     out_path: Path, mode: str = "sync",
                     show_fields: bool = True):

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
    update_states = run_steps(net, noisy, mode=mode, max_iter=max_iter, seed=seed)
    n_updates = len(update_states) - 1   # last state is the convergence-check copy
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
            lbl = f"sweep {si}" if mode == "async" else f"t = {si}"
            title = f"{lbl}\n({OUTCOME_LABEL[outcome]})"
            bc, lw = border, 3.0
            tc, tw, ec = border, "bold", border
        else:
            title = f"sweep {si}" if mode == "async" else f"t = {si}"
            bc, lw = "#888888", 1.8
            tc, tw, ec = TITLE_COLOR, "normal", SUB_COLOR

        _annotate_panel(ax, state, title, tc, tw, bc, lw, e, ec)

    fig.subplots_adjust(wspace=0.35, bottom=0.14, top=0.92)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight",
                pad_inches=0.25, facecolor=BG)
    plt.close(fig)
    print(f"Saved → {out_path}")

    # ── optional separate fields figure ──────────────────────────────────────
    if show_fields:
        fields_path = out_path.with_stem(out_path.stem + "_fields")
        plot_fields(
            net=net,
            orig=orig,
            update_states=update_states,
            noise=noise,
            n_flipped=n_flipped,
            outcome=outcome,
            label_str=label_str,
            mode=mode,
            out_path=fields_path,
        )


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
    parser.add_argument(
        "--mode", choices=["sync", "async"], default="sync",
        help="Update mode: sync (all neurons at once) or async (one sweep = one step)",
    )
    parser.add_argument(
        "--no-fields", action="store_true",
        help="Skip the separate local-field figure (saved as <out>_fields.png by default)",
    )
    args = parser.parse_args()

    plot_convergence(
        patterns_file=args.patterns_file,
        query_file=args.query_file,
        noise=args.noise,
        seed=args.seed,
        max_iter=args.max_iter,
        out_path=Path(args.out),
        mode=args.mode,
        show_fields=not args.no_fields,
    )


if __name__ == "__main__":
    main()
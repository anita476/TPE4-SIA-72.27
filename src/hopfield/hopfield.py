import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding='utf-8')

from utils.letters import (
    load_letters, group_analysis, best_match, load_query, load_patterns,
    add_noise, classify_recovery,
)
from utils.display_hopfield import print_pattern, print_separator
from HopfieldNetwork import HopfieldNetwork
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Run a Hopfield network on stored letter patterns."
    )
    parser.add_argument(
        "patterns_file",
        help="Text file with stored patterns (5x5, '*'/space, separated by '=')",
    )
    parser.add_argument(
        "query_file",
        help="Text file with a single 5x5 query pattern ('*'/space, no separator needed)",
    )
    parser.add_argument(
        "--max-iter", type=int, default=20,
        help="Maximum update iterations (default: 20)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Only print the final result, skip per-step output",
    )
    parser.add_argument("--noise", type=float, default=0.2)
    parser.add_argument("--seed", help="Seed for reproducibility", type=int)
    parser.add_argument(
        "--mode", choices=["sync", "async"], default="sync",
        help="Update mode: 'sync' (all neurons at once) or 'async' (one at a time, "
             "random order). Default: sync.",
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Print group orthogonality analysis of all letters before running",
    )
    args = parser.parse_args()

    if args.analyze:
        letters = load_letters("../../data/letters.txt")
        print("---------------------------------------------------")
        print("------------------GROUP ANALYSIS ------------------")
        group_analysis(letters)

    # load
    stored = load_patterns(args.patterns_file)
    query = load_query(args.query_file)
    noisy_query = add_noise(query, args.noise,args.seed)

    print_separator('═')
    print("  Stored patterns")
    print_separator()
    for name, pat in stored.items():
        print_pattern(name, pat)
        print()

    print_separator('═')
    print("  Query pattern")
    print_separator()
    print_pattern("query", query)
    print_pattern("noisy_query", noisy_query)
    print()

    # train
    pattern_matrix = np.array([p.flatten() for p in stored.values()])
    net = HopfieldNetwork(n=25)
    net.initialize_weights(pattern_matrix)

    print_separator('═')
    print("  Running Hopfield network")
    print_separator()


    result = net.predict(
        noisy_query.flatten(),
        mode=args.mode,
        max_iterations=args.max_iter,
        verbose=not args.quiet,
        seed=args.seed,
    )

    # result
    print_separator('═')
    print("  Final result")
    print_separator()
    print_pattern("result", result)
    print()

    orig = query.flatten()
    outcome = classify_recovery(result, orig, stored)
    match, similarity, _ = best_match(result, stored)

    outcome_msg = {
        "exact":    "exact recovery of original",
        "inverse":  "inverse of original",
        "wrong":    f"wrong pattern [{match}]",
        "spurious": f"spurious state ({similarity:.1f}% match with [{match}])",
    }
    print(f"  Result                 : {outcome_msg[outcome]}")

    print(f"  Final energy           : {net.energy(result):.4f}")
    print_separator('═')


if __name__ == "__main__":
    main()
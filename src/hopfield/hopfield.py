from utils.letters import load_letters, group_analysis, best_match, load_query,load_patterns,add_noise
from utils.display_hopfield import print_pattern, print_separator, pattern_to_str
from HopfieldNetwork import HopfieldNetwork
import numpy as np
import argparse

#
def main():
    letters = load_letters("../../data/letters.txt")
    n = 6
    letters_list = list(letters.values())

    remainder = len(letters_list) % n
    if remainder != 0:
        letters_list += [np.ones((5, 5)) * (-1)] * (n - remainder)

    # print the letter plots
    """
    for i in range(len(letters_list) // n):
        letter_group = letters_list[i * n:(i + 1) * n]
        print_letters_line(letter_group)
    """
    group_analysis(letters)

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
    parser.add_argument("--seed",help="Seed for reproducibility",type=int)
    args = parser.parse_args()

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
        max_iterations=args.max_iter,
        verbose=not args.quiet,
    )

    # result
    print_separator('═')
    print("  Final result")
    print_separator()
    print_pattern("result", result)
    print()

    match, similarity = best_match(result, stored)
    print(f"  Closest stored pattern : {match}  ({similarity:.1f}% match)")
    print(f"  Final energy           : {net.energy(result):.4f}")
    print_separator('═')

if __name__ == "__main__":
    main()
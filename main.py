import argparse
from pathlib import Path
from pca_plot import plot_pca


def main():
    parser = argparse.ArgumentParser(description="Visualise linear non-separability of a binary classification dataset.")
    parser.add_argument("--data", default="data/europe.csv",required=True, help="Path to CSV dataset")
    parser.add_argument("--out", default="results/plots", help="Output directory")
    parser.add_argument("--seed", default=1, help="Seed for this run")
    args = parser.parse_args()

    data_dir = Path(args.data)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed

    plot_pca(seed, data_dir, out_dir)

    print(f"\nDone. Plots saved to: {out_dir}")
    

if __name__ == "__main__":
    main()
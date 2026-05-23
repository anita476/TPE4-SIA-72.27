import numpy as np

def pattern_to_str(matrix: np.ndarray) -> str:
    """Render a 5x5 matrix as a printable string."""
    rows = []
    for row in matrix.reshape(5, 5):
        rows.append(' '.join('*' if v == 1 else '.' for v in row))
    return '\n'.join(rows)


def print_pattern(name: str, matrix: np.ndarray):
    print(f"  [{name}]")
    for line in pattern_to_str(matrix).splitlines():
        print(f"    {line}")


def print_separator(char='─', width=40):
    print(char * width)


import numpy as np

def pattern_to_str(matrix: np.ndarray) -> str:
    """Render a flat or 2-D pattern as a printable grid."""
    flat = matrix.flatten()
    side = int(len(flat) ** 0.5)
    rows = []
    for row in flat.reshape(side, side):
        rows.append(' '.join('*' if v == 1 else '.' for v in row))
    return '\n'.join(rows)


def print_pattern(name: str, matrix: np.ndarray):
    print(f"  [{name}]")
    for line in pattern_to_str(matrix).splitlines():
        print(f"    {line}")


def print_separator(char='─', width=40):
    print(char * width)


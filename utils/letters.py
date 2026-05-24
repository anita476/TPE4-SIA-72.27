import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd
import string

BG_COLOR='#fff5ec'
MY_CMAP = ListedColormap([BG_COLOR, '#76373b'])
def create_letter_plot(letters,ax,cmap=MY_CMAP,bg_color=BG_COLOR):
    ax.set_facecolor(bg_color)
    p = sns.heatmap(letters, ax=ax, annot=False,cbar=False, cmap=cmap,square=True,linewidth=2,linecolor='black')
    p.xaxis.set_visible(False)
    p.yaxis.set_visible(False)
    return p


def print_letters_line(letters,cmap=MY_CMAP,cmaps=[],bg_color=BG_COLOR):
    fig,ax =plt.subplots(1,len(letters))
    fig.set_dpi(360)
    fig.patch.set_facecolor(bg_color)
    if not cmaps:
        cmaps =[cmap]*len(letters)
    if len(cmaps)!=len(letters):
        raise Exception("cmap list should be the same length as letters")
    for i,subplot in enumerate(ax):
        create_letter_plot(letters[i].reshape(5,5),ax=subplot,cmap=cmaps[i])
    plt.show()

def load_letters(filepath):
    letters = {}
    current = np.ones((5, 5)) * -1
    idx = 0

    with open(filepath) as fp:
        for line in fp:
            stripped = line.strip('\n')
            if stripped == '=':
                if idx > 0:  # save whatever we've built so far
                    letters[string.ascii_uppercase[len(letters)]] = current
                current = np.ones((5, 5)) * -1
                idx = 0
            elif idx < 5:
                for i, c in enumerate(stripped.ljust(5)):
                    if i < 5:
                        current[idx][i] = 1 if c == '*' else -1
                idx += 1

    if idx > 0:
        letters[string.ascii_uppercase[len(letters)]] = current

    return letters


def load_query(filepath: str) -> np.ndarray:
    """
    Load a single 5x5 pattern from a text file
    Same '*'/space encoding as load_patterns.
    """
    pattern = np.ones((5, 5)) * -1
    with open(filepath) as fp:
        for idx, line in enumerate(fp):
            if idx >= 5:
                break
            stripped = line.strip('\n')
            for i, c in enumerate(stripped.ljust(5)):
                if i < 5:
                    pattern[idx][i] = 1 if c == '*' else -1
    return pattern


def load_patterns(filepath: str) -> dict[str, np.ndarray]:
    letters = {}
    current = np.ones((5, 5)) * -1
    current_name = None
    idx = 0

    with open(filepath) as fp:
        for line in fp:
            stripped = line.strip('\n')
            if stripped.startswith('='):
                if idx > 0 and current_name is not None:   # save previous pattern
                    letters[current_name] = current
                current = np.ones((5, 5)) * -1
                current_name = stripped[1:].strip() or string.ascii_uppercase[len(letters)]
                idx = 0
            elif idx < 5:
                for i, c in enumerate(stripped.ljust(5)):
                    if i < 5:
                        current[idx][i] = 1 if c == '*' else -1
                idx += 1

    if idx > 0 and current_name is not None:   # save last pattern
        letters[current_name] = current

    return letters


def best_match(result: np.ndarray, stored: dict[str, np.ndarray]) -> tuple[str, float, bool]:
    """
    Return (name, similarity%, is_inverse) for the stored pattern closest to result.
    Checks both result and -result so inverse attractors are detected correctly.
    """
    flat = result.flatten()
    best_name, best_score, is_inverse = None, -np.inf, False
    for name, pat in stored.items():
        pat_flat = pat.flatten()
        score     = float(np.dot( flat, pat_flat))
        inv_score = float(np.dot(-flat, pat_flat))
        if score > best_score:
            best_score, best_name, is_inverse = score, name, False
        if inv_score > best_score:
            best_score, best_name, is_inverse = inv_score, name, True
    similarity = best_score / len(flat) * 100
    return best_name, similarity, is_inverse


def group_analysis(letters):
    flat_letters = {
        k: m.flatten() for k, m in letters.items()
    }
    all_groups = itertools.combinations(flat_letters.keys(), 4)

    avg_dot_product = []
    max_dot_product = []

    for g in all_groups:
        group = np.array([v for k,v in flat_letters.items() if k in g])
        orto_matrix = group.dot(group.T)
        np.fill_diagonal(orto_matrix, 0)
        #print(f'{g}\n{orto_matrix}\n-------------------------')
        row, _ = orto_matrix.shape
        avg_dot_product.append((np.abs(orto_matrix).sum()/(orto_matrix.size - row),g))
        mav_v = np.abs(orto_matrix).max()
        max_dot_product.append(((mav_v,np.count_nonzero(np.abs(orto_matrix)==mav_v)/2),g))


    # ahora imprimo los grupos de valores mas bajo, medio, alto
    df = pd.DataFrame(sorted(avg_dot_product),columns=['|<,>| medio','group'])
    #df.head(15).style.format({'|<,>| medio':"{:.2f}"}).hide(axis='index')
    print("Average Dot product\n-------------------")

    print("Best 15:\n")
    print(df.head(15).to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    print("Worst 5:\n")
    print(df.tail(5).to_string(index=False,float_format=lambda  x:f'{x:.2f}'))
    print("Max Dot product\n-------------------")

    print("Best 15 (lowest)\n")
    df2 = pd.DataFrame(sorted(max_dot_product),columns=['|<,>| max','group'])
    print(df2.head(15).to_string(index=False, formatters={'|<,>| max': lambda t: f'max: {t[0]:.0f} | count: {int(t[1])}'}))

    df3 = df2.merge(df)
    df3 = df3[['|<,>| max','|<,>| medio', 'group']]

    print(df3.head(15).to_string(index=False,float_format=lambda  x:f'{x:.2f}'))

    return


def add_noise(pattern: np.ndarray, noise_pct: float, seed: int = None) -> np.ndarray:
    """
    Flip a percentage of pixels in a 5x5 pattern randomly
    """
    rng = np.random.default_rng(seed)

    flat = pattern.flatten().copy()
    n_flip = max(1, int(len(flat) * noise_pct))
    flip_indices = rng.choice(len(flat), size=n_flip, replace=False)
    flat[flip_indices] *= -1

    return flat.reshape(pattern.shape)
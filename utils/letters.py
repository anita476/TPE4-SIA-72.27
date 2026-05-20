import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string

def create_letter_plot(letters,ax,cmap='Blues'):
    p = sns.heatmap(letters, ax=ax, annot=False,cbar=False, cmap=cmap,square=True,linewidth=2,linecolor='black')
    p.xaxis.set_visible(False)
    p.yaxis.set_visible(False)
    return p


def print_letters_line(letters,cmap='Blues',cmaps=[]):
    fig,ax =plt.subplots(1,len(letters))
    fig.set_dpi(360)
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
from utils.letters import load_letters, print_letters_line, group_analysis
import numpy as np

# plot the letters
def main():
    letters = load_letters("../data/letters.txt")
    n = 6
    letters_list = list(letters.values())

    remainder = len(letters_list) % n
    if remainder != 0:
        letters_list += [np.ones((5, 5)) * (-1)] * (n - remainder)

    # print the letter plots
    for i in range(len(letters_list) // n):
        letter_group = letters_list[i * n:(i + 1) * n]
        print_letters_line(letter_group)
if __name__ == "__main__":
    main()
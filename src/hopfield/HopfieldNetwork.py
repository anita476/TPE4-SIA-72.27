import numpy as np


class HopfieldNetwork:
    # n is number of neurons
    def __init__(self,n: int):
        self.n = n
        self.W = np.zeros((n,n)) # we have to initialize weigths


    # we know that:
    # w_i,j is 1/neurons * sum (pat_i * pat_j ^ T) forall patterns
    # this means the weigth matrix can be initially computed by doing :
    def train(self,patterns: np.ndarray):
        return


import numpy as np


def _step(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1.0, -1.0) # step perceptron !


class HopfieldNetwork:

    # n is number of neurons
    def __init__(self,n: int):
        self.n = n
        self.W = np.zeros((n,n)) # we have to initialize weigths

    # we know that:
    # w_i,j is 1/neurons * sum (pat_i * pat_j ^ T) forall patterns
    # this means the weigth matrix can be initially computed by doing :
    def initialize_weights(self,patterns: np.ndarray):
        self.W = (patterns.T @ patterns) / self.n  # ← corrected
        np.fill_diagonal(self.W, 0)
        return

    def predict(self, initial_state: np.array,max_iterations:int, randomize:bool):
        """
            S_i(t+1) = sign( sum_{j != i} w_ij * S_j(t) )
            iterate until S(t+1) == S(t) or max_iterations is reached
            """
        s = initial_state.copy().astype(float)
        for _ in range(max_iterations):
            s_prev = s.copy()
            s = _step(self.W @ s)  # if i=j it doesnt matter bc diagonal is 0
            if np.array_equal(s, s_prev):  #  @todo add other cpnvergence methods
                break
        return s


    def energy(self, state:np.array):
        return -0.5 * state @ self.W @ state

    # @todo investigate
    def max_patterns(self) -> float:
        return 0.15 * self.n
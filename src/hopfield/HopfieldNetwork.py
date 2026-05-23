import numpy as np
from utils.letters import print_letters_line
from utils.display_hopfield import print_pattern, print_separator, pattern_to_str

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

    def predict(
            self,
            initial_state: np.ndarray,
            max_iterations: int = 20,
            verbose: bool = True,
    ) -> np.ndarray:

        """
        S_i(t+1) = sign( sum_{j != i} w_ij * S_j(t) )
        iterate until S(t+1) == S(t) or max_iterations is reached
        """

        s = initial_state.copy().astype(float)

        if verbose:
            print("  Initial state  (energy: {:.2f})".format(self.energy(s)))
            print_pattern("init", s)

        for step in range(1, max_iterations + 1):
            s_prev = s.copy()
            s = _step(self.W @ s)

            if verbose:
                print(f"\n  Step {step}  (energy: {self.energy(s):.2f})")
                print_pattern(f"t={step}", s)

            if np.array_equal(s, s_prev): # @todo should have a convergence window
                if verbose:
                    print(f"\n  Converged after {step} step(s).")
                break
        else:
            if verbose:
                print(f"\n  Did not converge in {max_iterations} steps.")

        return s

    def energy(self, state:np.array):
        return -0.5 * state @ self.W @ state

    # @todo investigate
    def max_patterns(self) -> int:
        return 0.15 * self.n
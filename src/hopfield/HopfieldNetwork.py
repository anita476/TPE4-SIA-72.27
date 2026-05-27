import numpy as np
from utils.display_hopfield import print_pattern


def _step(x: np.ndarray) -> np.ndarray:
    """Step function: +1 if x > 0, else -1.""" 
    return np.where(x > 0, 1.0, -1.0)   # Step perceptron!


class HopfieldNetwork:
    """
    Hopfield network with Hebbian weight rule.

    Supports two update modes:
      - "sync"  : all neurons update simultaneously each step.
                  May converge to a fixed point or oscillate in a 2-cycle.
      - "async" : one neuron at a time updates in random order.
                  Guaranteed to converge to a fixed point (no cycles possible).
                  One "step" = one full sweep through all N neurons.
    """

    # n is the number of neurons
    def __init__(self, n: int):
        self.n = n
        self.W = np.zeros((n, n)) # We have to initialize weights

    # we know that:
    # w_i,j is 1/neurons * sum (pat_i * pat_j ^ T) forall patterns
    # this means the weigth matrix can be initially computed by doing :
    def initialize_weights(self, patterns: np.ndarray):
        self.W = (patterns.T @ patterns) / self.n
        np.fill_diagonal(self.W, 0)

    def predict(
        self,
        initial_state: np.ndarray,
        mode: str = "sync",
        max_iterations: int = 20,
        verbose: bool = True,
        seed: int = None,
    ) -> np.ndarray:
        """
        Iterate the network until convergence or max_iterations.

        Parameters
        ----------
        initial_state : (N,) array of ±1
        mode          : "sync"  → synchronous update (all neurons at once)
                        "async" → asynchronous update (one neuron at a time,
                                  random order, one sweep = one step)
        max_iterations: maximum number of steps (sweeps in async mode)
        verbose       : print state and energy after each step
        seed          : RNG seed (only used in async mode for neuron ordering)
        """
        if mode == "sync":
            return self._predict_sync(initial_state, max_iterations, verbose)
        if mode == "async":
            return self._predict_async(initial_state, max_iterations, verbose, seed)
        raise ValueError(f"Unknown mode '{mode}'. Use 'sync' or 'async'.")

    def _predict_sync(self, initial_state, max_iterations, verbose):
        s = initial_state.copy().astype(float)

        if verbose:
            print(f"  [sync] initial state  (energy: {self.energy(s):.2f})")
            print_pattern("init", s)

        s_pprev = None
        for step in range(1, max_iterations + 1):
            s_prev = s.copy()
            s = _step(self.W @ s)

            if verbose:
                print(f"\n  [sync] step {step}  (energy: {self.energy(s):.2f})")
                print_pattern(f"t={step}", s)

            # period-1: fixed point
            if np.array_equal(s, s_prev):
                if verbose:
                    print(f"\n  Converged after {step} step(s).")
                return s
            # period-2: cycle
            if s_pprev is not None and np.array_equal(s, s_pprev):
                if verbose:
                    print(f"\n  Cycle-2 detected after {step} step(s).")
                return s
            s_pprev = s_prev

        if verbose:
            print(f"\n  Did not converge in {max_iterations} steps.")
        return s

    def _predict_async(self, initial_state, max_iterations, verbose, seed):
        s   = initial_state.copy().astype(float)
        rng = np.random.default_rng(seed)

        if verbose:
            print(f"  [async] initial state  (energy: {self.energy(s):.2f})")
            print_pattern("init", s)

        for step in range(1, max_iterations + 1):
            order   = rng.permutation(self.n)
            changes = 0

            for i in order:
                h_i    = float(self.W[i] @ s)
                new_i  = 1.0 if h_i > 0 else -1.0
                if new_i != s[i]:
                    s[i] = new_i
                    changes += 1

            if verbose:
                print(
                    f"\n  [async] sweep {step}  "
                    f"(energy: {self.energy(s):.2f},  "
                    f"neurons changed: {changes}/{self.n})"
                )
                print_pattern(f"t={step}", s)

            if changes == 0:
                if verbose:
                    print(f"\n  Converged after {step} sweep(s).")
                return s

        if verbose:
            print(f"\n  Did not converge in {max_iterations} sweeps.")
        return s

    def energy(self, state: np.ndarray) -> float:
        return float(-0.5 * state @ self.W @ state)

    def max_patterns(self) -> int:
        return int(0.15 * self.n)

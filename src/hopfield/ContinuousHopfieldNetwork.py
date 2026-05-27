import numpy as np
from typing import Optional


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D vector."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _lse(beta: float, x: np.ndarray) -> float:
    """
    Log-sum-exp: (1/beta) * log( sum_i exp(beta * x_i) )
    """
    m = np.max(x)
    return m + np.log(np.sum(np.exp(beta * (x - m)))) / beta


class ContinuousHopfieldNetwork():
    """
    Modern Hopfield Network with continuous states (Ramsauer et al., 2021).
    "Hopfield Networks Is All You Need."
    """

    def __init__(self, d: int, beta: float = 1.0):
        self.d = d
        self.n = d
        self.beta = beta
        self.X: Optional[np.ndarray] = None  # stored patterns, shape (d, P)

    def store_patterns(self, patterns: np.ndarray):
        """
        Store a set of continuous patterns.
        """
        if patterns.ndim == 1:
            patterns = patterns[np.newaxis, :]
        if patterns.shape[1] != self.d:
            raise ValueError(
                f"Pattern dimension {patterns.shape[1]} ≠ network dimension {self.d}."
            )
        self.X = patterns.T.astype(float)  # shape: (d, P)

    # Alias so the base-class interface still works
    def initialize_weights(self, patterns: np.ndarray):
        self.store_patterns(patterns)

    def energy(self, xi: np.ndarray) -> float:
        """
        Modern continuous Hopfield energy  (eq. 2):
            E = -lse(beta, X^T S ) + 1/2 Sᵀ * S  + (1/Beta) * log(N) + 1/2 * M^2

        Acotada: 0 ≤ E ≤ 2·M²  (Lemma A1 of paper).
        """
        if self.X is None:
            raise RuntimeError("No patterns stored. Call store_patterns() first.")
        P = self.X.shape[1]
        M = float(np.max(np.linalg.norm(self.X, axis=0)))
        interactions = self.X.T @ xi  # shape (P,): x_i^T ξ for each i
        return (
                -_lse(self.beta, interactions)
                + 0.5 * float(xi @ xi)
                + np.log(P) / self.beta
                + 0.5 * M ** 2
        )

    def _update(self, xi: np.ndarray) -> np.ndarray:
        """
        One synchronous update step:
            State_new = X · softmax(β · Xᵀ * State)

        This is equivalent to transformer attention with Q=ξ, K=X, V=X,
        scale β (= 1/√d in the transformer convention) .... may be useful later on

        Guaranteed to decrease (or leave unchanged) the energy E,
        converging globally to a stationary point ! (Theorems 1 & 2).
        """
        interactions = self.X.T @ xi  # (P,)   dot products
        p = _softmax(self.beta * interactions)  # (P,)   attention weights
        return self.X @ p  # (d,)   weighted sum of patterns

    def predict(
            self,
            initial_state: np.ndarray,
            max_iterations: int = 20,
            tol: float = 1e-10,
            verbose: bool = True,
    ) -> np.ndarray:
        if self.X is None:
            raise RuntimeError("No patterns stored. Call store_patterns() first.")

        xi = initial_state.copy().astype(float)

        if verbose:
            print(f"  [continuous] initial  (energy: {self.energy(xi):.6f})")

        for step in range(1, max_iterations + 1):
            xi_new = self._update(xi)
            delta = float(np.linalg.norm(xi_new - xi))

            if verbose:
                print(
                    f"  [continuous] step {step:3d}  "
                    f"(energy: {self.energy(xi_new):.6f},  Δξ: {delta:.2e})"
                )

            xi = xi_new

            if delta < tol:
                if verbose:
                    print(f"  Converged after {step} step(s)  (Δξ < {tol}).")
                return xi

        if verbose:
            print(f"  Did not converge within tolerance in {max_iterations} steps.")
        return xi

    def max_patterns(self) -> int:
        """
        Approximate storage capacity of the modern network.
        Cconservative bound
        with c=1.37 (Theorem 3, d≥75, K=1, p=0.001).
        -> actual capacity is higher
        """
        return int(1.37 ** ((self.d - 1) / 4))
    def __repr__(self) -> str:
        P = 0 if self.X is None else self.X.shape[1]
        return (
            f"ContinuousHopfieldNetwork("
            f"d={self.d}, beta={self.beta}, stored_patterns={P})"
        )



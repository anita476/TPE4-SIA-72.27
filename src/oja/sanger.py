import numpy as np

from src.oja.base import LinearHebbianBase


class SangerNetwork(LinearHebbianBase):

    def __init__(self, n_components=2, **kwargs):
        """n_components: cantidad k de componentes principales a estimar.
        """
        super().__init__(**kwargs)
        self.n_components = n_components

    def _initialize_weights(self, n_features):
        """Inicializa W ~ U(0,1)^(k x n) y, si corresponde, normaliza cada fila."""
        k = self.n_components
        if k > n_features:
            raise ValueError(
                "n_components no puede superar la cantidad de features")
        self.weights = self.rng.uniform(0, 1, size=(k, n_features))
        if self.normalize_initial_weights:
            self._normalize_rows()

    def _activation(self, x):
        return self.weights @ x

    def _update(self, x):
        """Aplica un paso de la regla de Sanger, vectorizado sobre las k neuronas:

            w_j = w_j + η·y_j·(x - Σ_{l≤j} y_l·w_l)
        """
        y = self._activation(x)
        eta = self._current_lr()
        lower = np.tril(np.outer(y, y))
        self.weights = self.weights + eta * (np.outer(y, x) - lower @ self.weights)
        self._t += 1

    def _normalize_rows(self):
        """Normaliza in-place cada fila de W a norma 1 (deja intactas las nulas)."""
        norms = np.linalg.norm(self.weights, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        self.weights = self.weights / norms

    def project(self, X):
        """Proyecta X sobre las k componentes; devuelve scores (n_samples x k)."""
        return X @ self.weights.T

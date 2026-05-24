import numpy as np

from src.oja.base import LinearHebbianBase

class OjaNeuron(LinearHebbianBase):
    """Neurona lineal entrenada con la regla de Oja: estima la primera componente
    principal (PC1) de los datos.
    """

    def _initialize_weights(self, n_features):
        """Inicializa w ~ U(0,1)^n y, si corresponde, lo normaliza a norma 1."""
        self.weights = self.rng.uniform(0, 1, size=n_features)
        if self.normalize_initial_weights:
            norm = np.linalg.norm(self.weights)
            if norm > 0:
                self.weights = self.weights / norm

    def _activation(self, x):
        """Salida lineal de la neurona: y = x · w."""
        return np.dot(x, self.weights)

    def _update(self, x):
        """Aplica un paso de la regla de Oja: w = w + η·y·(x - y·w)."""
        y = self._activation(x)
        eta = self._current_lr()
        self.weights = self.weights + eta * y * (x - y * self.weights)
        self._t += 1

    def project(self, X):
        """Proyecta cada fila de X sobre w; devuelve los scores del PC1."""
        return X @ self.weights

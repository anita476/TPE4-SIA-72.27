import numpy as np


class OjaNeuron:
    
    def __init__(self, learning_rate=0.05, epochs=500, shuffle=True,
                 normalize_initial_weights=True, decay=0.05, seed=None,
                 record_history=False):
        """Configura hiperparámetros y el RNG.
        learning_rate: η0, tasa de aprendizaje inicial.
        shuffle: si True, mezcla las muestras en cada época (evita sesgo por orden).
        normalize_initial_weights: si True, arranca con ‖w‖ = 1.
        decay: factor del decaimiento η_t = η0 / (1 + decay·t); 0 lo desactiva.
        record_history: si True, guarda w al final de cada época en self.history_.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.shuffle = shuffle
        self.normalize_initial_weights = normalize_initial_weights
        self.decay = decay
        self.record_history = record_history
        self.rng = np.random.default_rng(seed)
        self.weights = None
        self.history_ = None
        self._t = 0

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

    def _current_lr(self):
        """Learning rate efectivo en el paso t: η0 / (1 + decay·t)."""
        if self.decay <= 0:
            return self.learning_rate
        return self.learning_rate / (1 + self.decay * self._t)

    def _update(self, x):
        """Aplica un paso de la regla de Oja: w = w + η·y·(x - y·w).
        """
        y = self._activation(x)
        eta = self._current_lr()
        self.weights = self.weights + eta * y * (x - y * self.weights)
        self._t += 1

    def fit(self, X, init_w=None):
        """Entrena sobre X (n_samples x n_features) y devuelve w normalizado.

        Asume X estandarizado.
        Itera 'epochs' veces sobre todas las muestra. Al final normaliza w para ‖w‖ = 1.
        Si record_history=True, guarda w por época (pre-normalización final) en
        self.history_ con shape (epochs+1, n_features).
        Si init_w no es None, se usa como w_0 (ignorando normalize_initial_weights).
        """
        m, n = X.shape
        if init_w is not None:
            self.weights = np.asarray(init_w, dtype=float).copy()
        else:
            self._initialize_weights(n)
        self._t = 0

        if self.record_history:
            snapshots = [self.weights.copy()]

        for _ in range(self.epochs):
            indices = self.rng.permutation(m) if self.shuffle else np.arange(m)
            for idx in indices:
                self._update(X[idx])
            if self.record_history:
                snapshots.append(self.weights.copy())

        if self.record_history:
            self.history_ = np.array(snapshots)

        norm = np.linalg.norm(self.weights)
        if norm > 0:
            self.weights = self.weights / norm
        return self.weights

    def project(self, X):
        """Proyecta cada fila de X sobre w; devuelve los scores del PC1."""
        return X @ self.weights

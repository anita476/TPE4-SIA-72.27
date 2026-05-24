import numpy as np


class LinearHebbianBase:
    def __init__(self, learning_rate=0.05, epochs=500, shuffle=True,
                 normalize_initial_weights=True, decay=0.05, seed=None,
                 record_history=False):
        """Configura hiperparámetros y el RNG.
        learning_rate: η0, tasa de aprendizaje inicial.
        shuffle: si True, mezcla las muestras en cada época (evita sesgo por orden).
        normalize_initial_weights: si True, arranca con ‖w‖ = 1.
        decay: factor del decaimiento η_t = η0 / (1 + decay·t); 0 lo desactiva.
        record_history: si True, guarda los pesos al final de cada época en self.history_.
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

    def _current_lr(self):
        if self.decay <= 0:
            return self.learning_rate
        return self.learning_rate / (1 + self.decay * self._t)

    
    def _initialize_weights(self, n_features):
        """Inicializa self.weights con la geometría propia de la regla."""
        raise NotImplementedError

    def _update(self, x):
        """Aplica un paso de la regla de aprendizaje sobre la muestra x."""
        raise NotImplementedError


    def fit(self, X, init_w=None):
        """Asume X estandarizado. Itera 'epochs' veces sobre todas las muestras.
        Si record_history=True, guarda los pesos por época en self.history_. Si init_w no es None, se usa como w_0.
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

        return self.weights

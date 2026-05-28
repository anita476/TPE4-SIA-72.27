import numpy as np

class Kohonen:
    def __init__(self, k, input_dim, eta_0=0.1, radius_0=None,
                 similarity="euclidean", weight_init="random",
                 n_iter=None, seed=None):
        """
        Parámetros
        ----------
        k : int
            Lado de la grilla (la grilla es k x k).
        input_dim : int
            Dimensión n de los vectores de entrada.
        eta_0 : float
            Tasa de aprendizaje inicial η(0), 0 < η(0) < 1.
        radius_0 : float or None
            Radio inicial del vecindario R(0). Si es None, usar k (tamaño de la red).
        similarity : {"euclidean", "exponential"}
            Medida de similitud para elegir la neurona ganadora.
        weight_init : {"random", "samples"}
            "random": pesos uniformes aleatorios.
            "samples": pesos inicializados con muestras del conjunto de entrenamiento
                       (evita unidades muertas).
        seed : int or None
            Para reproducibilidad.
        """
        self.k = k
        self.input_dim = input_dim
        self.eta_0 = eta_0
        self.radius_0 = radius_0
        if self.radius_0 is None:
            self.radius_0 = float(self.k)
        self.similarity = similarity
        self.weight_init = weight_init
        self.n_iter = n_iter  # None → fit() uses 500 * input_dim
        self.rng = np.random.default_rng(seed)
        self.weights = None

    # -----------------------------------------------------------------
    # Inicialización de pesos
    # -----------------------------------------------------------------
    def _init_weights(self, X):
        shape = (self.k, self.k, self.input_dim)

        if self.weight_init == "random":
            self.weights = self.rng.uniform(X.min(), X.max(), size=shape)
        elif self.weight_init == "samples":
            # elegimos k*k filas de X al azar (con reemplazo porque k*k puede ser > P (cantidad de entradas = paises))
            n_samples = self.k*self.k
            if n_samples <= X.shape[0]:
                # sin reemplazo: cada peso es un pais distinto
                indices = self.rng.choice(X.shape[0], size=n_samples, replace=False)
            else:
                # con reemplazo: inevitable
                indices = self.rng.choice(X.shape[0], size=n_samples, replace=True)
            self.weights = X[indices].reshape(shape)
        else:
            raise ValueError(f"Error: unknown weight_init: {self.weight_init}")

    # -----------------------------------------------------------------
    # Similitud y neurona ganadora
    # -----------------------------------------------------------------
    def _winner(self, x):
        """
        Dado un vector de entrada x (shape (input_dim,)), devuelve las coordenadas
        (i, j) de la neurona ganadora en la grilla.

        - "euclidean": argmin de ||x - W_ij||
        - "exponential": argmax de exp(-||x - W_ij||^2)
        """
        diff = self.weights - x

        if self.similarity == "euclidean":
            scores = np.linalg.norm(diff, axis=2)

        elif self.similarity == "exponential":
            # ‖x - W‖² y luego score = -exp(-d²)
            dist2 = np.sum(diff * diff, axis=-1)
            scores = -np.exp(-dist2)
        else:
            raise ValueError(
                f"Unknown similarity measure: '{self.similarity}'. "
                f"Expected 'euclidean' or 'exponential'."
            )

        flat_idx = np.argmin(scores) if self.similarity == "euclidean" else np.argmax(scores)
        return np.unravel_index(flat_idx, scores.shape)

    # -----------------------------------------------------------------
    # Vecindario
    # -----------------------------------------------------------------
    def _neighbors(self, winner_idx, radius):
        """
        Devuelve una máscara booleana (k, k) con True en las neuronas que están
        dentro del radio `radius` de la ganadora.

        Definición (apunte):
            N_k(t) = { neu / ||neu - neu_k|| < R(t) }
        donde la distancia es entre coordenadas de la grilla.
        """
        # ii[a,b], jj[a,b]) = (a, b)
        ii, jj = np.indices((self.k, self.k))
        i_win, j_win = winner_idx
        distances = np.sqrt((ii - i_win)**2 + (jj-j_win)**2)
        return distances < radius

    # -----------------------------------------------------------------
    # Actualización de pesos (regla de Kohonen)
    # -----------------------------------------------------------------
    def _update(self, x, winner_idx, eta, radius):
        """
        Aplica la regla:
            W_j(t+1) = W_j(t) + η(t) * (x - W_j(t))   si j ∈ N_k(t)
            W_j(t+1) = W_j(t)                         si j ∉ N_k(t)
        """
        mask = self._neighbors(winner_idx, radius)
        update = eta * (x - self.weights)
        # multiplicamos con mascara (solo aplica actualizacion donde en mask es True)
        self.weights += update * mask[:,:,None]

    # -----------------------------------------------------------------
    # Schedules de η(t) y R(t)
    # -----------------------------------------------------------------
    def _eta(self, t, t_max):
        # η(n) = η₀ · exp(-n/τ₂),  τ₂ = 1000,  floor = 0.01
        return max(0.01, self.eta_0 * np.exp(-t / 1000))
        

    def _radius(self, t, t_max):
        # σ(n) = σ₀ · exp(-n/τ₁),  τ₁ = 1000/log(σ₀),  floor = 1
        tau1 = 1000 / np.log(self.radius_0) if self.radius_0 > 1 else 1000
        r = self.radius_0 * np.exp(-t / tau1)
        return max(1.0, r)

    # -----------------------------------------------------------------
    # Entrenamiento
    # -----------------------------------------------------------------
    def fit(self, X, n_iter=None, n_snapshots=20):
        """
        Entrena la red.

        Parámetros
        ----------
        X : np.ndarray, shape (P, input_dim)
            Datos de entrenamiento ya estandarizados.
        n_iter : int or None
            Cantidad total de iteraciones. Se sugiere ~ 500 * n
            (con n = input_dim) si no se pasa nada.
        """
        if n_iter is None:
            n_iter = self.n_iter if self.n_iter is not None else 500 * self.input_dim

        P = X.shape[0]
        self._init_weights(X)

        self.weight_history = []  # list of (k*k, input_dim) arrays
        self.qe_history = []     # list of (t, qe)
        self.te_history = []     # list of (t, te)
        self.delta_history = []  # list of mean per-neuron ‖W(t+1)−W(t)‖, one per iteration
        snapshot_interval = max(1, n_iter // n_snapshots)

        for t in range(n_iter):
            eta = self._eta(t, n_iter)
            radius = self._radius(t, n_iter)

            if t % snapshot_interval == 0:
                self.weight_history.append(
                    self.weights.reshape(-1, self.input_dim).copy()
                )
                qe = self.quantization_error(X)
                self.qe_history.append((t, qe))
                self.te_history.append((t, self.topographic_error(X)))
                print(f"    t={t:6d}/{n_iter}  η={eta:.4f}  R={radius:.4f}  QE={qe:.4f}")

            # Reshuffle at the start of each epoch
            if t % P == 0:
                perm = self.rng.permutation(P)
            x = X[perm[t % P]]

            prev_W = self.weights.copy()
            winner = self._winner(x)
            self._update(x, winner, eta, radius)
            self.delta_history.append(
                np.linalg.norm(self.weights - prev_W, axis=-1).mean()
            )

        # Final snapshot
        self.weight_history.append(self.weights.reshape(-1, self.input_dim).copy())
        self.qe_history.append((n_iter, self.quantization_error(X)))
        self.te_history.append((n_iter, self.topographic_error(X)))
                
                
                
    # -----------------------------------------------------------------
    # Predicción / consulta
    # -----------------------------------------------------------------
    def predict(self, X):
        """
        Para cada fila de X devuelve las coordenadas (i, j) de la neurona ganadora.
        Útil para ver qué países cayeron en qué celda.
        """
        return np.array([self._winner(x) for x in X])

    # -----------------------------------------------------------------
    # Utilidades para los gráficos
    # -----------------------------------------------------------------
    def u_matrix(self):
        """
        U-matrix: para cada neurona, distancia promedio a sus vecinas en el
        espacio de los PESOS (no en la grilla).

        Devuelve un array (k, k) con esos promedios.
        """
        u = np.zeros((self.k, self.k))
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for i in range(self.k):
            for j in range(self.k):
                distances = []
                for di, dj in offsets:
                    ni, nj = i + di, j + dj
                    # Verificamos que el vecino este en la grilla
                    if 0 <= ni < self.k and 0 <= nj < self.k:
                        d = np.linalg.norm(self.weights[i,j]-self.weights[ni,nj])
                        distances.append(d)
                u[i,j] = np.mean(distances)
        return u

    def quantization_error(self, X):
        """
        Average distance between each input and its winning neuron's weight vector.
        Lower = neurons represent their assigned samples more accurately.
        """
        total = 0
        for x in X:
            i, j = self._winner(x)
            total += np.linalg.norm(x - self.weights[i, j])
        return total / len(X)

    def topographic_error(self, X):
        """
        Fraction of samples where the BMU and 2nd BMU are not adjacent on the
        grid (distance > 1). A low value means the map preserves
        the input topology well.
        """
        errors = 0
        for x in X:
            diff = self.weights - x
            if self.similarity == "euclidean":
                scores = np.linalg.norm(diff, axis=2)
            else:
                scores = -np.exp(-np.sum(diff * diff, axis=-1))
            flat_sorted = np.argsort(scores.ravel())
            bmu1 = np.unravel_index(flat_sorted[0], scores.shape)
            bmu2 = np.unravel_index(flat_sorted[1], scores.shape)
            if max(abs(bmu1[0] - bmu2[0]), abs(bmu1[1] - bmu2[1])) > 1:
                errors += 1
        return errors / len(X)

    def activations_per_neuron(self, X):
        """
        Cuenta cuántos registros de X cayeron en cada neurona.
        Devuelve un array (k, k) de enteros.
        """
        counts = np.zeros((self.k, self.k), dtype=int)
        for x in X:
            i, j = self._winner(x)
            counts[i, j] += 1
        return counts
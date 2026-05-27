# TPE4-SIA-72.27

Aprendizaje no supervisado: Kohonen, Oja/Sanger, Hopfield y PCA. Cuarto trabajo práctico para Sistemas de Inteligencia Artificial (ITBA).

### Integrantes

- Camila Lee
- Federico Etchegorry
- Matías Leporini Kogan
- Ana Negre

---

## Algoritmos implementados

| Módulo               | Descripción                                                    | Ubicación                                   |
| -------------------- | -------------------------------------------------------------- | ------------------------------------------- |
| **Kohonen (SOM)**    | Mapa autoorganizado `k×k` sobre datos estandarizados de Europa | `src/kohonen.py`                            |
| **Oja**              | Neurona lineal con regla de Oja; estima PC1                    | `src/oja/oja_neuron.py`                     |
| **Sanger**           | Red de neuronas con regla de Sanger (PCs en cascada)           | `src/oja/sanger.py`                         |
| **Hopfield clásico** | Red binaria ±1, actualización síncrona o asíncrona             | `src/hopfield/HopfieldNetwork.py`           |
| **Hopfield moderno** | Red continua (softmax), β configurable                         | `src/hopfield/ContinuousHopfieldNetwork.py` |
| **PCA**              | Biplot 2D con `sklearn` (referencia / preentrega)              | `PCA/pca_plot.py`, `PCA/pca.ipynb`          |

---

## Entry points

### Scripts (`scripts/`)

| Script                                | Propósito                                                                            |
| ------------------------------------- | ------------------------------------------------------------------------------------ |
| `scripts/kohonen_analysis.py`         | Entrena Kohonen con un JSON y genera el paquete completo de gráficos + `summary.txt` |
| `scripts/europe_map_plot.py`          | Solo mapa geográfico de clusters (`europe_geographic.png`)                           |
| `scripts/oja_experiments.py`          | Batería de experimentos Oja/Sanger sobre `data/europe.csv`                           |
| `scripts/hopfield_analysis.py`        | Batería de 9 figuras de análisis Hopfield (sync o async)                             |
| `scripts/hopfield_sync_vs_async.py`   | Comparación energía, pasos, trayectorias y tiempo sync vs async                      |
| `scripts/hopfield_noise_tolerance.py` | Curvas de tolerancia al ruido por patrón                                             |
| `scripts/plot_hopfield.py`            | Convergencia paso a paso de una consulta (energía y campos locales)                  |
| `scripts/compare_letters.py`          | Matriz de producto interno normalizado entre patrones                                |
| `scripts/plot_letters.py`             | Visualización interactiva del alfabeto 5×5 (`matplotlib`)                            |

### Drivers en `src/`

| Script                        | Propósito                                                             |
| ----------------------------- | --------------------------------------------------------------------- |
| `src/hopfield/hopfield.py`    | Recuperación puntual: patrones + consulta + ruido (clásico o moderno) |
| `src/oja/compare_with_pca.py` | Compara pesos de Oja contra PC1 de `sklearn` (consola)                |

### PCA (`PCA/`)

| Archivo           | Propósito                                        |
| ----------------- | ------------------------------------------------ |
| `PCA/main.py`     | CLI que guarda `pca.png` desde `data/europe.csv` |
| `PCA/pca_plot.py` | Función `plot_pca()` usada por `main.py`         |
| `PCA/pca.ipynb`   | Notebook exploratorio de PCA                     |

Ejecutar los comandos desde la **raíz del repositorio** salvo que se indique lo contrario (p. ej. `plot_letters.py`).

---

## Estructura del proyecto

```text
.
├── configs/                 # JSON de hiperparámetros Kohonen (kohonen_*.json)
├── data/                    # Europa (CSV), letras 5×5, patrones y consultas Hopfield
├── PCA/                     # PCA de referencia (main.py, pca_plot.py, notebook, PDF, pca.html)
├── results/                 # Salidas generadas (plots, hopfield, oja); no versionado por defecto
├── scripts/                 # Experimentos y visualizaciones
├── src/                     # Implementaciones de modelos
│   ├── kohonen.py
│   ├── hopfield/
│   └── oja/
└── utils/                   # Carga de letras, preprocesamiento, display Hopfield
```

---

## Requirements

Crear y activar un entorno virtual (recomendado):

```bash
python -m venv .venv
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar dependencias:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

| Paquete                 | Uso principal                                        |
| ----------------------- | ---------------------------------------------------- |
| `numpy`, `pandas`       | Modelos y datos tabulares                            |
| `matplotlib`            | Gráficos en scripts y Kohonen                        |
| `seaborn`               | Heatmaps de letras (`utils/letters.py`)              |
| `scikit-learn`          | PCA de referencia, métricas Oja                      |
| `geopandas`             | Mapa geográfico de Europa (Kohonen)                  |

**Notas:**

- Los mapas de Europa descargan geometrías de Natural Earth vía `geopandas`.
- `kohonen_analysis.py` omite `europe_geographic.png` si `geopandas` no está instalado, pero genera el resto.
- `europe_map_plot.py` **requiere** `geopandas` (import obligatorio al inicio del script).

---

## Formato de datos

### Europa (`data/europe.csv`)

CSV con **28** países europeos (más fila de encabezado). La columna `Country` es identificador; el resto son features numéricas. Kohonen y Oja las estandarizan vía `load_europe`; `PCA/pca_plot.py` usa `StandardScaler` por su cuenta.

| Columna        | Descripción                         |
| -------------- | ----------------------------------- |
| `Country`      | Nombre del país (no se estandariza) |
| `Area`         | Superficie                          |
| `GDP`          | PIB per cápita                      |
| `Inflation`    | Inflación                           |
| `Life.expect`  | Esperanza de vida                   |
| `Military`     | Gasto militar (% PIB)               |
| `Pop.growth`   | Crecimiento poblacional             |
| `Unemployment` | Desempleo                           |

Carga unificada: `utils.preprocessing.load_europe(path)` → `(countries, X_estandarizado, feature_columns)`.

### Letras 5×5 (`data/letters.txt`, `data/patterns*.txt`, `data/query*.txt`)

- Cada fila tiene **5 caracteres**: `*` = píxel activo (+1), espacio = inactivo (−1).
- En `letters.txt`, el separador `=` marca el fin de cada letra (A, B, C, … en orden).
- En archivos con nombre (`patterns.txt`, `10_letters_good.txt`, etc.), la línea `=<nombre>` introduce un patrón (p. ej. `=W`).
- Los archivos `query_*.txt` y `query.txt` contienen **un solo** patrón 5×5 sin separador de nombre.

Los patrones se convierten internamente a vectores ±1 de dimensión 25.

### Archivos en `data/`

| Archivo                                                                 | Contenido                                                             |
| ----------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `europe.csv`                                                            | Dataset principal para Kohonen y Oja                                  |
| `letters.txt`                                                           | Alfabeto completo A–Z en 5×5                                          |
| `patterns.txt`                                                          | Patrones almacenados para Hopfield (I, R, W, X en el set por defecto) |
| `patterns_worst.txt`                                                    | Conjunto alternativo: patrones menos ortogonales                      |
| `10_letters_good.txt`                                                   | Subconjunto de 10 letras con buena ortogonalidad                      |
| `query.txt`, `query_I.txt`, `query_R.txt`, `query_W.txt`, `query_X.txt` | Consultas individuales para recuperación                              |

---

## Configuraciones Kohonen (`configs/`)

Cada `kohonen_*.json` define un experimento. El nombre del archivo (sin extensión) es el subdirectorio de salida bajo `results/plots/`.

| Campo         | Tipo          | Descripción                                       |
| ------------- | ------------- | ------------------------------------------------- |
| `k`           | int           | Lado de la grilla (`k×k` neuronas)                |
| `eta_0`       | float         | Tasa de aprendizaje inicial η(0)                  |
| `radius_0`    | float \| null | Radio inicial del vecindario; `null` → se usa `k` |
| `n_iter`      | int \| null   | Iteraciones; `null` → `500 × nº de features` (7 → 3500 en Europa) |
| `weight_init` | str           | `random` o `samples`                              |
| `similarity`  | str           | Neurona ganadora: `euclidean` o `exponential`     |
| `seed`        | int           | Semilla para reproducibilidad                     |

| Archivo                        | `k` | `similarity`  | Notas                                                        |
| ------------------------------ | --- | ------------- | ------------------------------------------------------------ |
| `kohonen_2x2.json`             | 2   | `euclidean`   | Grilla mínima                                                |
| `kohonen_3x3.json`             | 3   | `euclidean`   |                                                              |
| `kohonen_4x4.json`             | 4   | `euclidean`   |                                                              |
| `kohonen_5x5_default.json`     | 5   | `euclidean`   | Configuración principal del informe                          |
| `kohonen_5x5_exponential.json` | 5   | `exponential` | Misma grilla 5×5, neurona ganadora por score exponencial     |
| `kohonen_10x10.json`           | 28  | `euclidean`   | Nombre histórico; grilla 28×28 (≈ un país por neurona)       |

Nuevos experimentos: copiar un JSON existente en `configs/` y pasar la ruta a `--config`; no hace falta tocar código.

---

## Comandos rápidos

```bash
# Kohonen (análisis completo)
python scripts/kohonen_analysis.py --config configs/kohonen_5x5_default.json

# Hopfield — batería de análisis
python scripts/hopfield_analysis.py --mode sync
python scripts/hopfield_analysis.py --mode async

# Hopfield — una recuperación
python src/hopfield/hopfield.py data/patterns.txt data/query_W.txt --noise 0.2 --seed 42 --mode sync

# Oja / Sanger
python scripts/oja_experiments.py
python -m src.oja.compare_with_pca

# PCA de referencia
python PCA/main.py --data data/europe.csv --out results/plots --seed 1
```

---

## Flujos de trabajo

### 1. Kohonen sobre Europa

```bash
python scripts/kohonen_analysis.py --config configs/kohonen_5x5_default.json
```

Salida en `results/plots/<nombre_config>/`:

| Archivo                                    | Contenido                                   |
| ------------------------------------------ | ------------------------------------------- |
| `schedules.png`                            | Evolución de η(t) y R(t)                    |
| `countries.png`                            | Países por neurona y conteo de activaciones |
| `umatrix_countries.png`                    | U-matrix con países superpuestos            |
| `variables.png`                            | Promedio por variable y neurona             |
| `europe_geographic.png`                    | Clusters en mapa de Europa                  |
| `cohesion_table.png`, `cohesion_table.csv` | Dispersión intra-cluster por variable       |
| `cluster_profiles.png`                     | Perfil promedio de cada cluster             |
| `summary.txt`                              | Resumen textual del experimento             |

Solo mapa geográfico:

```bash
python scripts/europe_map_plot.py --config configs/kohonen_5x5_default.json
```

### 2. Hopfield — análisis y comparaciones

Batería completa (patrones en `data/patterns.txt`, consulta de referencia: tercer patrón almacenado, típicamente `W`):

```bash
python scripts/hopfield_analysis.py --mode sync
```

Salida en `results/hopfield/<mode>/`:

| #   | Archivo                                              |
| --- | ---------------------------------------------------- |
| 1   | `1_stored_patterns.png`                              |
| 2   | `2_recovery_grid.png`                                |
| 3   | `3_recovery_steps.png`                               |
| 4   | `4_spurious_state.png`                               |
| 5   | `5_energy_convergence.png`                           |
| 6   | `6_noise_robustness.png`                             |
| 7   | `7_overlap_matrix.png`                               |
| 8   | `8_basin_by_pattern.png`                             |
| 9   | `9_capacity_experiment.png` (usa `data/letters.txt`) |

Comparación sync vs async (sin argumentos CLI; rutas fijas en el script):

```bash
python scripts/hopfield_sync_vs_async.py
```

Salida en `results/hopfield/comparison/`:

- `1_energy_overlay.png`
- `2_steps_distribution.png`
- `3_trajectory_low_noise.png`, `3_trajectory_mid_noise.png`
- `4_wallclock_compare.png`

Tolerancia al ruido:

```bash
python scripts/hopfield_noise_tolerance.py data/patterns.txt --noise-steps 20 --trials 30 --out-dir results/hopfield/noise --mode sync
```

Genera `stacked_areas.png`, `heatmap.png`, `retrieval_curves.png`.

Ortogonalidad de patrones:

```bash
python scripts/compare_letters.py data/patterns.txt --out-dir results/letters
```

### 3. Oja y comparación con PCA

```bash
python scripts/oja_experiments.py
```

Hiperparámetros en `main()` del script (no expuestos por CLI):

| Parámetro                 | Valor           |
| ------------------------- | --------------- |
| `LR`                      | `0.1`           |
| `EPOCHS`                  | `10000`         |
| `DECAY`                   | `0.01`          |
| `N_SEEDS`                 | `5`             |
| `HEBB_LR` / `HEBB_EPOCHS` | `0.001` / `200` |

Salida en `results/oja/`:

- `convergence_weights.png`
- `error_to_reference.png`
- `lr_decay_heatmap.png`
- `loadings.png`
- `scores_scatter.png`
- `unnormalized_init.png`
- `init_strategy.png`
- `shuffle.png`
- `explained_variance_spectrum.png`
- `explained_variance_spectrum_sanger.png`
- `variance_captured.png`
- `pc1_ranking_biplot.png`
- `hebb_vs_oja.png`

Comparación puntual Oja vs sklearn:

```bash
python -m src.oja.compare_with_pca
```

Imprime varianza explicada por PC1, similitud coseno, tabla de coeficientes y ranking de países. Usa `OjaNeuron(seed=1)` con defaults de `LinearHebbianBase` (`learning_rate=0.05`, `epochs=500`, `decay=0.05`).

### 4. PCA (preentrega / referencia)

```bash
python PCA/main.py --data data/europe.csv --out results/plots --seed 1
```

Guarda `pca.png` en el directorio `--out`. Alternativa interactiva: `PCA/pca.ipynb` o exportado estático `PCA/pca.html`.

---

## Referencia detallada de scripts

### `scripts/kohonen_analysis.py`

```bash
python scripts/kohonen_analysis.py --config configs/kohonen_5x5_default.json
```

| Argumento  | Tipo | Descripción                                  |
| ---------- | ---- | -------------------------------------------- |
| `--config` | str  | **Requerido.** Ruta al JSON de configuración |

### `scripts/europe_map_plot.py`

```bash
python scripts/europe_map_plot.py --config configs/kohonen_5x5_default.json
```

| Argumento  | Tipo | Descripción                                         |
| ---------- | ---- | --------------------------------------------------- |
| `--config` | str  | **Requerido.** Mismo JSON que para entrenar Kohonen |

### `scripts/hopfield_analysis.py`

```bash
python scripts/hopfield_analysis.py --mode sync
```

| Argumento | Tipo | Default | Descripción                                                |
| --------- | ---- | ------- | ---------------------------------------------------------- |
| `--mode`  | str  | `sync`  | `sync` o `async`; define subcarpeta en `results/hopfield/` |

### `src/hopfield/hopfield.py`

```bash
python src/hopfield/hopfield.py <patterns_file> <query_file> [opciones]
```

| Argumento       | Tipo  | Default   | Descripción                                           |
| --------------- | ----- | --------- | ----------------------------------------------------- |
| `patterns_file` | str   | —         | Archivo con patrones almacenados 5×5                  |
| `query_file`    | str   | —         | Archivo con un patrón de consulta 5×5                 |
| `--max-iter`    | int   | `20`      | Máximo de iteraciones / sweeps                        |
| `--quiet`       | flag  | off       | Solo resultado final                                  |
| `--noise`       | float | `0.2`     | Fracción de píxeles a invertir en la consulta         |
| `--seed`        | int   | —         | Semilla para ruido y orden async                      |
| `--mode`        | str   | `sync`    | `sync` o `async` (solo red clásica)                   |
| `--analyze`     | flag  | off       | Imprime análisis de ortogonalidad de grupos de letras |
| `--type`        | str   | `classic` | `classic` (binaria) o `modern` (continua)             |
| `--beta`        | float | `4.0`     | Temperatura inversa β (solo red moderna)              |

Ejemplo con red moderna:

```bash
python src/hopfield/hopfield.py data/patterns.txt data/query_W.txt --type modern --beta 4.0 --noise 0.2 --seed 42
```

### `scripts/plot_hopfield.py`

```bash
python scripts/plot_hopfield.py data/patterns.txt data/query_W.txt --noise 0.2 --seed 42 --max-iter 20 --mode sync --out results/hopfield/convergence.png
```

| Argumento       | Tipo  | Default           | Descripción                                              |
| --------------- | ----- | ----------------- | -------------------------------------------------------- |
| `patterns_file` | str   | —                 | Patrones almacenados                                     |
| `query_file`    | str   | —                 | Consulta                                                 |
| `--noise`       | float | `0.2`             | Fracción de ruido                                        |
| `--seed`        | int   | `42`              | Semilla                                                  |
| `--max-iter`    | int   | `20`              | Máximo de iteraciones                                    |
| `--out`         | str   | `convergence.png` | Ruta del PNG                                             |
| `--mode`        | str   | `sync`            | `sync` o `async`                                         |
| `--no-fields`   | flag  | off               | No generar figura de campos locales (p. ej. `convergence_fields.png` si `--out convergence.png`) |

### `scripts/hopfield_noise_tolerance.py`

```bash
python scripts/hopfield_noise_tolerance.py data/patterns.txt --noise-steps 20 --trials 30 --out-dir results/hopfield/noise
```

| Argumento       | Tipo  | Default   | Descripción                         |
| --------------- | ----- | --------- | ----------------------------------- |
| `patterns_file` | str   | —         | Patrones almacenados                |
| `--noise-steps` | int   | `20`      | Divisiones entre 0% y 100% de ruido |
| `--trials`      | int   | `30`      | Ensayos por nivel de ruido          |
| `--max-iter`    | int   | `20`      | Máximo de iteraciones               |
| `--seed`        | int   | `42`      | Semilla base                        |
| `--out-dir`     | str   | `.`       | Directorio de salida                |
| `--mode`        | str   | `sync`    | `sync` o `async` (clásica)          |
| `--type`        | str   | `classic` | `classic` o `modern`                |
| `--beta`        | float | `4.0`     | β para red moderna                  |

### `scripts/compare_letters.py`

```bash
python scripts/compare_letters.py data/patterns.txt --out-dir results/letters
```

| Argumento       | Tipo | Default | Descripción                   |
| --------------- | ---- | ------- | ----------------------------- |
| `patterns_file` | str  | —       | Archivo de patrones           |
| `--out-dir`     | str  | `.`     | Directorio para `heatmap.png` |

### `scripts/plot_letters.py`

```bash
cd scripts && python plot_letters.py
```

No recibe argumentos. Carga `../data/letters.txt` con rutas relativas al directorio `scripts/`. Desde la raíz del repo, ajustar el path en el script o ejecutar como arriba.

### `scripts/hopfield_sync_vs_async.py`

```bash
python scripts/hopfield_sync_vs_async.py
```

Sin parámetros CLI. Usa `data/patterns.txt` y escribe en `results/hopfield/comparison/`.

### `scripts/oja_experiments.py`

```bash
python scripts/oja_experiments.py
```

Sin parámetros CLI; datos fijos en `data/europe.csv`, salida en `results/oja/`.

### `PCA/main.py`

```bash
python PCA/main.py --data data/europe.csv --out results/plots --seed 1
```

| Argumento | Tipo | Default           | Descripción          |
| --------- | ---- | ----------------- | -------------------- |
| `--data`  | str  | `data/europe.csv` | CSV de entrada       |
| `--out`   | str  | `results/plots`   | Directorio de salida |
| `--seed`  | int  | `1`               | Semilla (`random_state`) para PCA 2D |

---

## Módulos en `src/` y `utils/`

### `src/kohonen.py` — clase `Kohonen`

- Grilla `k×k`, vecindario por radio en la grilla (máscara circular; actualización uniforme dentro del vecindario).
- Schedules: η(t) = η₀/(t+1), R(t) = max(1, R₀/(1+t)).
- `fit(X, n_iter=None)` entrena sobre `(n_samples, input_dim)`; si `n_iter` es `None`, usa `500 × input_dim`.
- `predict(X)` devuelve un array con la BMU `(i, j)` de cada fila de `X`.
- Utilidades: `u_matrix()`, `quantization_error(X)`, `activations_per_neuron(X)`.

### `src/hopfield/`

| Clase                       | Rol                                                                                  |
| --------------------------- | ------------------------------------------------------------------------------------ |
| `HopfieldNetwork`           | Pesos de Hebb ±1, `predict(..., mode='sync'\|'async')`, energía de Hopfield          |
| `ContinuousHopfieldNetwork` | Actualización continua tipo Ramsauer et al.; `store_patterns`, `predict` con softmax |

### `src/oja/`

| Módulo                | Rol                                     |
| --------------------- | --------------------------------------- |
| `base.py`             | Base Hebbiana con LR y decay            |
| `oja_neuron.py`       | Regla de Oja → PC1                      |
| `sanger.py`           | Regla de Sanger → múltiples componentes |
| `compare_with_pca.py` | Script de comparación con sklearn       |

### `utils/`

| Módulo                | Funciones principales                                                                                           |
| --------------------- | --------------------------------------------------------------------------------------------------------------- |
| `preprocessing.py`    | `standardize`, `load_europe`                                                                                    |
| `letters.py`          | `load_letters`, `load_patterns`, `load_query`, `add_noise`, `group_analysis`, `best_match`, `classify_recovery` |
| `display_hopfield.py` | `print_pattern`, `print_separator`, `pattern_to_str` (salida en consola)                                        |

---

## Directorio `results/` (generado)

No es obligatorio versionar las salidas. Estructura típica tras correr los flujos:

```text
results/
├── plots/
│   └── <nombre_config>/     # Kohonen
├── hopfield/
│   ├── sync/                # hopfield_analysis.py
│   ├── async/
│   ├── comparison/          # hopfield_sync_vs_async.py
│   └── noise/               # hopfield_noise_tolerance.py
├── letters/                 # compare_letters.py
└── oja/                     # oja_experiments.py
```

---

## Notas operativas

- **Raíz del repo:** casi todos los paths (`data/...`, `configs/...`, `results/...`) asumen ejecución desde la raíz. Excepción habitual: `plot_letters.py` (ver arriba).
- **Estandarización:** Kohonen, Oja y `load_europe` estandarizan columnas numéricas; PCA en `PCA/pca_plot.py` usa `StandardScaler` de sklearn.
- **Codificación:** `src/hopfield/hopfield.py` reconfigura `stdout` a UTF-8 para caracteres de separadores en consola.
- **Imports de Hopfield:** `hopfield.py` agrega la raíz del repo a `sys.path` y importa `HopfieldNetwork` / `ContinuousHopfieldNetwork` desde el mismo directorio; ejecutar como `python src/hopfield/hopfield.py ...` desde la raíz.
- **Reproducibilidad:** usar `seed` en JSON de Kohonen o `--seed` en scripts Hopfield cuando esté disponible.
- **Hopfield moderno:** `--mode` se ignora (actualización softmax síncrona); `--beta` controla la temperatura inversa (default `4.0`).
- **`--analyze`:** imprime ortogonalidad de combinaciones de letras; puede tardar (combinatoria sobre `letters.txt`).
- **Capacidad:** `hopfield_analysis.py` prueba subconjuntos crecientes de letras desde `data/letters.txt` (hasta 26 patrones).
- **Consulta por defecto en análisis:** `hopfield_analysis.py` y `hopfield_sync_vs_async.py` usan el **tercer** patrón de `patterns.txt` (orden de archivo: `I`, `R`, `W`, `X` → consulta `W`).

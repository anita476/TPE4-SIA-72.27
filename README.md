# TPE4 SIA - Aprendizaje No Supervisado

Trabajo Práctico para Sistemas de Inteligencia Artificial. Aprendizaje no supervisado: Kohonen, Oja/Sanger, Hopfield y PCA


### Integrantes 
* Camila Lee
* Federico Etchegorry 
* Matías Leporini Kogan
* Ana Negre
## Estructura del Proyecto

```text
.
├── configs/                 # Configuraciones JSON para experimentos de Kohonen
├── data/                    # Datasets y patrones de letras 5x5
├── PCA/                     # Script/notebook de PCA para preentrega
├── results/                 # Figuras y salidas ya generadas
├── scripts/                 # Experimentos y visualizaciones/plots
├── src/                     # Implementaciones principales de los modelos
└── utils/                   # Auxiliares, preprocesamiento
```

## Instalación y Entorno

Crear y activar un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instalar dependencias:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Nota: los gráficos geográficos de Europa usan `geopandas` y datos de Natural Earth. 

## Comandos Rápidos

Ejecutar análisis completo de Kohonen:

```bash
python3 scripts/kohonen_analysis.py --config configs/kohonen_5x5_default.json
```

Ejecutar análisis completo de Hopfield clásico:

```bash
python3 scripts/hopfield_analysis.py --mode sync
python3 scripts/hopfield_analysis.py --mode async
```

Ejecutar una recuperación puntual con Hopfield:

```bash
python3 src/hopfield/hopfield.py data/patterns.txt data/query_W.txt --noise 0.2 --seed 42 --mode sync
```

Ejecutar experimentos de Oja/Sanger:

```bash
python3 scripts/oja_experiments.py
```

Comparar Oja contra PCA de `scikit-learn`:

```bash
python3 -m src.oja.compare_with_pca
```

## Datos

- `data/europe.csv`: dataset de países europeos. 
- `data/letters.txt`: alfabeto en patrones 5x5, codificado con `*` para pixeles activos y espacios para inactivos.
- `data/patterns.txt`: patrones almacenados para Hopfield, con separadores `=<nombre>`.
- `data/patterns_worst.txt`: conjunto alternativo de los patrones menos ortogonales.
- `data/10_letters_good.txt`: conjunto de 10 letras seleccionado para experimentos (mejor conjunto de 10).
- `data/query*.txt`: patrones de consulta individuales para Hopfield.

Los patrones de Hopfield se convierten internamente a valores `+1` y `-1`.

## Configuraciones de Kohonen

Los archivos `configs/kohonen_*.json` controlan los experimentos de SOM:

- `k`: lado de la grilla. La red tiene `k x k` neuronas.
- `eta_0`: tasa de aprendizaje inicial.
- `radius_0`: radio inicial del vecindario. Si es `null`, se usa `k`.
- `n_iter`: cantidad de iteraciones. Si es `null`, se usa `500 * input_dim`.
- `weight_init`: inicialización de pesos. Valores: `random` o `samples`.
- `similarity`: criterio de neurona ganadora. Valores: `euclidean` o `exponential`.
- `seed`: semilla para reproducibilidad.

Ejemplos disponibles: `kohonen_2x2.json`, `kohonen_3x3.json`, `kohonen_4x4.json`, `kohonen_5x5_default.json`, `kohonen_5x5_exponential.json`, `kohonen_10x10.json`.

## Scripts Ejecutables

### `scripts/kohonen_analysis.py`

Entrena una red de Kohonen sobre `data/europe.csv` y genera gráficos de análisis.

Uso:

```bash
python3 scripts/kohonen_analysis.py --config configs/kohonen_5x5_default.json
```

Parámetros:

- `--config`: ruta al JSON de configuración. Obligatorio.

Salida en `results/plots/<nombre_config>/`:

- `schedules.png`: evolución de `eta(t)` y `R(t)`.
- `countries.png`: países asignados a cada neurona y cantidad de activaciones.
- `umatrix_countries.png`: U-matrix con países superpuestos.
- `variables.png`: heatmaps de valor promedio por variable.
- `europe_geographic.png`: clusters proyectados en mapa geográfico.
- `cohesion_table.png` y `cohesion_table.csv`: dispersión por variable dentro de cada cluster.
- `cluster_profiles.png`: perfil promedio de cada cluster.
- `summary.txt`: resumen textual del experimento.

### `scripts/europe_map_plot.py`

Entrena Kohonen con una configuración y genera únicamente el mapa geográfico de Europa.

Uso:

```bash
python3 scripts/europe_map_plot.py --config configs/kohonen_5x5_default.json
```

Parámetros:

- `--config`: ruta al JSON usado para entrenar.

Salida:

- `results/plots/<nombre_config>/europe_geographic.png`

### `scripts/hopfield_analysis.py`

Genera una batería completa de gráficos para Hopfield clásico con los patrones de `data/patterns.txt`.

Uso:

```bash
python3 scripts/hopfield_analysis.py --mode sync
```

Parámetros:

- `--mode`: modo de actualización de la red clásica. Valores: `sync` o `async`. Default: `sync`.

Salida en `results/hopfield/<mode>/`:

- `1_stored_patterns.png`
- `2_recovery_grid.png`
- `3_recovery_steps.png`
- `4_spurious_state.png`
- `5_energy_convergence.png`
- `6_noise_robustness.png`
- `7_overlap_matrix.png`
- `8_basin_by_pattern.png`
- `9_capacity_experiment.png`

### `src/hopfield/hopfield.py`

Ejecuta una recuperación puntual de Hopfield desde un archivo de patrones y un archivo de consulta. Soporta red clásica binaria y red moderna continua.

Uso:

```bash
python3 src/hopfield/hopfield.py data/patterns.txt data/query_W.txt --noise 0.2 --seed 42 --mode sync
```

Parámetros posicionales:

- `patterns_file`: archivo con patrones almacenados 5x5.
- `query_file`: archivo con un patrón de consulta 5x5.

Opciones:

- `--max-iter`: máximo de iteraciones/sweeps. Default: `20`.
- `--quiet`: imprime solo el resultado final.
- `--noise`: fracción de pixeles a invertir en la consulta. Default: `0.2`.
- `--seed`: semilla para ruido y orden aleatorio.
- `--mode`: actualización de Hopfield clásico. Valores: `sync` o `async`. Default: `sync`.
- `--analyze`: imprime análisis de ortogonalidad de grupos de letras antes de correr.
- `--type`: tipo de red. Valores: `classic` o `modern`. Default: `classic`.
- `--beta`: temperatura inversa de la red moderna. Default: `4.0`.

### `scripts/plot_hopfield.py`

Genera una figura con la convergencia paso a paso de una consulta Hopfield, incluyendo energía y opcionalmente campos locales.

Uso:

```bash
python3 scripts/plot_hopfield.py data/patterns.txt data/query_W.txt --noise 0.2 --seed 42 --max-iter 20 --mode sync --out results/hopfield/convergence.png
```

Parámetros posicionales:

- `patterns_file`: archivo de patrones almacenados.
- `query_file`: archivo de consulta.

Opciones:

- `--noise`: fracción de ruido. Default: `0.2`.
- `--seed`: semilla. Default: `42`.
- `--max-iter`: máximo de iteraciones. Default: `20`.
- `--out`: ruta del PNG de salida. Default: `convergence.png`.
- `--mode`: `sync` o `async`.
- `--no-fields`: no genera la figura separada de campos locales.

### `scripts/hopfield_noise_tolerance.py`

Mide tolerancia al ruido por patrón y genera gráficos agregados.

Uso:

```bash
python3 scripts/hopfield_noise_tolerance.py data/patterns.txt --noise-steps 20 --trials 30 --out-dir results/hopfield/noise
```

Parámetros:

- `patterns_file`: archivo de patrones almacenados.
- `--noise-steps`: cantidad de divisiones entre 0 y 100% de ruido. Default: `20`.
- `--trials`: ensayos por nivel de ruido. Default: `30`.
- `--max-iter`: máximo de iteraciones. Default: `20`.
- `--seed`: semilla base. Default: `42`.
- `--out-dir`: directorio de salida. Default: directorio actual.
- `--mode`: modo de Hopfield clásico, `sync` o `async`. Default: `sync`.
- `--type`: `classic` o `modern`. Default: `classic`.
- `--beta`: beta de Hopfield moderna. Default: `4.0`.

Salida:

- `stacked_areas.png`
- `heatmap.png`
- `retrieval_curves.png`

### `scripts/compare_letters.py`

Calcula y grafica la matriz de producto interno normalizado entre patrones. Sirve para evaluar ortogonalidad.

Uso:

```bash
python3 scripts/compare_letters.py data/patterns.txt --out-dir results/letters
```

Parámetros:

- `patterns_file`: archivo de patrones.
- `--out-dir`: directorio de salida. Default: directorio actual.

Salida:

- `heatmap.png`
- Matriz textual por consola.

### `scripts/plot_letters.py`

Muestra las letras de `data/letters.txt` en grupos de 6 usando `matplotlib`.

Uso:

```bash
python3 scripts/plot_letters.py
```

No recibe parámetros. Nota: usa la ruta relativa `../data/letters.txt`, por lo que debe ejecutarse desde `scripts/` o ajustarse la ruta si se ejecuta desde la raíz.

### `scripts/oja_experiments.py`

Ejecuta los experimentos de Oja/Sanger sobre `data/europe.csv`: convergencia, sensibilidad a `learning_rate`/`decay`, comparación con PCA, ranking de países, varianza capturada e inicialización.

Uso:

```bash
python3 scripts/oja_experiments.py
```

No recibe parámetros por CLI. Los hiperparámetros están definidos en `main()`:

- `LR = 0.1`
- `EPOCHS = 10000`
- `DECAY = 0.01`
- `N_SEEDS = 5`
- `HEBB_LR = 0.001`
- `HEBB_EPOCHS = 200`

Salida en `results/oja/`, incluyendo:

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

### `src/oja/compare_with_pca.py`

Entrena una neurona de Oja y compara su vector de pesos contra la primera componente principal de `sklearn.decomposition.PCA`.

Uso:

```bash
python3 -m src.oja.compare_with_pca
```

No recibe parámetros. Imprime:

- Varianza explicada por PC1.
- Norma del vector de Oja.
- Similitud coseno absoluta.
- Diferencias por componente.
- Ranking de países según score.

### `PCA/main.py` y `PCA/pca_plot.py`

`PCA/pca_plot.py` contiene `plot_pca()`, que genera un biplot PCA 2D para `data/europe.csv`.

`PCA/main.py` define CLI:

```bash
python3 PCA/main.py --data data/europe.csv --out results/plots --seed 1
```

Parámetros:

- `--data`: CSV de entrada. Default: `data/europe.csv`.
- `--out`: directorio de salida. Default: `results/plots`.
- `--seed`: semilla. Default: `1`.

Salida:

- `pca.png`

Nota operativa: `PCA/main.py` importa `plot_pca` desde `preentrega.pca_plot`, pero el archivo fuente versionado está en `PCA/pca_plot.py`. Si el comando falla con `ModuleNotFoundError`, corregir ese import o ejecutar la función desde el módulo correcto.


## Notas 

- Ejecutar los comandos desde la raíz del repositorio salvo que se indique lo contrario.
- Los datos de Europa se estandarizan antes de Kohonen, Oja y PCA.
- Para reproducibilidad se permite usar `seed` en configs o parámetros CLI

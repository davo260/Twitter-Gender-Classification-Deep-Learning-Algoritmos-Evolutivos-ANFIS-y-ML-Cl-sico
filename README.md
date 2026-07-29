# Deep Learning — Talleres

Repositorio con los talleres del curso de Deep Learning. Incluye un trabajo
final centrado en **ANFIS (Adaptive Neuro-Fuzzy Inference System)** aplicado
al dataset *Twitter Gender Classifier*, comparado contra el mejor MLP
construido previamente, dentro de un estudio comparativo mas amplio que
tambien incluye optimizacion evolutiva de hiperparametros y aprendizaje de
maquina clasico. El entorno corre JupyterLab dentro de un contenedor Docker
para garantizar reproducibilidad.

**Autores:** Juan Sandoval Garcia, Cristian Organista Perilla, Diego Vera
Ortega, Cristian Parra Santa — Facultad de Ingenieria, Pontificia
Universidad Javeriana (curso Sistemas Inteligentes / Deep Learning).

---

## Tabla de contenido

1. [Contexto y motivacion](#contexto-y-motivacion)
2. [Relacion con el paper de referencia (PCA-ANFIS, 2025)](#relacion-con-el-paper-de-referencia)
3. [Dataset](#dataset)
4. [Notebooks del repositorio](#notebooks-del-repositorio)
5. [Fase 1 — Baselines de redes neuronales](#fase-1--baselines-de-redes-neuronales)
6. [Fase 2 — Optimizacion con algoritmos evolutivos](#fase-2--optimizacion-con-algoritmos-evolutivos)
7. [Pipeline ANFIS — fases del proceso](#pipeline-anfis--fases-del-proceso)
8. [Experimentos ANFIS](#experimentos-anfis)
9. [Fase 4 — Aprendizaje de maquina clasico](#fase-4--aprendizaje-de-maquina-clasico)
10. [Comparacion final ANFIS vs mejor MLP](#comparacion-final-anfis-vs-mejor-mlp)
11. [Comparacion global de todas las fases](#comparacion-global-de-todas-las-fases)
12. [Conclusiones](#conclusiones)
13. [Limitaciones](#limitaciones)
14. [Trabajo futuro](#trabajo-futuro)
15. [Setup / instalacion](#setup--instalacion)
16. [Estructura del proyecto](#estructura-del-proyecto)
17. [Referencias](#referencias)

---

## Contexto y motivacion

Los MLPs son potentes pero opacos: no permiten responder *"por que el modelo
predijo eso?"*. **ANFIS** es una alternativa que combina:

- la **capacidad de aprendizaje** de una red neuronal (backpropagation sobre
  todos los parametros), y
- la **interpretabilidad** de un sistema difuso (reglas tipo
  *"SI x1 es bajo Y x2 es alto ENTONCES ..."*).

El reto principal de ANFIS es la **explosion combinatoria de reglas**:
`n_rules = n_mfs ^ n_inputs`. Con las ~1240 features TF-IDF + numericas
del dataset, ANFIS seria imposible de entrenar (>10^10 reglas). La
solucion estandar es **reducir dimensionalidad con PCA antes de ANFIS**.

Este trabajo es, ademas, un estudio comparativo mas amplio: se implementaron
y evaluaron cuatro familias de modelos en orden progresivo — (1) baselines
de redes neuronales, (2) MLP optimizado con algoritmos evolutivos, (3) ANFIS
con reduccion PCA, y (4) modelos de aprendizaje de maquina clasico (SVM,
Random Forest, XGBoost, Stacking) con feature engineering avanzado. Segun
nuestra busqueda, es la primera aplicacion de PCA-ANFIS a este dataset.

---

## Relacion con el paper de referencia

> **Exploiting adaptive neuro-fuzzy inference systems for cognitive patterns
> in multimodal brain signal analysis** — Scientific Reports (Nature), 2025.
> [DOI / link](https://www.nature.com/articles/s41598-025-93241-9)

Este paper propone exactamente la combinacion **PCA + ANFIS** y la compara
contra un **MLP**, en el dominio de senales cerebrales multimodales. Reporta
PCA-ANFIS = 99.5% vs MLP ~90% en su tarea de clasificacion de patrones
cognitivos.

### Que tomamos del paper

| Aspecto del paper | Como lo aplicamos en este trabajo |
|---|---|
| PCA como reduccion previa al ANFIS | `TruncatedSVD` (PCA sparse-friendly) sobre el espacio TF-IDF (~1240 dims) -> 5-8 componentes |
| ANFIS como modelo principal | Implementacion propia en Keras (5 capas custom: `GaussianMembership`, `RuleFiringStrength`, `Normalization`, `TSKConsequent`, `Aggregation`) |
| MLP como baseline competitivo | El "Modelo Mejorado" de `train.ipynb` (512 -> 256 con BatchNorm + LeakyReLU + Dropout + class_weight balanceado) |
| Justificacion de PCA por *curse of dimensionality* en ANFIS | Mismo argumento: sin PCA habria 10^10+ reglas |
| Comparacion accuracy + interpretabilidad | Replicamos esta comparacion y agregamos `f1_macro`, `n_params`, `train_time_s`, `n_rules` |

### Que diferenciamos

- **Dominio:** el paper aplica PCA-ANFIS a senales cerebrales (EEG / MEG);
  nosotros lo aplicamos a **clasificacion de genero en texto de Twitter** —
  un dominio donde ANFIS no ha sido explorado segun nuestra busqueda.
- **Tarea:** clasificacion multiclase (4 clases: `male`, `female`, `brand`,
  `unknown`) — el paper trabaja en clasificacion binaria.
- **Estabilidad numerica:** implementamos T-norma en **log-dominio** y un
  callback `SigmaFloor(1e-2)` para evitar el colapso de las MFs gaussianas.
  Detalles ingenieriles que el paper no documenta explicitamente pero son
  necesarios para que ANFIS converja en alta dimensionalidad.
- **Diseño experimental:** corremos un **grid pequeño** (4 experimentos)
  que aisla los dos hiperparametros estructurales (`n_mfs`, `n_components`)
  en vez de reportar una sola configuracion.

---

## Dataset

*Twitter Gender Classifier* (Kaggle / CrowdFlower [12]), ~20,000 perfiles con
texto del tweet, descripcion del perfil, variables numericas (favoritos,
numero de tweets, edad de la cuenta) y etiqueta de genero. Tras la limpieza:

| Clase | Muestras | Proporcion |
|---|---|---|
| female | 6,700 | 33.6% |
| male | 6,194 | 31.0% |
| brand | 5,942 | 29.8% |
| unknown | 1,117 | 5.6% |
| **Total** | **19,953** | **100%** |

Para Fase 4 (ML clasico) se trabajo con clasificacion binaria
(male/female, 12,894 muestras, 52%/48%). Las demas fases usaron
clasificacion multiclase con las 4 clases.

**Preprocesamiento:**
1. Limpieza — nulos, minusculas, remocion de URLs/caracteres especiales, one-hot de categoricas.
2. Representacion textual — TF-IDF basico (~1,240 features) para Fases 1-3. Para Fase 4: TF-IDF word-level con bigramas (8,000 features) + TF-IDF char-level n-gramas 2-4 (3,000 features) + 12 meta-features linguisticos (longitud, mayusculas, puntuacion, emojis, hashtags, menciones, etc.).
3. Escalado y reduccion — `MaxAbsScaler` para fases de deep learning; `TruncatedSVD` (5-8 componentes) + `StandardScaler` para ANFIS.

Split estratificado 60/20/20 (train/val/test) para Fases 1-3; split 80/20
con validacion cruzada 5-fold para Fase 4. Los transformadores se ajustaron
exclusivamente sobre entrenamiento.

---

## Notebooks del repositorio

| Notebook | Descripcion |
|---|---|
| `jupyter/gender-classifier-eda.ipynb` | Analisis exploratorio del dataset Gender Classifier |
| `jupyter/limpieza-de-datos.ipynb` | Limpieza y preprocesamiento del CSV crudo |
| `jupyter/variables-categoricas.ipynb` | Tratamiento de variables categoricas |
| `jupyter/train.ipynb` | Baselines: LogReg, NN1, NN2, **Modelo Mejorado** (MLP usado en la comparacion final) |
| `jupyter/hyperopt_ga.ipynb` | Optimizacion de hiperparametros del MLP con GA, CMA-ES, DE y NSGA-II |
| `jupyter/hyperopt_ga_experimentos.ipynb` | 12 configuraciones evolutivas comparadas |
| **`jupyter/train_anfis.ipynb`** | **Pipeline ANFIS end-to-end (fases 01-07) + comparacion contra MLP** |
| `jupyter/anfis_layers.py` | Modulo con las 5 capas ANFIS, `build_anfis`, `initialize_from_data`, `SigmaFloor` |

*(Los experimentos de Fase 4 — ML clasico — viven en `AM/`.)*

---

## Fase 1 — Baselines de redes neuronales

Cinco modelos entrenados en `train.ipynb` sobre las 4 clases:

| Modelo | Accuracy | F1-W |
|---|---|---|
| Logistic Regression | 65.8% | 65.8% |
| Perceptron (TLU) | 61.6% | 62.0% |
| MLP 1 capa oculta | 65.4% | 65.0% |
| MLP Mejorado | 65.1% | 65.0% |
| **Ensamble LR+NN (voting)** | **66.5%** | **67.0%** |

El ensamble por votacion logra el mejor resultado de la fase. El MLP
Mejorado, pese a su mayor complejidad, no supera significativamente al
modelo lineal — el cuello de botella esta en los features, no en la
capacidad del modelo.

---

## Fase 2 — Optimizacion con algoritmos evolutivos

Optimizacion del MLP (`hyperopt_ga.ipynb`, `hyperopt_ga_experimentos.ipynb`,
implementado con DEAP) sobre el espacio: neuronas por capa (64-1024),
dropout (0.0-0.7), learning rate (1e-4 - 1e-2), batch size (32/64/128) y
alpha de LeakyReLU (0.0-0.3). Cuatro algoritmos, tres variantes cada uno:

| Algoritmo | Variante | Val Acc | Tiempo (s) |
|---|---|---|---|
| GA canonico | Baseline | 68.0% | 197 |
| GA canonico | Exploracion | 68.3% | 82 |
| GA canonico | Explotacion | 68.3% | 84 |
| CMA-ES | Baseline | 68.1% | 201 |
| CMA-ES | Local | 67.7% | 191 |
| CMA-ES | Global | 67.4% | 163 |
| DE | Baseline (dithering) | 68.6% | 2,473 |
| **DE** | **B. Exploit** | **68.9%** | **2,709** |
| NSGA-II | Varias | ~68.2% | ~300 |

DE alcanza el maximo del proyecto (68.9%) pero a ~45x el costo del GA
canonico, que ofrece resultados comparables (68.0-68.3%) en ~50-100s — la
opcion practica cuando el presupuesto computacional es limitado. Las tres
variantes de GA son robustas entre si, y NSGA-II (frentes de Pareto
accuracy vs numero de parametros / latencia / diversidad) confirma que el
tamaño del modelo correlaciona con mayor accuracy, con rendimientos
decrecientes a partir de ~10^6 parametros.

---

## Pipeline ANFIS — fases del proceso

Todo el flujo ANFIS vive en `jupyter/train_anfis.ipynb` (100 celdas
estructuradas en 7 fases). Cada fase persiste sus artefactos en
`checkpoints/` para poder reanudar desde cualquier punto.

```
data/gender-classifier-clean.csv
        │
        ▼
┌─────────────────────────────┐
│ Fase 01 — Preparacion       │  → checkpoints/anfis_data.pkl
│ Features (num + OHE + TFIDF)│
│ Split 60/20/20 estratificado│
│ MaxAbsScaler (fit solo train)│
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ Fase 02 — PCA (TruncatedSVD)│  → checkpoints/anfis_pca.pkl
│ 1240 dims -> 5-8 componentes│
│ Eleccion por var. acumulada │
│ StandardScaler post-PCA     │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ Fase 03 — Arquitectura ANFIS│  → anfis_layers.py
│ 5 capas Keras custom        │
│ Log-domain T-norm           │
│ TSK orden 1 (p·x + q)       │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ Fase 04 — Inicializacion    │  → in-memory
│ Centros = cuantiles training│
│ σ = (max-min)/(2·n_mfs)     │
│ SigmaFloor(1e-2) callback   │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ Fase 05 — Entrenamiento ref │  → anfis_model.keras + summary
│ Adam(1e-3), batch 64        │
│ EarlyStopping(val_acc, p=15)│
│ NaN audit, dead-rule check  │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ Fase 06 — 4 Experimentos    │  → checkpoints/anfis_experiments.pkl
│ Grid {n_mfs, n_components}  │  → checkpoints/anfis_E*.keras
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ Fase 07 — Comparacion vs MLP│  → checkpoints/anfis_comparison.pkl
│ Tabla de metricas (5 modelos)│  → matrices de confusion (2x3)
│ Top-K reglas legibles       │  → discusion
└─────────────────────────────┘
```

### Por que cada fase

- **Fase 01** asegura que ANFIS y MLP veran exactamente las mismas features
  y el mismo split — comparacion justa.
- **Fase 02** es **lo que hace ANFIS posible** en este problema (replicando
  la decision del paper de referencia).
- **Fase 03** implementa la arquitectura propuesta originalmente por Jang
  (1993), con las mejoras de estabilidad numerica documentadas arriba.
- **Fase 04** evita reglas muertas desde el init — un problema clasico
  cuando los centros se inicializan al azar.
- **Fase 05** sirve de smoke test y produce las curvas de entrenamiento.
- **Fase 06** responde a la pregunta cientifica: *como afectan `n_mfs` y
  `n_components` al desempeño?*
- **Fase 07** responde a la pregunta practica: *vale la pena ANFIS en
  este problema?*

---

## Experimentos ANFIS

La fase 06 corre un **grid pequeño de 4 experimentos** que aisla los dos
hiperparametros estructurales mas importantes. Resultados finales:

| Experimento | `n_mfs` | `n_components` | `n_rules` | Val Acc | Epocas | Parametros | Hipotesis |
|---|---|---|---|---|---|---|---|
| **E1 (baseline)** | 2 | 6 | 64 | 52.2% | 42 | 1,368 | Punto de partida razonable |
| **E2 (mas MFs)** | 3 | 6 | 729 | 52.3% | 40 | 15,345 | Mas granularidad ayuda? |
| **E3 (menos PCA)** | 2 | 5 | 32 | 52.0% | 54 | 596 | Cuanto perdemos al simplificar? |
| **E4 (mas PCA)** | 2 | 8 | 256 | **55.4%** | 24 | 6,944 | Mas informacion ayuda? |

Ningun experimento presento reglas muertas, validando la inicializacion
data-driven por cuantiles.

**Lectura de resultados:**
- **E1 vs E2** (efecto de `n_mfs`, `n_components=6` fijo): aumentar de 2 a 3
  MFs multiplica las reglas ~11x (64 -> 729) con ganancia marginal en
  accuracy (+0.1 pp) — el costo no compensa.
- **E1 vs E3 vs E4** (efecto de `n_components`, `n_mfs=2` fijo): E3 produce
  el modelo mas interpretable (32 reglas, 596 parametros) con penalizacion
  minima (-0.2 pp vs E1). E4, con mayor varianza explicada retenida
  (21.9%), logra el mejor accuracy ANFIS del proyecto (55.4%).
- **Ganador:** **E4** (`n_mfs=2, n_components=8`) — mejor balance
  accuracy/interpretabilidad entre los cuatro, con 256 reglas.

---

## Fase 4 — Aprendizaje de maquina clasico

Cuatro modelos sobre clasificacion binaria (male/female, 11,012 features
totales), en `AM/`:

1. **SVM** — `LinearSVC(C=1.0)` con `CalibratedClassifierCV`.
2. **Random Forest** — `GridSearchCV` sobre `n_estimators∈{100,200,300}`,
   `max_depth∈{None,20,30}`, `max_features∈{sqrt,log2}` (90 candidatos x
   5 folds = 450 ajustes).
3. **XGBoost** — GPU/CUDA, `GridSearchCV` sobre
   `lr∈{0.05,0.1,0.2}`, `depth∈{3,5,7}`, `n_est∈{100,200,300}`,
   `subsample∈{0.8,1.0}` (270 candidatos x 5 folds = 1,350 ajustes).
4. **Stacking** — SVM+RF+XGBoost como base learners, Logistic Regression
   como meta-learner, `stack_method='predict_proba'`, CV=5.

| Modelo | Acc | F1-W | AUC | Recall female | Recall male |
|---|---|---|---|---|---|
| SVM (Linear) | 62.7% | 62.4% | 0.668 | 0.70 | 0.54 |
| Random Forest | 64.0% | 64.0% | 0.692 | 0.67 | 0.61 |
| XGBoost (GPU) | 51.6% | 50.9% | 0.530 | 0.63 | 0.39 |
| **Stacking** | **65.0%** | **65.0%** | **0.702** | 0.67 | 0.63 |

El Stacking domina en todos los umbrales de decision (curva ROC) y logra
el mejor balance entre clases. Todos los modelos muestran sesgo sistematico
hacia `female` (mayor recall que en `male`); XGBoost es el caso extremo
(recall male = 0.39). El feature mas discriminativo (Random Forest, Mean
Decrease Impurity) es el meta-feature `n_emoji`, por encima de cualquier
termino TF-IDF — valida la inversion en feature engineering.

---

## Comparacion final ANFIS vs mejor MLP

La fase 07 compara los 4 ANFIS contra el **mejor MLP de `train.ipynb`** —
el "Modelo Mejorado":

- `Dense(512)` + BatchNorm + LeakyReLU(0.01) + Dropout(0.4)
- `Dense(256)` + BatchNorm + LeakyReLU(0.01) + Dropout(0.3)
- `Dense(n_classes, softmax)`
- `Adam(lr=1e-3)`, `sparse_categorical_crossentropy`
- `class_weight='balanced'`
- `EarlyStopping(val_loss, patience=7)` + `ReduceLROnPlateau(factor=0.5, patience=3)`
- Hasta 100 epocas, `batch_size=128`

### Metricas reportadas (tabla en `anfis_comparison.pkl`)

| Columna | Significado | Por que importa |
|---|---|---|
| `accuracy_test` | Fraccion de aciertos en test | Metrica intuitiva |
| `f1_macro` | Media simple de F1 por clase | Penaliza ignorar minoritarias (`unknown`) — clave aqui |
| `precision_test` (macro) | Que tan precisas son las predicciones | Tradeoff con recall |
| `recall_test` (macro) | Que tan completa es la captura por clase | Tradeoff con precision |
| `n_params` | Tamaño del modelo | Proxy del costo de inferencia |
| `train_time_s` | Tiempo de entrenamiento | Costo computacional real |
| `n_rules` | Tamaño del sistema difuso (solo ANFIS) | Interpretabilidad inversa (menos = mas legible) |

### Resultado

El mejor ANFIS (E4, 55.4%) queda **13.5 pp por debajo** del mejor modelo
del proyecto (MLP + DE, 68.9%). La brecha se explica en buena parte por la
agresiva reduccion PCA (17-22% de varianza retenida) que impone la
restriccion combinatoria de reglas — no por una limitacion intrinseca de
ANFIS como clasificador.

### Visualizaciones producidas

1. **Tabla comparativa** con 5 filas (E1, E2, E3, E4, MLP).
2. **Matrices de confusion** en grid 2x3 — permite ver *donde* falla cada
   modelo (que clases confunde con cuales).
3. **Top-K reglas** del mejor experimento ANFIS (E4), renderizadas en
   lenguaje natural en español:

   > *"SI PC1 es bajo Y PC2 es alto Y PC3 es bajo ... ENTONCES
   > contribucion=[female:+0.34, male:-0.12, brand:+0.02, unknown:-0.05]"*

   Cada regla viene acompañada de su firing promedio sobre el val set
   (que tanto contribuye en promedio) y la contribucion lineal del
   consequent TSK por clase.

### Lo que el MLP NO puede entregar

Un MLP tiene patrones distribuidos entre miles de pesos sin estructura
semantica. ANFIS produce una **lista finita y legible de reglas** — base
ideal para auditoria, compliance y debugging, incluso cuando su accuracy
es menor.

---

## Comparacion global de todas las fases

| Modelo | Fase | Accuracy | Tarea | AUC | Interpretable |
|---|---|---|---|---|---|
| Ensamble LR+NN | 1 | 66.5% | 4 clases | – | No |
| **MLP + DE** | 2 | **68.9%** | 4 clases | – | No |
| ANFIS E4 | 3 (ANFIS) | 55.4% | 4 clases | – | **Si** |
| ANFIS E3 (simple) | 3 (ANFIS) | 52.0% | 4 clases | – | **Si** |
| Stacking | 4 | 65.0% | 2 clases (binario) | 0.702 | No |

La columna AUC solo aplica a Fase 4 (binaria); las fases anteriores operan
sobre 4 clases sin AUC-ROC binario calculado. La comparacion de accuracies
entre Fase 4 y el resto debe interpretarse con cautela por el cambio de
tarea (ver [Limitaciones](#limitaciones)).

---

## Conclusiones

### Tecnicas

1. **PCA es indispensable para ANFIS en alta dimensionalidad.** Sin
   reducir a 5-8 componentes, el problema seria imposible (>10^10 reglas).
   Este resultado replica directamente la decision del paper de Scientific
   Reports 2025 — sin PCA, ANFIS no escala.

2. **La estabilidad numerica no es opcional.** Tres decisiones de
   ingenieria evitan el colapso del entrenamiento:
   - Parametrizar σ como `log_sigma` (siempre positivo).
   - T-norma en **log-dominio** (`log_w = sum(log_mu)`) con una sola `exp` al final.
   - Callback `SigmaFloor(1e-2)` que clipa σ despues de cada batch.

   Sin estas tres, ANFIS produce NaN en las primeras epocas o colapsa todas
   sus reglas a un solo punto del espacio.

3. **La inicializacion por cuantiles previene reglas muertas.** Anclar los
   centros de las MFs a los percentiles del training set garantiza que toda
   regla dispara en >= 1% del training set desde la epoca 0. Los 4
   experimentos confirmaron 0 reglas muertas.

### Cientificas

4. **E4 gana entre los 4 ANFIS (55.4%).** Aumentar `n_mfs` de 2 a 3
   (E1 -> E2) multiplica las reglas ~11x con ganancia marginal (+0.1 pp) —
   no compensa el costo. Reducir `n_components` (E3) da el modelo mas
   interpretable (32 reglas) con penalizacion minima. Aumentar
   `n_components` (E4) da el mejor accuracy ANFIS gracias a mayor varianza
   explicada retenida (21.9%).

5. **La optimizacion evolutiva mejora significativamente el MLP.** DE elevo
   el accuracy de 65.1% a 68.9% (+3.8 pp) — el mejor resultado global del
   proyecto. GA canonico ofrece resultados comparables en ~20x menos
   tiempo, siendo la opcion practica con presupuesto computacional
   limitado.

6. **ANFIS vs MLP — gap de 13.5 pp esperado y explicable.** El MLP tiene
   mayor capacidad bruta y gana en accuracy puro. ANFIS se queda atras pero
   **ofrece reglas legibles**, que el MLP no puede dar — el tradeoff es
   consistente con la literatura (accuracy vs interpretabilidad).

7. **El Stacking valida la hipotesis del meta-clasificador (Fase 4).**
   SVM+RF+XGBoost supero a cada modelo individual, alcanzando el mayor
   AUC-ROC del proyecto (0.702). El feature engineering con tres espacios
   combinados (TF-IDF word + char + meta-features) fue determinante.

8. **Los meta-features son mas discriminativos que TF-IDF.** `n_emoji`
   lidera la importancia de features en Random Forest, validando la
   inversion en feature engineering sobre patrones estilisticos.

9. **Sesgo sistematico hacia `female` en todos los modelos de Fase 4.** No
   se explica por desbalance del dataset (52%/48%) sino por mayor
   estandarizacion del lenguaje femenino en Twitter (emojis, exclamaciones,
   patrones estilisticos mas predecibles). XGBoost es el caso extremo
   (recall male = 0.39).

### Practicas / para la presentacion

10. **Cuando elegir ANFIS sobre MLP en este tipo de problemas:**
    - Cuando el stakeholder necesita **explicabilidad** (auditoria, dominios regulados).
    - Cuando el numero de features es moderado (post-reduccion).
    - Cuando se acepta un costo de entrenamiento mayor a cambio de transparencia.

11. **Cuando elegir MLP:**
    - Cuando solo importa el accuracy maximo.
    - Cuando la dimensionalidad post-reduccion sigue siendo alta (>10-12)
      y `n_mfs^n_components` se vuelve inviable.
    - Cuando no se necesita derivar reglas para presentar al negocio.

12. **Aporte de este trabajo al estado del arte:** segun nuestra busqueda,
    ANFIS no se ha aplicado al *Twitter Gender Classifier dataset* — los
    baselines existentes son CNN, LSTM y transformers (PAN-2018, 85-91%
    acc). Este trabajo aporta:
    - Una implementacion limpia de PCA-ANFIS en Keras moderna.
    - Un diseño experimental controlado (4 configs aisladas).
    - Una comparacion sistematica de cuatro familias de modelos sobre el
      mismo dataset y pipeline (redes neuronales, evolutivos, neuro-difuso,
      ML clasico).
    - Las primeras reglas difusas legibles para este dataset.

---

## Limitaciones

- **Gap ANFIS vs MLP (13.5 pp).** Explicado parcialmente por la agresiva
  reduccion PCA (17-22% varianza retenida) impuesta por la restriccion
  combinatoria de reglas.
- **Sesgo sistematico hacia female** en Fase 4 (fenomeno linguistico, no
  de desbalance de clases).
- **XGBoost con bajo rendimiento.** TF-IDF disperso (99% ceros) no provee
  señal suficiente para boosting; embeddings densos (Word2Vec, GloVe)
  podrian revertir este resultado.
- **Comparabilidad entre fases.** Fase 4 usa clasificacion binaria
  (male/female), Fases 1-3 usan 4 clases. La comparacion de accuracies
  entre ellas debe interpretarse con cautela.

---

## Trabajo futuro

1. **Embeddings densos para ANFIS y XGBoost.** Reemplazar TF-IDF por
   Word2Vec o GloVe; en trabajos relacionados esta estrategia elevo el
   accuracy en >10 pp.
2. **ANFIS en tarea binaria.** El mejor rendimiento en clasificacion
   binaria sugiere que ANFIS podria beneficiarse al simplificar la tarea.
3. **Feature selection previa a PCA.** Aplicar mutual information o
   chi-squared antes de `TruncatedSVD` para retener mas informacion
   discriminativa.
4. **Optimizacion evolutiva de ANFIS.** Usar GA o DE para optimizar
   conjuntamente `n_mfs`, `n_components` y parametros de entrenamiento.
5. **Comparacion con transformers.** BERT/DistilBERT alcanzo 85.5% en el
   mismo dominio; cuantificar el gap residual contextualiza el trabajo
   frente al estado del arte.

---

## Setup / instalacion

### Dependencias

Python 3.11. Paquetes en `requirements.txt`:

| Categoria | Paquetes |
|---|---|
| Manipulacion de datos | `numpy`, `pandas` |
| Visualizacion | `matplotlib`, `seaborn` |
| ML / Deep Learning | `scikit-learn`, `tensorflow`, `keras` |
| Algoritmos Evolutivos | `deap`, `cma` |
| Jupyter | `jupyterlab`, `ipywidgets` |
| Utilidades | `kagglehub` |

### Requisitos previos

- [Docker](https://docs.docker.com/get-docker/) + Docker Compose
- [Task](https://taskfile.dev/installation/) (task runner)

### Uso con Task

```bash
task up       # Construye la imagen y levanta JupyterLab en http://localhost:8888
task logs     # Sigue los logs del contenedor
task shell    # Abre una shell dentro del contenedor
task down     # Detiene el contenedor
task restart  # Reconstruye y reinicia
```

### Uso solo con Docker Compose

```bash
docker compose build
docker compose up -d
```

### Uso sin Docker

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
jupyter lab
```

### Como reproducir el pipeline ANFIS

1. Descargar el CSV crudo de Kaggle y colocarlo en `data/gender-classifier-DFE-791531.csv`.
2. Ejecutar `jupyter/limpieza-de-datos.ipynb` (genera `data/gender-classifier-clean.csv`).
3. Ejecutar `jupyter/train_anfis.ipynb` end-to-end (Run All Cells).
   - Las fases 01-05 producen los artefactos base.
   - La fase 06 entrena los 4 experimentos (~5-15 min por experimento).
   - La fase 07 produce la tabla comparativa, matrices de confusion y reglas top-K.

Si una fase ya corrio, sus checkpoints permiten re-ejecutar solo las fases siguientes.

---

## Estructura del proyecto

```
.
├── Dockerfile
├── docker-compose.yml
├── Taskfile.yml
├── requirements.txt
├── README.md
├── data/                            # Datasets (ignorado por git)
│   └── gender-classifier-clean.csv  # Output de limpieza-de-datos.ipynb
├── checkpoints/                     # Artefactos del pipeline ANFIS (commiteable)
│   ├── anfis_data.pkl               # Fase 01
│   ├── anfis_pca.pkl                # Fase 02
│   ├── anfis_model.keras            # Fase 05 (modelo de referencia)
│   ├── anfis_training_summary.pkl   # Fase 05
│   ├── anfis_training_log.csv       # Fase 05
│   ├── anfis_experiments.pkl        # Fase 06 (4 experimentos)
│   ├── anfis_E*.keras               # Fase 06 (un modelo por experimento)
│   └── anfis_comparison.pkl         # Fase 07 (tabla + predicciones)
├── AM/                               # Fase 4: ML clasico (SVM, RF, XGBoost, Stacking)
└── jupyter/
    ├── gender-classifier-eda.ipynb
    ├── limpieza-de-datos.ipynb
    ├── variables-categoricas.ipynb
    ├── train.ipynb                   # MLP baseline (Modelo Mejorado)
    ├── hyperopt_ga.ipynb
    ├── hyperopt_ga_experimentos.ipynb
    ├── train_anfis.ipynb             # Pipeline ANFIS end-to-end
    └── anfis_layers.py               # Capas ANFIS, init, callbacks
```

---

## Referencias

### Paper principal de referencia

- **Exploiting adaptive neuro-fuzzy inference systems for cognitive patterns
  in multimodal brain signal analysis.** *Scientific Reports (Nature)*, 2025.
  [link](https://www.nature.com/articles/s41598-025-93241-9)
  Combina PCA + ANFIS y lo compara contra MLP. Reporta PCA-ANFIS 99.5% vs MLP ~90%.

### Soporte teorico

- **Adaptive Neuro-Fuzzy Inference System: Overview, Strengths, Limitations,
  and Solutions.** *Springer*, 2018.
  [link](https://link.springer.com/chapter/10.1007/978-3-319-61845-6_52)
  Documenta la explosion combinatoria de reglas y propone reduccion de
  dimensionalidad como solucion estandar.

- **Taxonomy of Adaptive Neuro-Fuzzy Inference System in Modern Engineering
  Sciences.** *PMC*, 2021.
  [link](https://pmc.ncbi.nlm.nih.gov/articles/PMC8437605/)
  Revision sistematica de aplicaciones ANFIS, incluyendo clasificacion.

- **Review of Medical Image Classification using the Adaptive Neuro-Fuzzy
  Inference System.** *PMC*, 2013.
  [link](https://pmc.ncbi.nlm.nih.gov/articles/PMC3592505/)
  Caso de uso clasico de ANFIS con reduccion de dimensionalidad previa.

### Baselines del dominio (Twitter gender classification)

- **Twitter-Based Gender Recognition Using Transformers.** *arXiv:2205.06801*, 2022.
  [link](https://arxiv.org/pdf/2205.06801)
  PAN-2018, 85.5% accuracy con BERT + ViT.

- **Gender Prediction from Tweets: Improving Neural Approaches** /
  **Combining Feature Spaces for Improved Classification.** *arXiv:1908.09919*, 2019.
  [link](https://arxiv.org/pdf/1908.09919)

### Otras referencias citadas en el paper del proyecto

- S. Vashisth et al., "Gender classification using machine learning with TF-IDF and meta-features," IEEE ICCMC, 2021.
- J. Yang et al., "A stacking meta-classifier for gender classification in health records," JAMIA Open, 2021.
- K. Deb, A. Pratap, S. Agarwal, T. Meyarivan, "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II," IEEE Trans. Evolutionary Computation, vol. 6, no. 2, pp. 182-197, 2002.
- CrowdFlower, "Twitter User Gender Classification," Kaggle, 2016.

### Articulo seminal de ANFIS

- **Jang, J.-S. R.** *ANFIS: Adaptive-Network-based Fuzzy Inference System.*
  IEEE Trans. on Systems, Man, and Cybernetics, vol. 23, no. 3, pp. 665-685, 1993.

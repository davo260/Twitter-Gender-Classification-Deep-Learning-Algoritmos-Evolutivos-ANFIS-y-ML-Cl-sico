"""
ANFIS — capas Keras (Fase 03 del pipeline `train_anfis.ipynb`).

Implementa una red ANFIS TSK de orden 1 como `tf.keras.layers.Layer`s
diferenciables. Diseñada para consumir las proyecciones PCA/SVD producidas en
la fase 02 (`checkpoints/anfis_pca.pkl`, `n_components in [5, 8]`).

Arquitectura (5 capas):
    1. GaussianMembership   — μ_ij(x) = exp(-((x - c_ij) / σ_ij)^2),
                              σ parametrizado como exp(log_sigma) (positividad sin
                              proyecciones).
    2. RuleFiringStrength   — producto T-norm en el dominio log: w_k = exp(Σ log μ).
                              Evita underflow numérico cuando hay varias entradas.
    3. Normalization        — w_k / (Σ w_k + ε), ε = 1e-12.
    4. TSKConsequent (ord1) — f_k = p_k · x + q_k.
    5. Aggregation          — logit = Σ w_k_norm * f_k, sigmoide/softmax en el head.

Restricción dura (cf. anfis-builder skill):
    n_inputs in [1, 10] y n_mfs in {2, 3}. Para esta tarea el rango efectivo es
    n_inputs in [5, 8].

Uso:

    from anfis_layers import build_anfis, count_rules

    model = build_anfis(n_inputs=6, n_mfs=2, n_classes=4)
    model.summary()

La inicialización data-driven (cuantiles) y los callbacks (sigma floor) se
implementan en la fase 04 — este módulo provee solo la arquitectura y un
factory compilado con defaults razonables.
"""

from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
import tensorflow as tf


# -----------------------------------------------------------------------------
# Constantes
# -----------------------------------------------------------------------------
EPS_FIRING = 1e-12          # estabilizador en la normalización
LOG_SIGMA_INIT = 0.0        # log(1.0); la fase 04 sobre-escribe con quantile init
CENTER_INIT_STD = 1.0       # init razonable para entradas StandardScaler-izadas


# -----------------------------------------------------------------------------
# Capa 1: Fuzzificación con MFs gaussianas
# -----------------------------------------------------------------------------
class GaussianMembership(tf.keras.layers.Layer):
    """Por cada entrada calcula `n_mfs` valores de pertenencia gaussianos.

    Pesos entrenables:
        c         shape [n_inputs, n_mfs]    — centros
        log_sigma shape [n_inputs, n_mfs]    — log de las amplitudes (positividad
                                               garantizada por exp).

    Entrada:  [batch, n_inputs]
    Salida:   [batch, n_inputs, n_mfs]  con μ_ij(x_i) ∈ (0, 1].
    """

    def __init__(self, n_inputs: int, n_mfs: int, name: str = "gaussian_mf", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_inputs = int(n_inputs)
        self.n_mfs = int(n_mfs)

    def build(self, input_shape):
        # Init de centros: spread uniforme en [-1, 1] por defecto; la fase 04
        # reemplaza con cuantiles del training set.
        if self.n_mfs == 1:
            init_centers = np.zeros((self.n_inputs, 1), dtype=np.float32)
        else:
            grid = np.linspace(-1.0, 1.0, self.n_mfs, dtype=np.float32)
            init_centers = np.tile(grid, (self.n_inputs, 1))

        self.c = self.add_weight(
            name="centers",
            shape=(self.n_inputs, self.n_mfs),
            initializer=tf.keras.initializers.Constant(init_centers),
            trainable=True,
        )
        self.log_sigma = self.add_weight(
            name="log_sigma",
            shape=(self.n_inputs, self.n_mfs),
            initializer=tf.keras.initializers.Constant(LOG_SIGMA_INIT),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        # x: [B, n_inputs]  ->  [B, n_inputs, 1]
        x_exp = tf.expand_dims(x, axis=-1)                       # [B, I, 1]
        c = tf.expand_dims(self.c, axis=0)                       # [1, I, M]
        sigma = tf.expand_dims(tf.exp(self.log_sigma), axis=0)   # [1, I, M]
        # μ = exp(-((x - c)/σ)^2)
        z = (x_exp - c) / sigma
        return tf.exp(-tf.square(z))                             # [B, I, M]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_inputs": self.n_inputs, "n_mfs": self.n_mfs})
        return cfg


# -----------------------------------------------------------------------------
# Capa 2: Fuerza de disparo de reglas (log-domain T-norm)
# -----------------------------------------------------------------------------
class RuleFiringStrength(tf.keras.layers.Layer):
    """Aplica la T-norma producto sobre las MFs de cada entrada en cada regla.

    Cada regla es una tupla (m_1, ..., m_I) ∈ [0, M)^I que selecciona una MF
    por entrada. Hay `M^I` reglas. Para evitar underflow, se computa:

        log w_k = Σ_i log μ_{i, m_i}(x_i)
        w_k     = exp(log w_k)

    Entrada:  μ  shape [B, n_inputs, n_mfs]
    Salida:   w  shape [B, n_rules]   con n_rules = n_mfs ** n_inputs
    """

    def __init__(self, n_inputs: int, n_mfs: int, name: str = "rule_firing", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_inputs = int(n_inputs)
        self.n_mfs = int(n_mfs)
        self.n_rules = self.n_mfs ** self.n_inputs

        # Índice [n_rules, n_inputs] con la MF activa por entrada para cada regla.
        # Producto cartesiano determinista (orden lexicográfico).
        combos = np.array(list(itertools.product(range(self.n_mfs), repeat=self.n_inputs)),
                          dtype=np.int32)  # [n_rules, n_inputs]
        # `rule_index` se usa con tf.gather sobre el eje M.
        self.rule_index = tf.constant(combos, dtype=tf.int32)

    def call(self, mu):
        # mu: [B, I, M]  ->  log μ con clip para evitar log(0).
        log_mu = tf.math.log(tf.clip_by_value(mu, 1e-30, 1.0))   # [B, I, M]

        # Para cada entrada i, recogemos log_mu[:, i, rule_index[:, i]]
        # via gather. Resultado: [I, B, n_rules] -> sumamos sobre I.
        # Implementación vectorizada usando one-hot:
        #   rule_one_hot: [n_rules, I, M]
        rule_one_hot = tf.one_hot(self.rule_index, depth=self.n_mfs, dtype=log_mu.dtype)
        # einsum: 'bim,rim->br'  (sumamos sobre I y M, dejando B x n_rules)
        log_w = tf.einsum("bim,rim->br", log_mu, rule_one_hot)
        # Exponenciamos al final (single exp, sin productos en cascada).
        w = tf.exp(log_w)
        return w

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_inputs": self.n_inputs, "n_mfs": self.n_mfs})
        return cfg


# -----------------------------------------------------------------------------
# Capa 3: Normalización
# -----------------------------------------------------------------------------
class Normalization(tf.keras.layers.Layer):
    """w_k_norm = w_k / (Σ_k w_k + ε)."""

    def __init__(self, name: str = "normalization", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, w):
        total = tf.reduce_sum(w, axis=-1, keepdims=True) + EPS_FIRING
        return w / total


# -----------------------------------------------------------------------------
# Capa 4: Consecuentes TSK orden 1
# -----------------------------------------------------------------------------
class TSKConsequent(tf.keras.layers.Layer):
    """Consecuente lineal por regla y por clase.

    Para binario / multiclase necesitamos `n_outputs` logits por regla.
    Cada regla k produce f_k(x) = p_k · x + q_k con
        p   shape [n_rules, n_inputs, n_outputs]
        q   shape [n_rules, n_outputs]

    Entrada:  x  [B, n_inputs]
    Salida:   f  [B, n_rules, n_outputs]
    """

    def __init__(self, n_rules: int, n_inputs: int, n_outputs: int,
                 name: str = "tsk_consequent", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_rules = int(n_rules)
        self.n_inputs = int(n_inputs)
        self.n_outputs = int(n_outputs)

    def build(self, input_shape):
        self.p = self.add_weight(
            name="p",
            shape=(self.n_rules, self.n_inputs, self.n_outputs),
            initializer="zeros",
            trainable=True,
        )
        self.q = self.add_weight(
            name="q",
            shape=(self.n_rules, self.n_outputs),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        # x: [B, I]   p: [R, I, O]   -> f_lin: [B, R, O]
        f_lin = tf.einsum("bi,rio->bro", x, self.p)
        f = f_lin + tf.expand_dims(self.q, axis=0)              # broadcast batch
        return f

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "n_rules": self.n_rules,
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
        })
        return cfg


# -----------------------------------------------------------------------------
# Capa 5: Agregación (suma ponderada por fuerza normalizada)
# -----------------------------------------------------------------------------
class Aggregation(tf.keras.layers.Layer):
    """Combina f_k y w_k_norm en logits de tamaño [B, n_outputs].

    logits[b, o] = Σ_k w_norm[b, k] * f[b, k, o]
    """

    def __init__(self, name: str = "aggregation", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, inputs):
        w_norm, f = inputs       # [B, R], [B, R, O]
        w_exp = tf.expand_dims(w_norm, axis=-1)            # [B, R, 1]
        return tf.reduce_sum(w_exp * f, axis=1)            # [B, O]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def count_rules(n_inputs: int, n_mfs: int) -> int:
    return int(n_mfs) ** int(n_inputs)


# -----------------------------------------------------------------------------
# Inicialización data-driven (Fase 04)
# -----------------------------------------------------------------------------
# Schedule de cuantiles por n_mfs. Documentado en `train_anfis.ipynb` y en el
# spec 04 (`Quantile-based membership initialization`).
QUANTILE_SCHEDULE = {
    2: (0.25, 0.75),
    3: (0.17, 0.50, 0.83),
}


def _find_layer(model: tf.keras.Model, cls) -> tf.keras.layers.Layer:
    for layer in model.layers:
        if isinstance(layer, cls):
            return layer
    raise ValueError(f"No se encontró capa de tipo {cls.__name__} en el modelo.")


def initialize_from_data(
    model: tf.keras.Model,
    Z_train: np.ndarray,
    n_mfs: Optional[int] = None,
    min_sigma: float = 1e-2,
) -> dict:
    """Inicializa MFs gaussianas con cuantiles del training set y resetea TSK a cero.

    - **Centros `c`** (shape [n_inputs, n_mfs]): para cada dimensión `i`, se
      colocan en los cuantiles definidos por `QUANTILE_SCHEDULE[n_mfs]` de
      `Z_train[:, i]`.
    - **Amplitudes `σ`** (parametrizadas como `log_sigma`): por dimensión,
      `σ_i = (max - min) / (2 * n_mfs)`. Se aplica un piso `max(σ_i, min_sigma)`
      antes de tomar log, para evitar log(0) si la columna fuera constante.
    - **TSK `p`, `q`**: cero. El gradiente del cross-entropy los moverá desde
      el origen sin sesgos arbitrarios.

    Args:
        model: modelo construido por `build_anfis(...)` (todavía sin entrenar).
        Z_train: matriz de entrenamiento [N, n_inputs] (ya escalada / PCA).
        n_mfs: número de MFs por entrada. Si es None, se toma `model.n_mfs`.
        min_sigma: piso para σ (no para `log_sigma`); se aplica también como
                   floor lógico de la inicialización.

    Returns:
        dict con los arrays asignados (`centers`, `log_sigma`, `sigma`) — útil
        para inspección y los tests de sanity de la fase 04.
    """
    Z = np.asarray(Z_train, dtype=np.float32)
    if Z.ndim != 2:
        raise ValueError(f"Z_train debe ser 2D, recibí shape={Z.shape}")

    if n_mfs is None:
        n_mfs = getattr(model, "n_mfs", None)
        if n_mfs is None:
            raise ValueError("n_mfs no provisto y el modelo no expone .n_mfs")
    n_mfs = int(n_mfs)

    if n_mfs not in QUANTILE_SCHEDULE:
        raise ValueError(
            f"n_mfs={n_mfs} no soportado por QUANTILE_SCHEDULE={list(QUANTILE_SCHEDULE)}"
        )

    n_inputs = Z.shape[1]
    quantiles = np.array(QUANTILE_SCHEDULE[n_mfs], dtype=np.float64)  # [n_mfs]

    # Centros: np.quantile sobre el eje 0 -> shape [n_mfs, n_inputs] -> .T
    centers = np.quantile(Z, quantiles, axis=0).T.astype(np.float32)   # [I, M]

    # Anchos por dimensión: (max - min) / (2 * n_mfs)
    ranges = (Z.max(axis=0) - Z.min(axis=0)).astype(np.float32)        # [I]
    sigma_per_input = np.maximum(ranges / (2.0 * n_mfs), float(min_sigma))
    sigma = np.tile(sigma_per_input.reshape(-1, 1), (1, n_mfs)).astype(np.float32)  # [I, M]
    log_sigma = np.log(sigma).astype(np.float32)

    # Asignar a la capa GaussianMembership
    mf_layer = _find_layer(model, GaussianMembership)
    if mf_layer.n_inputs != n_inputs:
        raise ValueError(
            f"n_inputs de Z_train={n_inputs} != modelo={mf_layer.n_inputs}"
        )
    if mf_layer.n_mfs != n_mfs:
        raise ValueError(
            f"n_mfs solicitado={n_mfs} != modelo={mf_layer.n_mfs}"
        )
    mf_layer.set_weights([centers, log_sigma])

    # Resetear TSK consequents (p, q) a cero
    tsk_layer = _find_layer(model, TSKConsequent)
    p_zero = np.zeros((tsk_layer.n_rules, tsk_layer.n_inputs, tsk_layer.n_outputs),
                      dtype=np.float32)
    q_zero = np.zeros((tsk_layer.n_rules, tsk_layer.n_outputs), dtype=np.float32)
    tsk_layer.set_weights([p_zero, q_zero])

    return {
        "centers": centers,
        "log_sigma": log_sigma,
        "sigma": sigma,
        "quantiles": quantiles,
        "ranges": ranges,
    }


# -----------------------------------------------------------------------------
# Callback: piso para σ (evita reglas muertas)
# -----------------------------------------------------------------------------
class SigmaFloor(tf.keras.callbacks.Callback):
    """Después de cada batch, clipea `log_sigma >= log(min_sigma)`.

    Sin este floor, σ colapsa a cero en reglas sobre-confiadas: la MF se vuelve
    un delta de Dirac, el gradiente desaparece y la regla muere. Mantener un
    piso barato evita ese fallo silencioso (cf. skill `anfis-builder`).
    """

    def __init__(self, min_sigma: float = 1e-2):
        super().__init__()
        self.min_sigma = float(min_sigma)
        self.log_min = float(np.log(self.min_sigma))
        self._mf_layer = None

    def set_model(self, model):
        super().set_model(model)
        self._mf_layer = None
        for layer in model.layers:
            if isinstance(layer, GaussianMembership):
                self._mf_layer = layer
                break
        if self._mf_layer is None:
            raise ValueError("SigmaFloor: no se encontró GaussianMembership.")

    def on_train_batch_end(self, batch, logs=None):
        ls = self._mf_layer.log_sigma
        # Clip in-place vía assign — evita rebuild del grafo.
        ls.assign(tf.maximum(ls, self.log_min))


def _validate_dims(n_inputs: int, n_mfs: int) -> None:
    if not (1 <= n_inputs <= 10):
        raise ValueError(f"n_inputs={n_inputs} fuera de [1, 10] — ANFIS no escala.")
    if n_mfs not in (2, 3):
        raise ValueError(f"n_mfs={n_mfs} no en {{2, 3}} (restricción del spec 03).")
    n_rules = count_rules(n_inputs, n_mfs)
    if n_rules > 6561:
        raise ValueError(
            f"n_rules={n_rules} (= {n_mfs}^{n_inputs}) supera el máximo seguro 6561."
        )


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------
def build_anfis(
    n_inputs: int,
    n_mfs: int,
    n_classes: int = 2,
    learning_rate: float = 1e-3,
    name: str = "anfis",
) -> tf.keras.Model:
    """Construye y compila un modelo ANFIS TSK de orden 1.

    Args:
        n_inputs: dimensionalidad de la entrada (típicamente PCA, en [5, 8]).
        n_mfs:    membership functions por entrada (∈ {2, 3}).
        n_classes: 2 -> head sigmoid (1 logit, BCE).
                   >2 -> head softmax (n_classes logits, CCE sparse).
        learning_rate: lr inicial del optimizador Adam.
        name: nombre del modelo Keras.

    Returns:
        `tf.keras.Model` compilado, listo para `fit`.
    """
    _validate_dims(n_inputs, n_mfs)

    n_rules = count_rules(n_inputs, n_mfs)
    # Para binario producimos 1 logit y aplicamos sigmoid; para multiclase
    # producimos n_classes logits y aplicamos softmax.
    n_outputs = 1 if n_classes == 2 else int(n_classes)

    inp = tf.keras.Input(shape=(n_inputs,), name="x")
    mu = GaussianMembership(n_inputs, n_mfs)(inp)
    w = RuleFiringStrength(n_inputs, n_mfs)(mu)
    w_norm = Normalization()(w)
    f = TSKConsequent(n_rules=n_rules, n_inputs=n_inputs, n_outputs=n_outputs)(inp)
    logits = Aggregation()([w_norm, f])

    if n_classes == 2:
        out = tf.keras.layers.Activation("sigmoid", name="prob")(logits)
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = [tf.keras.metrics.BinaryAccuracy(name="accuracy")]
    else:
        out = tf.keras.layers.Activation("softmax", name="prob")(logits)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    model = tf.keras.Model(inputs=inp, outputs=out, name=name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )
    # Atributos auxiliares para fases posteriores (init, diagnósticos, GA).
    model.n_inputs = n_inputs
    model.n_mfs = n_mfs
    model.n_rules = n_rules
    model.n_classes = n_classes
    model.n_outputs = n_outputs
    return model


# -----------------------------------------------------------------------------
# Smoke test al ejecutar el módulo directamente
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    for n_inputs, n_mfs, n_classes in [(6, 2, 2), (5, 3, 4), (8, 2, 4)]:
        m = build_anfis(n_inputs=n_inputs, n_mfs=n_mfs, n_classes=n_classes)
        x = rng.standard_normal((8, n_inputs)).astype(np.float32)
        y = m(x).numpy()
        expected_out = 1 if n_classes == 2 else n_classes
        assert y.shape == (8, expected_out), (y.shape, expected_out)
        assert not np.isnan(y).any(), "NaN en la salida del smoke test."
        print(f"OK  n_inputs={n_inputs} n_mfs={n_mfs} n_classes={n_classes} "
              f"-> rules={m.n_rules}  out_shape={y.shape}  params={m.count_params()}")

    # --- Smoke test fase 04: inicialización data-driven + SigmaFloor ---
    print("\n[Fase 04] initialize_from_data + SigmaFloor")
    for n_inputs, n_mfs in [(6, 2), (5, 3)]:
        m = build_anfis(n_inputs=n_inputs, n_mfs=n_mfs, n_classes=2)
        Z = rng.standard_normal((500, n_inputs)).astype(np.float32)
        info = initialize_from_data(m, Z, n_mfs=n_mfs, min_sigma=1e-2)

        # Centros deben coincidir con los cuantiles esperados
        q = QUANTILE_SCHEDULE[n_mfs]
        expected_c = np.quantile(Z, q, axis=0).T.astype(np.float32)
        assert np.allclose(info["centers"], expected_c), "centros != quantiles"

        # σ = (max - min) / (2 n_mfs), con floor
        ranges = Z.max(axis=0) - Z.min(axis=0)
        expected_sigma = np.maximum(ranges / (2.0 * n_mfs), 1e-2)
        assert np.allclose(info["sigma"][:, 0], expected_sigma, atol=1e-6)

        # TSK ceros
        for w in m.get_layer("tsk_consequent").get_weights():
            assert np.all(w == 0.0)

        # Cada regla dispara con prob > 0 en al menos 1% del training (proxy:
        # mean firing > 0 después de normalización).
        mu = m.get_layer("gaussian_mf")(Z).numpy()
        w_layer = m.get_layer("rule_firing")
        w_un = w_layer(mu).numpy()
        w_norm = w_un / (w_un.sum(axis=1, keepdims=True) + 1e-12)
        active_frac = (w_norm > 1e-6).mean(axis=0)
        assert (active_frac >= 0.01).all(), (
            f"alguna regla dispara < 1%: min={active_frac.min():.4f}"
        )
        print(
            f"  OK  n_inputs={n_inputs} n_mfs={n_mfs} rules={m.n_rules}  "
            f"min_active_frac={active_frac.min():.3f}  σ∈[{info['sigma'].min():.3f},"
            f"{info['sigma'].max():.3f}]"
        )

    # SigmaFloor: tras un fit corto, log_sigma >= log(min_sigma)
    m = build_anfis(n_inputs=5, n_mfs=2, n_classes=2)
    Z = rng.standard_normal((256, 5)).astype(np.float32)
    y = (rng.standard_normal(256) > 0).astype(np.float32)
    initialize_from_data(m, Z, n_mfs=2)
    # Forzar log_sigma a un valor pequeño para verificar que el callback lo eleva
    mf = m.get_layer("gaussian_mf")
    bad = np.full_like(mf.get_weights()[1], np.log(1e-5))
    mf.set_weights([mf.get_weights()[0], bad])
    cb = SigmaFloor(min_sigma=1e-2)
    m.fit(Z, y, epochs=1, batch_size=64, verbose=0, callbacks=[cb])
    ls = mf.log_sigma.numpy()
    assert (ls >= np.log(1e-2) - 1e-6).all(), f"SigmaFloor no clipeó: min={ls.min()}"
    print(f"  OK  SigmaFloor activo: log_sigma_min={ls.min():.4f} >= log(1e-2)={np.log(1e-2):.4f}")

"""
Siembra el checkpoint del notebook `hyperopt_ga_experimentos.ipynb` con los
experimentos que ya terminaron en la corrida anterior (3h que murio en DE C).

Uso:
    # Desde la raiz del repo (donde esta /data):
    python jupyter/seed_checkpoint_from_previous_run.py

Despues de correrlo, al reejecutar el notebook con kernel limpio:
    - GA (A/B/C), CMA-ES (A/B/C) y DE (A/B) se saltan via `already_done()`.
    - Solo se corren DE C + NSGA-II (A/B/C) -> deberia tardar ~20-30 min
      con el fix de popsize, en lugar de las ~3h de la corrida original.

Si prefieres reejecutar DE completo con el budget corregido (6 individuos en
lugar de los 42 que scipy usaba por bug), borra las entradas "DE" del pickle
o simplemente borra el archivo antes de reejecutar el notebook.
"""
from pathlib import Path
import pickle

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
CHECKPOINT_PATH = CHECKPOINT_DIR / "hyperopt_ga_experimentos.pkl"


def row(algo, exp, val_acc, t, p, extra):
    return {
        "algo": algo,
        "experimento": exp,
        "best_val_acc": val_acc,
        "time_s": t,
        "best_n_hidden1": p[0],
        "best_n_hidden2": p[1],
        "best_dropout1": p[2],
        "best_dropout2": p[3],
        "best_lr": p[4],
        "best_batch_size": p[5],
        "best_leaky_alpha": p[6],
        **extra,
    }


RESULTS = [
    row("GA canonico", "A. Baseline",    0.6804, 197.3,
        (734, 195, 0.18, 0.59, 4.7e-04, 64, 0.010),
        {"tournsize": 3, "cxpb": 0.7, "mutpb": 0.3}),
    row("GA canonico", "B. Exploracion", 0.6830,  81.7,
        (734, 195, 0.11, 0.58, 4.7e-04, 64, 0.010),
        {"tournsize": 2, "cxpb": 0.9, "mutpb": 0.1}),
    row("GA canonico", "C. Explotacion", 0.6833,  84.1,
        (734, 195, 0.16, 0.58, 3.8e-04, 64, 0.010),
        {"tournsize": 5, "cxpb": 0.5, "mutpb": 0.5}),

    row("CMA-ES", "A. Baseline", 0.6806, 200.7,
        (926,  68, 0.57, 0.56, 4.9e-04,  64, 0.094),
        {"sigma0": 0.3}),
    row("CMA-ES", "B. Local",    0.6769, 190.5,
        (683, 237, 0.37, 0.34, 4.3e-04, 128, 0.063),
        {"sigma0": 0.1}),
    row("CMA-ES", "C. Global",   0.6740, 162.6,
        (582,  63, 0.37, 0.28, 8.4e-04,  64, 0.000),
        {"sigma0": 0.5}),

    row("DE", "A. Baseline (dithering)", 0.6862, 2473.3,
        (144, 132, 0.50, 0.43, 6.3e-04, 64, 0.047),
        {"F": "(0.5, 1.0)", "CR": 0.7, "strategy": "best1bin"}),
    row("DE", "B. Exploit",              0.6886, 2709.3,
        (873,  95, 0.52, 0.36, 3.7e-04, 64, 0.077),
        {"F": "0.5", "CR": 0.9, "strategy": "best1bin"}),
]

payload = {
    "RESULTS": RESULTS,
    "EVAL_CACHE": {},
    "EVAL_LOG": [],
    "NSGA_FRONTS_RAW": {},
}

with open(CHECKPOINT_PATH, "wb") as f:
    pickle.dump(payload, f)

print(f"Checkpoint sembrado en {CHECKPOINT_PATH.resolve()}")
print(f"Experimentos pre-cargados: {len(RESULTS)}")
for r in RESULTS:
    print(f"  - {r['algo']:<12s} / {r['experimento']:<25s} "
          f"val_acc={r['best_val_acc']:.4f}  t={r['time_s']:7.1f}s")

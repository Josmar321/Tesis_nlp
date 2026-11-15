"""Fuga de entidades sensibles a partir de 04_errors.csv (Tabla 9 de la tesis).

Fuga completa: entidades de referencia sin ningun solapamiento con una prediccion
(error_type == "false_negative"). Fuga con frontera: fuga completa + boundary_error.
Las confusiones de tipo no cuentan como fuga: la entidad se detecta como sensible.

Uso (desde la raiz del repositorio):
    python scripts/05_fuga_entidades.py
"""
import collections
import csv
from pathlib import Path

RESULTS = Path("results")
RUNS = [
    "beto_baseline__augmented_baseline",
    "beto_mixed__augmented_50",
    "xlmr_baseline__augmented_baseline",
    "xlmr_mixed__augmented_50",
]
# Ajustar a la taxonomia de docs/taxonomia_entidades.md
DIRECTOS = {
    "NOMBRE_SUJETO_ASISTENCIA", "ID_SUJETO_ASISTENCIA", "ID_ASEGURAMIENTO",
    "ID_CONTACTO_ASISTENCIAL", "NUMERO_TELEFONO", "NUMERO_FAX",
    "CORREO_ELECTRONICO", "CALLE", "FAMILIARES_SUJETO_ASISTENCIA",
}
ERRORS_LIMIT = 1000  # valor por defecto de --errors-limit en 04_evaluate_ner.py


def pct(n, d):
    return f"{100 * n / d:.2f} %"


def main():
    print("run;fuga_completa;fuga_con_frontera;fuga_completa_directos;fuga_frontera_directos")
    for run in RUNS:
        d = RESULTS / run
        with open(d / "04_per_entity.csv", encoding="utf-8-sig") as f:
            per = {r["label"]: r for r in csv.DictReader(f)}
        with open(d / "04_errors.csv", encoding="utf-8-sig") as f:
            errors = list(csv.DictReader(f))
        if len(errors) >= ERRORS_LIMIT:
            print(f"{run}: 04_errors.csv truncado ({len(errors)} filas); se omite")
            continue

        support = {k: int(v["support"]) for k, v in per.items()}
        strict_fn = sum(int(v["fn"]) for v in per.values())
        fn = collections.Counter(e["true_label"] for e in errors if e["error_type"] == "false_negative")
        bd = collections.Counter(e["true_label"] for e in errors if e["error_type"] == "boundary_error")
        tc = sum(1 for e in errors if e["error_type"] == "type_confusion")
        # Control: cada FN estricto es omision, frontera o confusion de tipo
        assert sum(fn.values()) + sum(bd.values()) + tc == strict_fn, run

        tot, dsup = sum(support.values()), sum(support.get(k, 0) for k in DIRECTOS)
        fdir = sum(fn[k] for k in DIRECTOS)
        bdir = sum(bd[k] for k in DIRECTOS)
        print(";".join([
            run,
            f"{pct(sum(fn.values()), tot)} ({sum(fn.values())})",
            pct(sum(fn.values()) + sum(bd.values()), tot),
            f"{pct(fdir, dsup)} ({fdir})",
            pct(fdir + bdir, dsup),
        ]))


if __name__ == "__main__":
    main()

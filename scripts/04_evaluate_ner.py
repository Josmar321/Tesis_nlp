#!/usr/bin/env python3
"""04_evaluate_ner.py - Fase 4: evaluacion estricta sobre test.jsonl.

Diseno experimental de la tesis
-------------------------------
Se evaluan 3 escenarios de datos (baseline, mixto 50%, full 100%) con 2 modelos
(BETO y XLM-RoBERTa). El archivo test.jsonl es identico en los 3 escenarios y
NUNCA se uso para entrenar ni para seleccionar hiperparametros. Este script lee
EXCLUSIVAMENTE test.jsonl: no abre train.jsonl ni dev.jsonl.

Protocolo de evaluacion
-----------------------
1. Carga pesos, tokenizador y labels.json desde --model-dir.
2. Reproduce la misma ventana deslizante del entrenamiento:
   max_length=512, stride=128, is_split_into_words=True.
3. Para cada palabra toma la prediccion de su primer sub-token en la primera
   ventana que la contiene (misma regla de alineacion que 03_train_ner.py).
   Opcionalmente, --aggregation mean promedia las probabilidades de todas las
   apariciones completas de la palabra en las ventanas solapadas.
4. Repara secuencias BIO invalidas del modelo (I- huerfana -> B-) y cuenta las
   reparaciones. Las etiquetas gold se validan y NO se reparan.
5. Calcula metricas oficiales con seqeval (nivel entidad) y reporta:
   - micro, macro (etiquetas presentes en gold o prediccion, como seqeval) y weighted.
   - classification_report completo por entidad.
   - TP/FP/FN por etiqueta y conteos de spans.
   - Bootstrap opcional por documentos para intervalos de confianza del F1.
   - Analisis exploratorio de errores (falsos negativos, falsos positivos,
     confusiones de tipo y errores de frontera).

Salidas (en --output-dir)
-------------------------
- 04_eval_summary.json          : resumen completo y metricas.
- 04_classification_report.txt  : reporte seqeval legible.
- 04_classification_report.json : reporte seqeval estructurado.
- 04_per_entity.csv             : P/R/F1/soporte/TP/FP/FN por entidad.
- 04_bootstrap.json             : IC bootstrap (si --bootstrap > 0).
- 04_errors.csv                 : errores individuales (si --errors-limit > 0).
- 04_confusions.csv             : confusiones de tipo/frontera agregadas.
- 04_predictions.jsonl          : predicciones por documento (opcional).

Ejemplos
--------
# Evaluacion estandar de BETO en el escenario full
python scripts/04_evaluate_ner.py \
    --model-dir models/beto_augmented_100 \
    --data-dir data/augmented_100 \
    --output-dir results/beto_augmented_100 \
    --bootstrap 1000

# Evaluacion de XLM-R en el escenario baseline
python scripts/04_evaluate_ner.py \
    --model-dir models/xlmr_augmented_baseline \
    --data-dir data/augmented_baseline \
    --output-dir results/xlmr_augmented_baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import numpy as np
    import torch
    import transformers
    from seqeval.metrics import classification_report
    from transformers import AutoModelForTokenClassification, AutoTokenizer
except ModuleNotFoundError as exc:  # pragma: no cover - depende del entorno
    raise SystemExit(
        f"Falta la dependencia '{exc.name}'. Instala el entorno con:\n"
        "    pip install -r requirements.txt"
    ) from exc

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MAX_LENGTH = 512
DEFAULT_STRIDE = 128
DEFAULT_BATCH_SIZE = 16

# Entidades que la tesis quiere destacar en el reporte de consola.
KEY_ENTITIES = ("PAIS", "HOSPITAL", "SEXO_SUJETO_ASISTENCIA", "FECHAS")
KEY_ENTITY_PREFIXES = ("ID_",)


# ---------------------------------------------------------------------------
# Utilidades de rutas y JSON
# ---------------------------------------------------------------------------
def resolver_ruta(path: Path) -> Path:
    """Resuelve rutas relativas contra la raiz del proyecto."""
    path = path.expanduser()
    return path if path.is_absolute() else (BASE_DIR / path)


def relpath(path: Path) -> str:
    """Devuelve una ruta relativa a BASE_DIR cuando sea posible."""
    try:
        return str(path.resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def jsonable(obj: Any) -> Any:
    """Convierte tipos de NumPy a tipos serializables por json."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def write_json(path: Path, obj: Any) -> None:
    """Escribe JSON UTF-8 con indentacion y tipos normalizados."""
    path.write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Escribe CSV UTF-8 con BOM para facilitar su apertura en Excel."""
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


# ---------------------------------------------------------------------------
# Lectura y validacion estricta de BIO
# ---------------------------------------------------------------------------
def validate_bio(tags: list[str], doc_id: str) -> None:
    """Valida IOB2: toda I-X debe seguir a B-X o I-X de la misma entidad."""
    prev = "O"
    for i, tag in enumerate(tags):
        if tag == "O":
            prev = tag
            continue
        if not (tag.startswith("B-") or tag.startswith("I-")):
            raise ValueError(f"{doc_id}: etiqueta BIO invalida en posicion {i}: {tag!r}")
        entity_type = tag[2:]
        if not entity_type:
            raise ValueError(f"{doc_id}: etiqueta sin tipo en posicion {i}: {tag!r}")
        if tag.startswith("I-"):
            if prev == "O" or prev[2:] != entity_type:
                raise ValueError(
                    f"{doc_id}: I- huerfana en posicion {i}: {tag!r} (anterior={prev!r})"
                )
        prev = tag


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Lee y valida un JSONL con id, tokens y tags (o etiquetas_ner)."""
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    docs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            tokens = item.get("tokens")
            tags = item.get("tags", item.get("etiquetas_ner"))
            if not isinstance(tokens, list) or not isinstance(tags, list):
                raise ValueError(f"{path.name}:{line_num}: faltan 'tokens' o 'tags'.")
            if len(tokens) != len(tags):
                raise ValueError(
                    f"{path.name}:{line_num}: tokens={len(tokens)} != tags={len(tags)}."
                )
            if not tokens:
                raise ValueError(f"{path.name}:{line_num}: documento vacio.")
            if any((not isinstance(t, str)) or (not t) or any(c.isspace() for c in t) for t in tokens):
                raise ValueError(f"{path.name}:{line_num}: token vacio o con espacios.")

            doc_id = str(item.get("id", item.get("id_documento", f"doc_{line_num:05d}")))
            if doc_id in seen_ids:
                raise ValueError(f"{path.name}:{line_num}: id duplicado: {doc_id!r}.")
            seen_ids.add(doc_id)

            tags = [str(t) for t in tags]
            validate_bio(tags, doc_id)
            docs.append({"id": doc_id, "tokens": [str(t) for t in tokens], "tags": tags})

    if not docs:
        raise ValueError(f"{path} no contiene documentos.")
    return docs


def repair_bio(tags: list[str]) -> tuple[list[str], int]:
    """Repara predicciones BIO invalidas del modelo (I- huerfana -> B-)."""
    repaired: list[str] = []
    prev = "O"
    repairs = 0
    for tag in tags:
        if tag.startswith("I-"):
            entity_type = tag[2:]
            if prev == "O" or prev[2:] != entity_type:
                tag = "B-" + entity_type
                repairs += 1
        repaired.append(tag)
        prev = tag
    return repaired, repairs


# ---------------------------------------------------------------------------
# Spans, conteos y metricas
# ---------------------------------------------------------------------------
def tags_to_spans(tags: list[str]) -> list[tuple[int, int, str]]:
    """Convierte una secuencia IOB2 valida en spans (inicio, fin_exclusivo, tipo)."""
    spans: list[tuple[int, int, str]] = []
    start: int | None = None
    label: str | None = None

    for i, tag in enumerate(tags):
        if tag == "O":
            if start is not None:
                spans.append((start, i, label or ""))
                start, label = None, None
        elif tag.startswith("B-"):
            if start is not None:
                spans.append((start, i, label or ""))
            start, label = i, tag[2:]
        elif tag.startswith("I-"):
            entity_type = tag[2:]
            if start is None or label != entity_type:
                if start is not None:
                    spans.append((start, i, label or ""))
                start, label = i, entity_type
        else:
            raise ValueError(f"Etiqueta BIO invalida: {tag!r}")

    if start is not None:
        spans.append((start, len(tags), label or ""))
    return spans


def spans_to_counts(
    true_spans: list[tuple[int, int, str]],
    pred_spans: list[tuple[int, int, str]],
) -> dict[str, dict[str, int]]:
    """Calcula TP/FP/FN por etiqueta usando coincidencia exacta de span."""
    true_set = set(true_spans)
    pred_set = set(pred_spans)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for span in true_set & pred_set:
        counts[span[2]]["tp"] += 1
    for span in pred_set - true_set:
        counts[span[2]]["fp"] += 1
    for span in true_set - pred_set:
        counts[span[2]]["fn"] += 1
    return counts


def label_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Precision, recall y F1 para una etiqueta."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def micro_metrics(counts: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Metricas micro agregando TP/FP/FN de todas las etiquetas."""
    tp = sum(c["tp"] for c in counts.values())
    fp = sum(c["fp"] for c in counts.values())
    fn = sum(c["fn"] for c in counts.values())
    metrics = label_metrics(tp, fp, fn)
    metrics.update({"tp": tp, "fp": fp, "fn": fn})
    return metrics


def macro_metrics(counts: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Macro promedio sobre etiquetas presentes en gold o en prediccion.

    Incluye etiquetas con soporte 0 pero con falsos positivos, igual que
    seqeval, para que el macro propio coincida con el reporte oficial.
    """
    labels = [
        label
        for label, c in counts.items()
        if c["tp"] + c["fn"] > 0 or c["fp"] > 0
    ]
    if not labels:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "labels_evaluated": 0}
    per_label = [label_metrics(counts[label]["tp"], counts[label]["fp"], counts[label]["fn"]) for label in labels]
    return {
        "precision": mean(m["precision"] for m in per_label),
        "recall": mean(m["recall"] for m in per_label),
        "f1": mean(m["f1"] for m in per_label),
        "labels_evaluated": len(labels),
    }


def weighted_metrics(counts: dict[str, dict[str, int]]) -> dict[str, float]:
    """Promedio ponderado por soporte (tp + fn) de cada etiqueta."""
    total_support = sum(c["tp"] + c["fn"] for c in counts.values())
    if total_support == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    per_label = {
        label: label_metrics(c["tp"], c["fp"], c["fn"])
        for label, c in counts.items()
        if c["tp"] + c["fn"] > 0
    }
    return {
        "precision": sum(m["precision"] * (counts[label]["tp"] + counts[label]["fn"]) for label, m in per_label.items()) / total_support,
        "recall": sum(m["recall"] * (counts[label]["tp"] + counts[label]["fn"]) for label, m in per_label.items()) / total_support,
        "f1": sum(m["f1"] * (counts[label]["tp"] + counts[label]["fn"]) for label, m in per_label.items()) / total_support,
    }


def compute_seqeval_report(
    y_true: list[list[str]],
    y_pred: list[list[str]],
) -> tuple[dict[str, Any], str]:
    """Genera el classification_report de seqeval (dict y texto)."""
    try:
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        report_str = classification_report(y_true, y_pred, digits=4, zero_division=0)
    except TypeError:
        # Compatibilidad con versiones antiguas de seqeval sin zero_division.
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        report_str = classification_report(y_true, y_pred, digits=4)
    return report_dict, report_str


# ---------------------------------------------------------------------------
# Carga de modelo, tokenizador y etiquetas
# ---------------------------------------------------------------------------
def load_train_summary(model_dir: Path) -> dict[str, Any] | None:
    """Lee 03_train_summary.json si existe (para heredar max_length y stride)."""
    path = model_dir / "03_train_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_tokenizer(model_dir: Path):
    """Carga el tokenizador guardado por 03, con fallback para versiones viejas."""
    try:
        return AutoTokenizer.from_pretrained(str(model_dir), use_fast=True, add_prefix_space=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)


def load_labels(model_dir: Path, model) -> dict[int, str]:
    """Carga el mapa id->etiqueta, prefiriendo labels.json cuando es informativo."""
    config_id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    if sorted(config_id2label) != list(range(len(config_id2label))):
        raise ValueError("config.id2label no tiene indices contiguos 0..N-1.")
    if model.config.num_labels is not None and len(config_id2label) != model.config.num_labels:
        raise ValueError(
            f"config.num_labels={model.config.num_labels} != len(id2label)={len(config_id2label)}."
        )

    labels_path = model_dir / "labels.json"
    if not labels_path.exists():
        print("[aviso] No se encontro labels.json; se usa config.id2label.")
        return config_id2label

    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    json_id2label = {int(k): str(v) for k, v in payload.get("id2label", {}).items()}
    if not json_id2label and "labels" in payload:
        json_id2label = {i: str(label) for i, label in enumerate(payload["labels"])}

    if not json_id2label:
        print("[aviso] labels.json no contiene id2label ni labels; se usa config.id2label.")
        return config_id2label

    generic_config = all(value.startswith("LABEL_") for value in config_id2label.values())
    if json_id2label == config_id2label:
        return config_id2label
    if generic_config and sorted(json_id2label) == sorted(config_id2label):
        print("[aviso] config.id2label es generico; se usan las etiquetas de labels.json.")
        return json_id2label

    print("[aviso] labels.json no coincide con config.id2label; se usa config.id2label.")
    return config_id2label


def resolve_device(requested: str) -> torch.device:
    """Resuelve 'auto', 'cpu' o 'cuda[:N]' a un torch.device."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("Se solicito CUDA pero no esta disponible.")
        return torch.device(requested)
    raise SystemExit(f"Dispositivo no valido: {requested!r}")


# ---------------------------------------------------------------------------
# Ventana deslizante y prediccion por documento
# ---------------------------------------------------------------------------
@dataclass
class Window:
    """Una ventana de sub-tokens y su mapeo a palabras del documento original."""

    doc_idx: int
    window_idx: int
    input_ids: list[int]
    attention_mask: list[int]
    word_ids: list[int | None]


def build_windows(
    tokenizer,
    docs: list[dict[str, Any]],
    max_length: int,
    stride: int,
) -> tuple[list[Window], dict[str, Any]]:
    """Tokeniza con overflow y devuelve las ventanas con su word_ids."""
    encoded = tokenizer(
        [doc["tokens"] for doc in docs],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        padding=False,
    )
    mapping = encoded.pop("overflow_to_sample_mapping")

    windows: list[Window] = []
    for i, doc_idx in enumerate(mapping):
        windows.append(
            Window(
                doc_idx=int(doc_idx),
                window_idx=i,
                input_ids=list(encoded["input_ids"][i]),
                attention_mask=list(encoded["attention_mask"][i]),
                word_ids=encoded.word_ids(batch_index=i),
            )
        )

    per_doc = Counter(w.doc_idx for w in windows)
    stats = {
        "n_windows": len(windows),
        "docs_with_multiple_windows": sum(1 for c in per_doc.values() if c > 1),
        "max_windows_per_doc": max(per_doc.values()) if per_doc else 0,
        "max_length": max_length,
        "stride": stride,
    }
    return windows, stats


def predict_documents(
    model,
    tokenizer,
    windows: list[Window],
    docs: list[dict[str, Any]],
    id2label: dict[int, str],
    device: torch.device,
    batch_size: int,
    aggregation: str,
) -> tuple[list[list[str]], dict[str, Any]]:
    """Predice etiquetas por palabra y las agrega a nivel de documento.

    Regla 'first' (default): cada palabra recibe la prediccion de su primer
    sub-token en la primera ventana que la contiene. Es la misma regla que uso
    03_train_ner.py para asignar etiquetas durante el entrenamiento.

    Regla 'mean': promedia las probabilidades de todas las apariciones de la
    palabra en ventanas solapadas, omitiendo la continuacion de una palabra que
    quedo partida al final de la ventana anterior.
    """
    num_labels = len(id2label)
    prob_sum: dict[tuple[int, int], np.ndarray] = defaultdict(
        lambda: np.zeros(num_labels, dtype=np.float64)
    )
    prob_count: Counter = Counter()
    seen_words: dict[int, set[int]] = defaultdict(set)

    prev_last_word_id: int | None = None
    last_doc_idx: int | None = None
    model.eval()

    for start in range(0, len(windows), batch_size):
        batch = windows[start : start + batch_size]
        features = [
            {"input_ids": w.input_ids, "attention_mask": w.attention_mask} for w in batch
        ]
        padded = tokenizer.pad(features, padding=True, return_tensors="pt")
        padded = {key: value.to(device) for key, value in padded.items()}

        with torch.inference_mode():
            logits = model(**padded).logits
        if logits.shape[-1] != num_labels:
            raise ValueError(
                f"El modelo devolvio {logits.shape[-1]} clases; se esperaban {num_labels}."
            )
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()

        for j, window in enumerate(batch):
            if window.doc_idx != last_doc_idx:
                prev_last_word_id = None
                last_doc_idx = window.doc_idx

            non_null = [wid for wid in window.word_ids if wid is not None]
            first_word_id = non_null[0] if non_null else None
            last_word_id = non_null[-1] if non_null else None
            skip_first = (
                aggregation == "mean"
                and first_word_id is not None
                and first_word_id == prev_last_word_id
            )

            first_word_seen = False
            prev_word_id: int | None = None
            for pos, word_id in enumerate(window.word_ids):
                if word_id is None:
                    prev_word_id = None
                    continue
                if word_id != prev_word_id:
                    is_first_word = word_id == first_word_id and not first_word_seen
                    first_word_seen = True
                    key = (window.doc_idx, word_id)

                    if aggregation == "first":
                        if word_id not in seen_words[window.doc_idx]:
                            prob_sum[key] += probs[j, pos].astype(np.float64)
                            prob_count[key] += 1
                            seen_words[window.doc_idx].add(word_id)
                    else:
                        if not (is_first_word and skip_first):
                            prob_sum[key] += probs[j, pos].astype(np.float64)
                            prob_count[key] += 1
                prev_word_id = word_id

            prev_last_word_id = last_word_id

    pred_tags: list[list[str]] = []
    missing_words = 0
    bio_repairs = 0
    for doc_idx, doc in enumerate(docs):
        tags: list[str] = []
        for word_id in range(len(doc["tokens"])):
            key = (doc_idx, word_id)
            if prob_count.get(key, 0) == 0:
                tags.append("O")
                missing_words += 1
                continue
            averaged = prob_sum[key] / prob_count[key]
            tags.append(id2label[int(np.argmax(averaged))])

        tags, repairs = repair_bio(tags)
        bio_repairs += repairs
        pred_tags.append(tags)

    stats = {
        "aggregation": aggregation,
        "missing_word_predictions": missing_words,
        "bio_repairs": bio_repairs,
        "n_words": sum(len(doc["tokens"]) for doc in docs),
        "n_windows": len(windows),
    }
    return pred_tags, stats


# ---------------------------------------------------------------------------
# Bootstrap por documentos y analisis de errores
# ---------------------------------------------------------------------------
def bootstrap_ci(
    doc_spans: list[tuple[list[tuple[int, int, str]], list[tuple[int, int, str]]]],
    n_iter: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    """IC bootstrap percentil para micro-F1 y macro-F1 remuestreando documentos."""
    rng = np.random.default_rng(seed)
    n_docs = len(doc_spans)
    samples: dict[str, list[float]] = {"micro": [], "macro": []}

    for _ in range(n_iter):
        indices = rng.integers(0, n_docs, size=n_docs)
        counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        for idx in indices:
            true_spans, pred_spans = doc_spans[int(idx)]
            true_set = set(true_spans)
            pred_set = set(pred_spans)
            for span in true_set & pred_set:
                counts[span[2]]["tp"] += 1
            for span in pred_set - true_set:
                counts[span[2]]["fp"] += 1
            for span in true_set - pred_set:
                counts[span[2]]["fn"] += 1
        samples["micro"].append(micro_metrics(counts)["f1"])
        samples["macro"].append(macro_metrics(counts)["f1"])

    alpha = (1 - confidence / 100) / 2
    result: dict[str, Any] = {"n_iter": n_iter, "seed": seed, "confidence": confidence}
    for key, values in samples.items():
        arr = np.asarray(values, dtype=np.float64)
        low, high = np.percentile(arr, [100 * alpha, 100 * (1 - alpha)])
        result[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "ci_low": float(low),
            "ci_high": float(high),
        }
    return result


def _error_row(
    doc: dict[str, Any],
    error_type: str,
    true_span: tuple[int, int, str] | None,
    pred_span: tuple[int, int, str] | None,
) -> dict[str, Any]:
    """Construye una fila de error con texto y contexto legibles."""
    def span_text(span: tuple[int, int, str] | None) -> str:
        if span is None:
            return ""
        return " ".join(doc["tokens"][span[0] : span[1]])

    starts = [span[0] for span in (true_span, pred_span) if span is not None]
    ends = [span[1] for span in (true_span, pred_span) if span is not None]
    context_start = max(0, min(starts) - 6)
    context_end = min(len(doc["tokens"]), max(ends) + 6)

    return {
        "doc_id": doc["id"],
        "error_type": error_type,
        "true_label": true_span[2] if true_span else "O",
        "pred_label": pred_span[2] if pred_span else "O",
        "true_text": span_text(true_span),
        "pred_text": span_text(pred_span),
        "true_start": true_span[0] if true_span else -1,
        "pred_start": pred_span[0] if pred_span else -1,
        "context": " ".join(doc["tokens"][context_start:context_end]),
    }


def error_analysis(
    docs: list[dict[str, Any]],
    y_true: list[list[str]],
    y_pred: list[list[str]],
    limit: int,
) -> list[dict[str, Any]]:
    """Analisis exploratorio de errores a nivel de entidad (no es la metrica oficial)."""
    rows: list[dict[str, Any]] = []

    for doc_idx, doc in enumerate(docs):
        true_spans = tags_to_spans(y_true[doc_idx])
        pred_spans = tags_to_spans(y_pred[doc_idx])
        true_set = set(true_spans)
        pred_set = set(pred_spans)
        matched = true_set & pred_set
        used_pred: set[tuple[int, int, str]] = set()

        for true_span in true_spans:
            if true_span in matched:
                continue
            overlapping = [
                pred_span
                for pred_span in pred_spans
                if max(true_span[0], pred_span[0]) < min(true_span[1], pred_span[1])
            ]
            if not overlapping:
                rows.append(_error_row(doc, "false_negative", true_span, None))
                continue

            same_boundary = [
                pred_span
                for pred_span in overlapping
                if pred_span[0] == true_span[0] and pred_span[1] == true_span[1]
            ]
            if same_boundary:
                pred_span = same_boundary[0]
                rows.append(_error_row(doc, "type_confusion", true_span, pred_span))
                used_pred.add(pred_span)
            else:
                pred_span = max(
                    overlapping,
                    key=lambda p: min(true_span[1], p[1]) - max(true_span[0], p[0]),
                )
                rows.append(_error_row(doc, "boundary_error", true_span, pred_span))
                used_pred.add(pred_span)

        for pred_span in pred_spans:
            if pred_span in matched or pred_span in used_pred:
                continue
            rows.append(_error_row(doc, "false_positive", None, pred_span))

    rows.sort(key=lambda row: (row["doc_id"], row["true_start"], row["pred_start"]))
    return rows[:limit] if limit > 0 else rows


# ---------------------------------------------------------------------------
# CLI y flujo principal
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directorio local con los pesos, tokenizador y labels.json guardados por 03.",
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/augmented_baseline"),
        help="Escenario que contiene test.jsonl (default: data/augmented_baseline).",
    )
    ap.add_argument(
        "--test-file",
        type=Path,
        default=None,
        help="Ruta explicita al test.jsonl. Si se omite, usa <data-dir>/test.jsonl.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directorio de salida (default: results/<modelo>__<escenario>).",
    )
    ap.add_argument("--run-name", default=None, help="Nombre corto del experimento para los reportes.")
    ap.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Longitud maxima de ventana (default: la de 03_train_summary.json o 512).",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Solapamiento entre ventanas (default: el de 03_train_summary.json o 128).",
    )
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Lote de inferencia.")
    ap.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu' o 'cuda[:N]' (default: auto).",
    )
    ap.add_argument(
        "--aggregation",
        choices=("first", "mean"),
        default="first",
        help="'first' replica la alineacion del entrenamiento; 'mean' promedia ventanas solapadas.",
    )
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Iteraciones de bootstrap por documentos (0 = desactivado).",
    )
    ap.add_argument(
        "--bootstrap-confidence",
        type=float,
        default=95.0,
        help="Nivel de confianza del IC bootstrap (default: 95).",
    )
    ap.add_argument("--bootstrap-seed", type=int, default=42, help="Semilla del bootstrap.")
    ap.add_argument(
        "--errors-limit",
        type=int,
        default=1000,
        help="Maximo de filas en 04_errors.csv (0 = desactivado).",
    )
    ap.add_argument(
        "--limit-docs",
        type=int,
        default=0,
        help="Evalua solo los primeros N documentos (0 = todos; util para smoke tests).",
    )
    ap.add_argument(
        "--save-predictions",
        action="store_true",
        help="Guarda 04_predictions.jsonl con las etiquetas gold y predichas por documento.",
    )
    ap.add_argument(
        "--allow-non-test",
        action="store_true",
        help="Permite evaluar un archivo que no se llame test.jsonl (ej. gold set peruano).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise SystemExit("--batch-size debe ser > 0.")
    if args.bootstrap < 0:
        raise SystemExit("--bootstrap debe ser >= 0.")
    if not 0 < args.bootstrap_confidence < 100:
        raise SystemExit("--bootstrap-confidence debe estar entre 0 y 100.")
    if args.errors_limit < 0:
        raise SystemExit("--errors-limit debe ser >= 0.")
    if args.limit_docs < 0:
        raise SystemExit("--limit-docs debe ser >= 0.")

    model_dir = resolver_ruta(args.model_dir)
    data_dir = resolver_ruta(args.data_dir)
    test_path = resolver_ruta(args.test_file) if args.test_file is not None else data_dir / "test.jsonl"

    if not args.allow_non_test and test_path.name != "test.jsonl":
        raise SystemExit(
            "Por seguridad, la evaluacion oficial solo acepta un archivo llamado test.jsonl. "
            "Usa --allow-non-test si evaluas un gold set peruano con otro nombre."
        )
    if not model_dir.exists():
        raise SystemExit(f"No existe --model-dir: {model_dir}")
    if args.test_file is None and not data_dir.exists():
        raise SystemExit(f"No existe --data-dir: {data_dir}")
    if not test_path.exists():
        raise SystemExit(f"No existe el archivo de test: {test_path}")

    train_summary = load_train_summary(model_dir)
    max_length = (
        args.max_length
        if args.max_length is not None
        else int((train_summary or {}).get("max_length", DEFAULT_MAX_LENGTH))
    )
    stride = (
        args.stride
        if args.stride is not None
        else int((train_summary or {}).get("stride", DEFAULT_STRIDE))
    )
    if max_length <= 0:
        raise SystemExit("--max-length debe ser > 0.")
    if stride < 0 or stride >= max_length:
        raise SystemExit("--stride debe cumplir 0 <= stride < max_length.")

    scenario_name = data_dir.name if test_path.parent == data_dir else test_path.parent.name
    run_name = args.run_name or f"{model_dir.name}__{scenario_name}"
    run_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_name) or "run"
    output_dir = (
        resolver_ruta(args.output_dir)
        if args.output_dir is not None
        else BASE_DIR / "results" / run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    print("=" * 80)
    print("FASE 4: EVALUACION ESTRICTA NER CLINICO")
    print("=" * 80)
    print(f"  Modelo          : {model_dir}")
    print(f"  Datos (test)    : {test_path}")
    print(f"  Salida          : {output_dir}")
    print(f"  Dispositivo     : {device}")
    print(f"  Ventana         : max_length={max_length} | stride={stride}")
    print(f"  Agregacion      : {args.aggregation}")
    print("=" * 80)

    tokenizer = load_tokenizer(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(model_max, int) and 0 < model_max < 1_000_000 and max_length > model_max:
        raise SystemExit(
            f"--max-length={max_length} excede el maximo del tokenizador ({model_max})."
        )

    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    model.to(device)
    id2label = load_labels(model_dir, model)

    docs = read_jsonl(test_path)
    if args.limit_docs > 0 and args.limit_docs < len(docs):
        print(
            f"[aviso] --limit-docs={args.limit_docs}: evaluacion PARCIAL sobre "
            f"{args.limit_docs} de {len(docs)} documentos. No usar para resultados finales."
        )
        docs = docs[: args.limit_docs]
    test_tags = {tag for doc in docs for tag in doc["tags"]}
    model_tags = set(id2label.values())
    missing_labels = sorted(test_tags - model_tags)
    if missing_labels:
        raise SystemExit(
            "El test contiene etiquetas que el modelo no conoce: " + ", ".join(missing_labels)
        )
    unused_labels = sorted(model_tags - test_tags)

    if train_summary and train_summary.get("scenario_data_dir"):
        trained_on = resolver_ruta(Path(str(train_summary["scenario_data_dir"])))
        try:
            same_scenario = trained_on.resolve() == data_dir.resolve()
        except OSError:
            same_scenario = False
        if not same_scenario:
            print(
                f"[aviso] El modelo fue entrenado con {trained_on}; "
                f"se evalua con test de {data_dir}."
            )

    windows, window_stats = build_windows(tokenizer, docs, max_length, stride)
    print(f"  Documentos test : {len(docs)}")
    print(f"  Tokens          : {sum(len(doc['tokens']) for doc in docs)}")
    print(
        f"  Ventanas        : {window_stats['n_windows']} "
        f"(docs multi-ventana: {window_stats['docs_with_multiple_windows']})"
    )

    y_pred, pred_stats = predict_documents(
        model=model,
        tokenizer=tokenizer,
        windows=windows,
        docs=docs,
        id2label=id2label,
        device=device,
        batch_size=args.batch_size,
        aggregation=args.aggregation,
    )
    y_true = [doc["tags"] for doc in docs]

    report_dict, report_str = compute_seqeval_report(y_true, y_pred)

    doc_spans: list[tuple[list[tuple[int, int, str]], list[tuple[int, int, str]]]] = []
    all_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for true_tags, pred_tags_doc in zip(y_true, y_pred):
        true_spans = tags_to_spans(true_tags)
        pred_spans = tags_to_spans(pred_tags_doc)
        doc_spans.append((true_spans, pred_spans))
        for label, counts in spans_to_counts(true_spans, pred_spans).items():
            for key in ("tp", "fp", "fn"):
                all_counts[label][key] += counts[key]

    report_entities = {
        key: value
        for key, value in report_dict.items()
        if key not in ("micro avg", "macro avg", "weighted avg", "accuracy")
    }
    per_entity: dict[str, dict[str, Any]] = {}
    for label in sorted(set(all_counts) | set(report_entities)):
        counts = all_counts.get(label, {"tp": 0, "fp": 0, "fn": 0})
        local = label_metrics(counts["tp"], counts["fp"], counts["fn"])
        seq = report_entities.get(label, {})
        per_entity[label] = {
            "support": counts["tp"] + counts["fn"],
            "predicted": counts["tp"] + counts["fp"],
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
            "precision": float(seq.get("precision", local["precision"])),
            "recall": float(seq.get("recall", local["recall"])),
            "f1": float(seq.get("f1-score", local["f1"])),
        }

    micro = micro_metrics(all_counts)
    macro = macro_metrics(all_counts)
    weighted = weighted_metrics(all_counts)

    seq_micro = report_dict.get("micro avg", {})
    seq_macro = report_dict.get("macro avg", {})
    seq_weighted = report_dict.get("weighted avg", {})
    seqeval_summary = {
        "micro": {
            "precision": float(seq_micro.get("precision", 0.0)),
            "recall": float(seq_micro.get("recall", 0.0)),
            "f1": float(seq_micro.get("f1-score", 0.0)),
            "support": int(seq_micro.get("support", 0)),
        },
        "macro": {
            "precision": float(seq_macro.get("precision", 0.0)),
            "recall": float(seq_macro.get("recall", 0.0)),
            "f1": float(seq_macro.get("f1-score", 0.0)),
            "support": int(seq_macro.get("support", 0)),
        },
        "weighted": {
            "precision": float(seq_weighted.get("precision", 0.0)),
            "recall": float(seq_weighted.get("recall", 0.0)),
            "f1": float(seq_weighted.get("f1-score", 0.0)),
            "support": int(seq_weighted.get("support", 0)),
        },
    }
    for name, local_metrics in (("micro", micro), ("macro", macro), ("weighted", weighted)):
        diff = abs(local_metrics["f1"] - seqeval_summary[name]["f1"])
        if diff > 1e-6:
            print(
                f"[aviso] {name}-F1 span={local_metrics['f1']:.6f} vs seqeval="
                f"{seqeval_summary[name]['f1']:.6f}; revisar alineacion."
            )

    key_labels = [label for label in KEY_ENTITIES if label in per_entity]
    key_labels += [
        label
        for label in sorted(per_entity)
        if any(label.startswith(prefix) for prefix in KEY_ENTITY_PREFIXES)
        and label not in key_labels
    ]
    key_entities = {label: per_entity[label] for label in key_labels}

    summary = {
        "run_name": run_name,
        "model_dir": relpath(model_dir),
        "model_name": str(getattr(model.config, "_name_or_path", "") or ""),
        "data_dir": relpath(data_dir),
        "test_file": relpath(test_path),
        "scenario": data_dir.name if test_path.parent == data_dir else test_path.parent.name,
        "n_docs": len(docs),
        "n_tokens": sum(len(doc["tokens"]) for doc in docs),
        "n_entities_true": sum(len(true_spans) for true_spans, _ in doc_spans),
        "n_entities_pred": sum(len(pred_spans) for _, pred_spans in doc_spans),
        "micro": micro,
        "macro": macro,
        "weighted": weighted,
        "seqeval": seqeval_summary,
        "per_entity": per_entity,
        "key_entities": key_entities,
        "label_coverage": {
            "labels_in_test": sorted(test_tags),
            "labels_in_model_not_in_test": unused_labels,
        },
        "prediction": pred_stats,
        "windows": window_stats,
        "bootstrap": None,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(device),
            "platform": platform.platform(),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "04_eval_summary.json", summary)
    (output_dir / "04_classification_report.txt").write_text(report_str, encoding="utf-8")
    write_json(output_dir / "04_classification_report.json", report_dict)

    per_entity_rows = [{"label": label, **metrics} for label, metrics in per_entity.items()]
    write_csv(
        output_dir / "04_per_entity.csv",
        per_entity_rows,
        ["label", "support", "predicted", "tp", "fp", "fn", "precision", "recall", "f1"],
    )

    print("\nMetricas oficiales (seqeval, nivel entidad):")
    print(
        f"  Micro    P={seqeval_summary['micro']['precision']:.4f} "
        f"R={seqeval_summary['micro']['recall']:.4f} "
        f"F1={seqeval_summary['micro']['f1']:.4f}"
    )
    print(
        f"  Macro    P={seqeval_summary['macro']['precision']:.4f} "
        f"R={seqeval_summary['macro']['recall']:.4f} "
        f"F1={seqeval_summary['macro']['f1']:.4f} "
        f"({macro['labels_evaluated']} etiquetas evaluadas)"
    )
    print(
        f"  Weighted P={seqeval_summary['weighted']['precision']:.4f} "
        f"R={seqeval_summary['weighted']['recall']:.4f} "
        f"F1={seqeval_summary['weighted']['f1']:.4f}"
    )

    print("\nEntidades clave:")
    print(f"  {'Etiqueta':38s} {'P':>7s} {'R':>7s} {'F1':>7s} {'Soporte':>8s}")
    for label in key_labels:
        metrics = per_entity[label]
        print(
            f"  {label:38s} {metrics['precision']:7.4f} {metrics['recall']:7.4f} "
            f"{metrics['f1']:7.4f} {metrics['support']:8d}"
        )

    if args.bootstrap > 0:
        print(f"\nBootstrap: {args.bootstrap} iteraciones (seed={args.bootstrap_seed})...")
        boot = bootstrap_ci(
            doc_spans=doc_spans,
            n_iter=args.bootstrap,
            seed=args.bootstrap_seed,
            confidence=args.bootstrap_confidence,
        )
        summary["bootstrap"] = boot
        write_json(output_dir / "04_eval_summary.json", summary)
        write_json(output_dir / "04_bootstrap.json", boot)
        print(
            f"  Micro F1 IC {args.bootstrap_confidence:.0f}%: "
            f"[{boot['micro']['ci_low']:.4f}, {boot['micro']['ci_high']:.4f}]"
        )
        print(
            f"  Macro F1 IC {args.bootstrap_confidence:.0f}%: "
            f"[{boot['macro']['ci_low']:.4f}, {boot['macro']['ci_high']:.4f}]"
        )

    if args.errors_limit > 0:
        # Se calculan todos los errores para que las confusiones no queden truncadas;
        # el CSV respeta --errors-limit para no generar archivos gigantes.
        errors_all = error_analysis(docs, y_true, y_pred, 0)
        errors = errors_all[: args.errors_limit]
        write_csv(
            output_dir / "04_errors.csv",
            errors,
            [
                "doc_id",
                "error_type",
                "true_label",
                "pred_label",
                "true_text",
                "pred_text",
                "true_start",
                "pred_start",
                "context",
            ],
        )
        confusions = Counter(
            (row["true_label"], row["pred_label"])
            for row in errors_all
            if row["error_type"] in ("type_confusion", "boundary_error")
        )
        confusion_rows = [
            {"true_label": key[0], "pred_label": key[1], "count": count}
            for key, count in confusions.most_common()
        ]
        write_csv(
            output_dir / "04_confusions.csv",
            confusion_rows,
            ["true_label", "pred_label", "count"],
        )
        summary["error_analysis"] = {
            "total": len(errors_all),
            "written": len(errors),
            "by_type": dict(Counter(row["error_type"] for row in errors_all)),
        }
        write_json(output_dir / "04_eval_summary.json", summary)
        print(
            f"[ok] Errores: {len(errors)} de {len(errors_all)} filas en 04_errors.csv "
            f"(confusiones calculadas sobre el total)"
        )

    if args.save_predictions:
        with (output_dir / "04_predictions.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for doc, pred_tags_doc in zip(docs, y_pred):
                fh.write(
                    json.dumps(
                        {
                            "id": doc["id"],
                            "tokens": doc["tokens"],
                            "true_tags": doc["tags"],
                            "pred_tags": pred_tags_doc,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"[ok] Predicciones: {output_dir / '04_predictions.jsonl'}")

    print(f"\n[ok] Resumen          : {output_dir / '04_eval_summary.json'}")
    print(f"[ok] Reporte          : {output_dir / '04_classification_report.txt'}")
    print(f"[ok] Por entidad      : {output_dir / '04_per_entity.csv'}")


if __name__ == "__main__":
    main()

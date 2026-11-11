#!/usr/bin/env python3
"""03_train_ner.py - Fase 3: Entrenamiento de modelos Transformer para NER Clinico.

Diseno Experimental (Tesis de Ingenieria de Sistemas):
    Entrena modelos basados en arquitecturas Transformer (BETO y XLM-RoBERTa)
    sobre los 3 escenarios del corpus SPACCC_MEDDOCAN adaptado al contexto peruano:
      1. Baseline (0%):   data/augmented_baseline
      2. Mixto (50%):     data/augmented_50
      3. Full (100%):     data/augmented_100

    Protocolo estricto contra Data Leakage:
      - Este script consume UNICAMENTE `train.jsonl` y `dev.jsonl` desde `--data-dir`.
      - El archivo `test.jsonl` queda reservado exclusivamente para `04_evaluate_ner.py`.

    Alineacion de Sub-tokens y Ventana Deslizante:
      - Usa `is_split_into_words=True` y `word_ids()` para mapear etiquetas BIO
        al primer sub-token de cada palabra, asignando -100 (IGNORE_INDEX) a los
        sub-tokens subsiguientes y a los tokens especiales ([CLS]/<s>, [SEP]/</s>, PAD).
      - Los documentos clinicos superan frecuentemente los 512 sub-tokens. Se activa
        `return_overflowing_tokens=True` con `stride=128` para dividir documentos
        largos en ventanas solapadas de hasta 512 sub-tokens sin perder entidades
        al final de la historia clinica.

Ejemplos de uso:
    # 1. Smoke-test rapido (50 docs de train/dev, 1 epoca)
    python scripts/03_train_ner.py \
        --data-dir data/augmented_baseline \
        --output-dir models/smoke_beto \
        --model-name dccuchile/bert-base-spanish-wwm-cased \
        --smoke-test

    # 2. Entrenamiento completo con BETO (Escenario Full 100%)
    python scripts/03_train_ner.py \
        --data-dir data/augmented_100 \
        --output-dir models/beto_augmented_100 \
        --model-name dccuchile/bert-base-spanish-wwm-cased \
        --epochs 5 --batch-size 16

    # 3. Entrenamiento completo con XLM-RoBERTa (Escenario Mixto 50%)
    python scripts/03_train_ner.py \
        --data-dir data/augmented_50 \
        --output-dir models/xlmr_augmented_50 \
        --model-name xlm-roberta-base \
        --epochs 5 --batch-size 16
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

BASE_DIR = Path(__file__).resolve().parent.parent
IGNORE_INDEX = -100
MAX_LENGTH = 512
DOC_STRIDE = 128
SMOKE_TEST_DOCS = 50


# ---------------------------------------------------------------------------
# Utilidades de E/S y rutas
# ---------------------------------------------------------------------------
def resolver_ruta(p: Path) -> Path:
    """Resuelve rutas relativas respecto a la raiz del proyecto (BASE_DIR)."""
    return p if p.is_absolute() else (BASE_DIR / p)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Lee un archivo .jsonl validando que contenga `id`, `tokens` y `tags`."""
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo requerido: {path}")

    docs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            tokens = item.get("tokens")
            tags = item.get("tags")
            if not isinstance(tokens, list) or not isinstance(tags, list):
                raise ValueError(f"{path.name}:{line_num}: 'tokens' y 'tags' deben ser listas.")
            if len(tokens) != len(tags):
                raise ValueError(
                    f"{path.name}:{line_num}: desalineacion tokens ({len(tokens)}) != tags ({len(tags)})."
                )
            docs.append(
                {
                    "id": str(item.get("id", f"doc_{line_num:05d}")),
                    "tokens": [str(t) for t in tokens],
                    "tags": [str(g) for g in tags],
                }
            )

    if not docs:
        raise ValueError(f"El archivo {path} esta vacio.")
    return docs


def build_label_list(
    *doc_splits: list[dict[str, Any]],
) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """Construye el vocabulario ordenado de etiquetas BIO garantizando 'O' en el indice 0."""
    unique_tags = {tag for split in doc_splits for doc in split for tag in doc["tags"]}
    unique_tags.discard("O")
    labels = ["O"] + sorted(unique_tags)
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return labels, label2id, id2label


# ---------------------------------------------------------------------------
# Tokenizacion, Ventana Deslizante y Realineacion Matematica BIO
# ---------------------------------------------------------------------------
def load_ner_tokenizer(model_name_or_path: str):
    """Carga AutoTokenizer habilitando `add_prefix_space=True` para modelos RoBERTa/BPE."""
    try:
        return AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
            add_prefix_space=True,
        )
    except TypeError:
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)


def tokenize_and_align_labels(
    examples: dict[str, list[Any]],
    tokenizer,
    label2id: dict[str, int],
    max_length: int = MAX_LENGTH,
    stride: int = DOC_STRIDE,
) -> dict[str, list[Any]]:
    """Realinea etiquetas BIO a nivel de palabra con sub-tokens (WordPiece / SentencePiece).

    Reglas matematicas de asignacion para cada sub-token k en la ventana w:
      1. Si word_ids(w)[k] is None (tokens especiales [CLS]/<s>, [SEP]/</s>, [PAD]):
         -> label_id = -100 (IGNORE_INDEX)
      2. Si word_ids(w)[k] == word_ids(w)[k - 1] (sub-token interno de la misma palabra):
         -> label_id = -100 (solo el primer sub-token representa a la palabra)
      3. Si la ventana w > 0 del mismo documento y el sub-token cae dentro de la zona
         de contexto solapado (`stride`) ya cubierta por la ventana w - 1:
         -> label_id = -100 (evita entrenar/evaluar dos veces la misma palabra)
      4. En caso contrario (primer sub-token no solapado de la palabra j = word_ids(w)[k]):
         -> label_id = label2id[tags[doc_idx][j]]
    """
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        padding=False,  # El padding dinamico por lote lo realiza DataCollatorForTokenClassification
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    all_labels: list[list[int]] = []

    # Rastrearemos hasta que `word_id` ya fue etiquetado en la ventana previa del mismo documento
    prev_doc_idx: int | None = None
    max_word_id_seen: int = -1

    for window_idx, doc_idx in enumerate(sample_mapping):
        tags = examples["tags"][doc_idx]
        word_ids = tokenized.word_ids(batch_index=window_idx)

        if doc_idx != prev_doc_idx:
            max_word_id_seen = -1
            prev_doc_idx = doc_idx

        label_ids: list[int] = []
        previous_word_id: int | None = None
        window_max_word_id: int = max_word_id_seen

        for word_id in word_ids:
            if word_id is None:
                # Regla 1: Tokens especiales ([CLS], [SEP], etc.)
                label_ids.append(IGNORE_INDEX)
            elif word_id != previous_word_id:
                if word_id <= max_word_id_seen:
                    # Regla 3: Palabra de contexto en el solapamiento (stride) ya cubierta antes
                    label_ids.append(IGNORE_INDEX)
                else:
                    # Regla 4: Primer sub-token efectivo de la palabra -> recibe etiqueta BIO
                    tag_str = tags[word_id]
                    label_ids.append(label2id.get(tag_str, label2id["O"]))
                    if word_id > window_max_word_id:
                        window_max_word_id = word_id
            else:
                # Regla 2: Sub-tokens secundarios de la misma palabra
                label_ids.append(IGNORE_INDEX)

            previous_word_id = word_id

        max_word_id_seen = window_max_word_id
        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized


# ---------------------------------------------------------------------------
# Metricas seqeval durante validacion
# ---------------------------------------------------------------------------
def make_compute_metrics(id2label: dict[int, str]):
    """Construye la funcion de evaluacion seqeval (nivel entidad BIO) para Hugging Face Trainer."""

    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        true_labels: list[list[str]] = []
        true_predictions: list[list[str]] = []

        for pred_row, label_row in zip(predictions, labels):
            doc_true: list[str] = []
            doc_pred: list[str] = []
            for pred_id, label_id in zip(pred_row, label_row):
                if label_id == IGNORE_INDEX:
                    continue
                doc_true.append(id2label[int(label_id)])
                doc_pred.append(id2label[int(pred_id)])
            true_labels.append(doc_true)
            true_predictions.append(doc_pred)

        return {
            "precision": float(precision_score(true_labels, true_predictions, zero_division=0)),
            "recall": float(recall_score(true_labels, true_predictions, zero_division=0)),
            "f1": float(f1_score(true_labels, true_predictions, zero_division=0)),
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# CLI y flujo principal de entrenamiento
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/augmented_baseline"),
        help="Directorio del escenario que contiene train.jsonl y dev.jsonl (default: data/augmented_baseline).",
    )
    ap.add_argument(
        "--output-dir",
        "--output",
        dest="output_dir",
        type=Path,
        default=Path("models/ner_model"),
        help="Directorio donde se guardaran los pesos finales y el resumen (default: models/ner_model).",
    )
    ap.add_argument(
        "--model-name",
        "--model",
        dest="model_name",
        default="dccuchile/bert-base-spanish-wwm-cased",
        help="Checkpoint en Hugging Face Hub (ej. 'dccuchile/bert-base-spanish-wwm-cased' o 'xlm-roberta-base').",
    )
    ap.add_argument(
        "--epochs",
        type=float,
        default=5.0,
        help="Numero de epocas de entrenamiento (default: 5.0; se fuerza a 1.0 con --smoke-test).",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Tamano de lote por dispositivo para train y eval (default: 16).",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Tasa de aprendizaje inicial para AdamW (default: 2e-5).",
    )
    ap.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay para regularizacion L2 (default: 0.01).",
    )
    ap.add_argument(
        "--max-length",
        type=int,
        default=MAX_LENGTH,
        help="Longitud maxima de ventana en sub-tokens (default: 512).",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=DOC_STRIDE,
        help="Solapamiento en sub-tokens entre ventanas de documentos largos (default: 128).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla global de reproducibilidad (default: 42).",
    )
    ap.add_argument(
        "--smoke-test",
        action="store_true",
        help=f"Modo rapido de verificacion: recorta train/dev a {SMOKE_TEST_DOCS} documentos y ejecuta 1 epoca.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Fijar semillas de reproducibilidad en Python, NumPy y PyTorch
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_seed(args.seed)

    data_dir = resolver_ruta(args.data_dir)
    output_dir = resolver_ruta(args.output_dir)

    train_path = data_dir / "train.jsonl"
    dev_path = data_dir / "dev.jsonl"

    train_docs = read_jsonl(train_path)
    dev_docs = read_jsonl(dev_path)

    # Construimos el mapa completo de etiquetas ANTES de recortar por smoke-test
    # para garantizar que la capa de clasificacion tenga siempre las 45 clases del corpus.
    labels, label2id, id2label = build_label_list(train_docs, dev_docs)

    epochs = 1.0 if args.smoke_test else args.epochs
    if args.smoke_test:
        print(
            f"\n[SMOKE-TEST ACTIVADO] Limitando train ({len(train_docs)} -> {min(len(train_docs), SMOKE_TEST_DOCS)}) "
            f"y dev ({len(dev_docs)} -> {min(len(dev_docs), SMOKE_TEST_DOCS)}) documentos | epocas = {epochs}"
        )
        train_docs = train_docs[:SMOKE_TEST_DOCS]
        dev_docs = dev_docs[:SMOKE_TEST_DOCS]

    device_str = (
        f"CUDA ({torch.cuda.get_device_name(0)})"
        if torch.cuda.is_available()
        else "CPU (Advertencia: entrenamiento lento sin GPU)"
    )
    print("=" * 80)
    print("FASE 3: ENTRENAMIENTO NER CLINICO (SPACCC_MEDDOCAN -> PERU)")
    print("=" * 80)
    print(f"  Dispositivo       : {device_str}")
    print(f"  Modelo base       : {args.model_name}")
    print(f"  Escenario (datos) : {data_dir}")
    print(f"  Directorio salida : {output_dir}")
    print(f"  Documentos        : train={len(train_docs)} | dev={len(dev_docs)}")
    print(f"  Etiquetas BIO     : {len(labels)} clases")
    print(f"  Hiperparametros   : epochs={epochs} | batch_size={args.batch_size} | lr={args.lr} | seed={args.seed}")
    print(f"  Ventana tokeniz.  : max_length={args.max_length} | stride={args.stride}")
    print("=" * 80)

    tokenizer = load_ner_tokenizer(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    train_ds = Dataset.from_list(train_docs).map(
        lambda batch: tokenize_and_align_labels(
            batch,
            tokenizer=tokenizer,
            label2id=label2id,
            max_length=args.max_length,
            stride=args.stride,
        ),
        batched=True,
        remove_columns=["id", "tokens", "tags"],
        desc="Tokenizando y alineando train",
    )
    eval_ds = Dataset.from_list(dev_docs).map(
        lambda batch: tokenize_and_align_labels(
            batch,
            tokenizer=tokenizer,
            label2id=label2id,
            max_length=args.max_length,
            stride=args.stride,
        ),
        batched=True,
        remove_columns=["id", "tokens", "tags"],
        desc="Tokenizando y alineando dev",
    )

    print(
        f"Ventanas generadas (max {args.max_length} sub-tokens): "
        f"train={len(train_ds)} (de {len(train_docs)} docs) | "
        f"dev={len(eval_ds)} (de {len(dev_docs)} docs)\n"
    )

    # Padding dinamico al maximo del batch usando IGNORE_INDEX (-100) para las etiquetas
    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        label_pad_token_id=IGNORE_INDEX,
    )

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    strategy_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"

    ta_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir / "checkpoints"),
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": epochs,
        "weight_decay": args.weight_decay,
        strategy_key: "epoch",
        "save_strategy": "no" if args.smoke_test else "epoch",
        "save_total_limit": 2,
        "logging_steps": 10 if args.smoke_test else 25,
        "seed": args.seed,
        "data_seed": args.seed,
        "fp16": torch.cuda.is_available(),
        "report_to": "none",
        "load_best_model_at_end": not args.smoke_test,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
    }

    training_args = TrainingArguments(**ta_kwargs)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": collator,
        "compute_metrics": make_compute_metrics(id2label),
    }
    # Compatibilidad con transformers >= 4.46 (processing_class) y versiones previas (tokenizer)
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    train_result = trainer.train()

    # Guardado de pesos finales, tokenizador y mapeo de etiquetas
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    (output_dir / "labels.json").write_text(
        json.dumps(
            {"labels": labels, "label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    dev_metrics = trainer.evaluate()
    print("\nMetricas de validacion (dev):")
    for key, value in dev_metrics.items():
        if isinstance(value, float):
            print(f"  {key:25s}: {value:.4f}")
        else:
            print(f"  {key:25s}: {value}")

    summary = {
        "model_name": args.model_name,
        "scenario_data_dir": str(data_dir.relative_to(BASE_DIR)) if data_dir.is_relative_to(BASE_DIR) else str(data_dir),
        "output_dir": str(output_dir.relative_to(BASE_DIR)) if output_dir.is_relative_to(BASE_DIR) else str(output_dir),
        "smoke_test": args.smoke_test,
        "seed": args.seed,
        "epochs": epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "max_length": args.max_length,
        "stride": args.stride,
        "train_docs": len(train_docs),
        "train_windows": len(train_ds),
        "dev_docs": len(dev_docs),
        "dev_windows": len(eval_ds),
        "num_labels": len(labels),
        "train_metrics": train_result.metrics,
        "dev_metrics": dev_metrics,
    }
    summary_path = output_dir / "03_train_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[ok] Modelo guardado en : {output_dir}")
    print(f"[ok] Resumen guardado en: {summary_path}")


if __name__ == "__main__":
    main()

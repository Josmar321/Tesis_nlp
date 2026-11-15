# Directorio de Modelos (`models/`)

Este directorio almacena los metadatos de configuración, vocabularios de etiquetas y resúmenes de entrenamiento generados por [03_train_ner.py](../scripts/03_train_ner.py).

---

## 1. Qué contiene actualmente el repositorio (Versionado en Git)

Para mantener el repositorio ligero y reproducible, en Git **solo se versionan los artefactos de configuración y trazabilidad** de las **6 corridas oficiales** (3 de **BETO** `dccuchile/bert-base-spanish-wwm-cased` y 3 de **XLM-RoBERTa** `xlm-roberta-base`) ejecutadas en Google Colab (GPU Tesla T4, 5 épocas, `seed=42`):

| Subdirectorio | Escenario | Dev F1 (`03_train_summary.json`) | Archivos versionados | Pesos (`model.safetensors`) |
|---|---|---:|---|---|
| `models/beto_baseline/` | `data/augmented_baseline` | `0.9401` | `03_train_summary.json`, `config.json`, `labels.json`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json` | No versionados (~440 MB); salida en `models/beto_baseline/` |
| `models/beto_mixed/` | `data/augmented_50` | `0.9363` | `03_train_summary.json`, `config.json`, `labels.json`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json` | No versionados (~440 MB); salida en `models/beto_mixed/` |
| `models/beto_full/` | `data/augmented_100` | `0.7827` | `03_train_summary.json`, `config.json`, `labels.json`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json` | No versionados (~440 MB); salida en `models/beto_full/` |
| `models/xlmr_baseline/` | `data/augmented_baseline` | `0.9264` | `03_train_summary.json`, `config.json`, `labels.json`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`, `sentencepiece.bpe.model` | No versionados (~1.1 GB); salida en `models/xlmr_baseline/` |
| `models/xlmr_mixed/` | `data/augmented_50` | `0.9309` | `03_train_summary.json`, `config.json`, `labels.json`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`, `sentencepiece.bpe.model` | No versionados (~1.1 GB); salida en `models/xlmr_mixed/` |
| `models/xlmr_full/` | `data/augmented_100` | `0.7490` | `03_train_summary.json`, `config.json`, `labels.json`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`, `sentencepiece.bpe.model` | No versionados (~1.1 GB); salida en `models/xlmr_full/` |

**Modelo seleccionado para la tesis:** **BETO, escenario Mixto (50 %)** (`models/beto_mixed/`), con Dev F1 `0.9363`, Micro-F1 `0.9363` y Macro-F1 `0.7091` en `test.jsonl`. Conserva el rendimiento global de BETO Baseline, incorpora el 50 % de reemplazo sintético peruano y evita el colapso de Full. La fuga completa es `1.89 %`, ligeramente superior a la de BETO Baseline (`1.73 %`); la selección prioriza el compromiso entre retención y adaptación, no una mejora de privacidad. El test es MEDDOCAN (español peninsular), no texto clínico peruano.

---

## 2. Qué falta y advertencias importantes sobre artefactos locales

1. **Pesos finales (`model.safetensors`) fuera del repositorio Git:**
   - El entrenamiento escribe los pesos en `models/<run>/model.safetensors` (BETO ~440 MB; XLM-R ~1.1 GB), junto con `config.json`, tokenizer y `03_train_summary.json`. Los binarios están excluidos por [`.gitignore`](../.gitignore) y no se versionan.
   - En este clon **no están presentes** los pesos de las 6 corridas. Deben conservarse en el almacenamiento externo del autor (Google Drive institucional, Hugging Face Hub privado o Git LFS). Aún no hay enlaces ni SHA256 registrados por corrida; si se añaden, deben documentarse en esta sección.
2. **ADVERTENCIA CRÍTICA sobre `models/beto_baseline/checkpoints/checkpoint-64`:**
   - En el sistema de archivos local puede existir la carpeta `models/beto_baseline/checkpoints/checkpoint-64` (ignorada por Git).
   - Ese checkpoint corresponde a una **prueba parcial/antigua de 1 sola época en CPU** (`best_metric: 0.8429`, `eval_runtime: 139 s`) y **NO corresponde** al modelo final entrenado por 5 épocas en GPU Tesla T4 reportado en `models/beto_baseline/03_train_summary.json` (`eval_f1: 0.9401`, `eval_runtime: 5.71 s`). **No debe utilizarse como modelo final.**
3. **Corrida de prueba `models/smoke_beto/`:**
   - Generada con `--smoke-test` (50 documentos, 1 época) únicamente para verificar el pipeline. Está excluida en `.gitignore` y no debe usarse para evaluación.
4. **Modelos XLM-RoBERTa (`xlm-roberta-base`) completados:**
   - Las 3 corridas de `xlm-roberta-base` (`models/xlmr_baseline`, `models/xlmr_mixed`, `models/xlmr_full`) están ejecutadas y sus metadatos/tokenizer están versionados. La matriz experimental queda en **6/6** (BETO + XLM-R × Baseline/Mixto/Full).
   - Los summaries de XLM-R registran `test_file_sha256 = d596b103...` (LF); los de BETO, `8b30b803...` (CRLF). Es el mismo `test.jsonl`: 250 documentos, 126.093 tokens y 5.661 entidades. El hash canónico adoptado es `d596b103...`; la variante CRLF se documenta en `docs/protocolo_experimental.md`.

---

## 3. Cómo regenerar los modelos (Fase 3)

En un entorno con GPU (ej. Google Colab Tesla T4), ejecuta desde la raíz del proyecto:

```bash
# 1. BETO (dccuchile/bert-base-spanish-wwm-cased)
python scripts/03_train_ner.py --data-dir data/augmented_baseline --output-dir models/beto_baseline --model-name dccuchile/bert-base-spanish-wwm-cased --epochs 5 --batch-size 16 --seed 42
python scripts/03_train_ner.py --data-dir data/augmented_50       --output-dir models/beto_mixed    --model-name dccuchile/bert-base-spanish-wwm-cased --epochs 5 --batch-size 16 --seed 42
python scripts/03_train_ner.py --data-dir data/augmented_100      --output-dir models/beto_full     --model-name dccuchile/bert-base-spanish-wwm-cased --epochs 5 --batch-size 16 --seed 42

# 2. XLM-RoBERTa (xlm-roberta-base) — 3/3 escenarios completados
python scripts/03_train_ner.py --data-dir data/augmented_baseline --output-dir models/xlmr_baseline --model-name xlm-roberta-base --epochs 5 --batch-size 16 --seed 42
python scripts/03_train_ner.py --data-dir data/augmented_50       --output-dir models/xlmr_mixed    --model-name xlm-roberta-base --epochs 5 --batch-size 16 --seed 42
python scripts/03_train_ner.py --data-dir data/augmented_100      --output-dir models/xlmr_full     --model-name xlm-roberta-base --epochs 5 --batch-size 16 --seed 42
```

# Directorio de Modelos (`models/`)

Este directorio almacena los metadatos de configuración, vocabularios de etiquetas y resúmenes de entrenamiento generados por [03_train_ner.py](file:///c:/Users/kotem/Documents/Universidad/26.2/Tesis/github/proyecto_tesis_nlp/scripts/03_train_ner.py).

---

## 1. Qué contiene actualmente el repositorio (Versionado en Git)

Para mantener el repositorio ligero y reproducible, en Git **solo se versionan los artefactos de configuración y trazabilidad** de las 3 corridas oficiales de **BETO** (`dccuchile/bert-base-spanish-wwm-cased`) ejecutadas en Google Colab (GPU Tesla T4, 5 épocas, `seed=42`):

| Subdirectorio | Escenario de datos | Dev Micro-F1 | Archivos versionados en Git | Pesos finales (`model.safetensors`) |
|---|---|---:|---|---|
| `models/beto_baseline/` | `data/augmented_baseline` | `0.9401` | `03_train_summary.json`, `config.json`, `labels.json`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json` | No versionados en Git (~440 MB) |
| `models/beto_mixed/` | `data/augmented_50` | `0.9363` | `03_train_summary.json`, `config.json`, `labels.json`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json` | No versionados en Git (~440 MB) |
| `models/beto_full/` | `data/augmented_100` | `0.7827` | `03_train_summary.json`, `config.json`, `labels.json`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json` | No versionados en Git (~440 MB) |

---

## 2. Qué falta y advertencias importantes sobre artefactos locales

1. **Pesos finales (`model.safetensors`) ausentes del repositorio Git:**
   - Los archivos binarios de pesos entrenados en Google Colab no fueron subidos a Git por superar el límite de tamaño de GitHub y están excluidos en [`.gitignore`](file:///c:/Users/kotem/Documents/Universidad/26.2/Tesis/github/proyecto_tesis_nlp/.gitignore).
   - **TODO (Decisión humana):** Definir el almacenamiento externo definitivo de los pesos finales (Google Drive institucional, Hugging Face Hub privado o Git LFS) en caso de requerir inferencia directa sin reentrenar.
2. **ADVERTENCIA CRÍTICA sobre `models/beto_baseline/checkpoints/checkpoint-64`:**
   - En el sistema de archivos local puede existir la carpeta `models/beto_baseline/checkpoints/checkpoint-64` (ignorada por Git).
   - Ese checkpoint corresponde a una **prueba parcial/antigua de 1 sola época en CPU** (`best_metric: 0.8429`, `eval_runtime: 139 s`) y **NO corresponde** al modelo final entrenado por 5 épocas en GPU Tesla T4 reportado en `models/beto_baseline/03_train_summary.json` (`eval_f1: 0.9401`, `eval_runtime: 5.71 s`). **No debe utilizarse como modelo final.**
3. **Corrida de prueba `models/smoke_beto/`:**
   - Generada con `--smoke-test` (50 documentos, 1 época) únicamente para verificar el pipeline. Está excluida en `.gitignore` y no debe usarse para evaluación.
4. **Modelos XLM-RoBERTa (`xlm-roberta-base`) pendientes:**
   - Las 3 corridas de `xlm-roberta-base` (`models/xlmr_baseline`, `models/xlmr_mixed`, `models/xlmr_full`) aún no se han ejecutado (3/6 combinaciones completadas).

---

## 3. Cómo regenerar los modelos (Fase 3)

En un entorno con GPU (ej. Google Colab Tesla T4), ejecuta desde la raíz del proyecto:

```bash
# 1. BETO (dccuchile/bert-base-spanish-wwm-cased)
python scripts/03_train_ner.py --data-dir data/augmented_baseline --output-dir models/beto_baseline --model-name dccuchile/bert-base-spanish-wwm-cased --epochs 5 --batch-size 16 --seed 42
python scripts/03_train_ner.py --data-dir data/augmented_50       --output-dir models/beto_mixed    --model-name dccuchile/bert-base-spanish-wwm-cased --epochs 5 --batch-size 16 --seed 42
python scripts/03_train_ner.py --data-dir data/augmented_100      --output-dir models/beto_full     --model-name dccuchile/bert-base-spanish-wwm-cased --epochs 5 --batch-size 16 --seed 42

# 2. XLM-RoBERTa (xlm-roberta-base) — cuando se decida ejecutar las 3 corridas restantes
python scripts/03_train_ner.py --data-dir data/augmented_baseline --output-dir models/xlmr_baseline --model-name xlm-roberta-base --epochs 5 --batch-size 16 --seed 42
python scripts/03_train_ner.py --data-dir data/augmented_50       --output-dir models/xlmr_mixed    --model-name xlm-roberta-base --epochs 5 --batch-size 16 --seed 42
python scripts/03_train_ner.py --data-dir data/augmented_100      --output-dir models/xlmr_full     --model-name xlm-roberta-base --epochs 5 --batch-size 16 --seed 42
```

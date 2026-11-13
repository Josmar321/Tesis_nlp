# Reconocimiento de Entidades Nombradas (NER) en Textos Clínicos: Adaptación al Contexto Peruano

Repositorio de investigación de tesis (Ingeniería de Sistemas) sobre **Desidentificación y Reconocimiento de Entidades Nombradas (NER) en Textos Clínicos en Español**, evaluando el impacto del reemplazo sintético progresivo de entidades culturales, geográficas y administrativas hacia el contexto hospitalario peruano sobre el corpus **MEDDOCAN** (IberLEF 2019).

---

## 1. Estado del Proyecto por Fases

| Fase | Componente | Estado | Detalle |
| :--- | :--- | :--- | :--- |
| **Fase 1** | Conversión BRAT → BIO (`01_brat_to_bio.py`) | **Cerrada y congelada** | 1,000 documentos (`train`: 500, `dev`: 250, `test`: 250), 509,964 tokens, 44 etiquetas BIO (22 tipos de entidad; 21 con soporte en `test`). |
| **Fase 2** | Aumentación Sintética Peruana (`02_augmentation.py`) | **Cerrada y congelada** | 3 escenarios (`augmented_baseline` 0%, `augmented_50` 50%, `augmented_100` 100%). `dev` y `test` permanecen intactos. |
| **Fase 3** | Entrenamiento Fine-Tuning (`03_train_ner.py`) | **Completada (6/6 corridas)** | BETO (`dccuchile/bert-base-spanish-wwm-cased`) y XLM-RoBERTa (`xlm-roberta-base`) en los 3 escenarios, GPU Colab T4. |
| **Fase 4** | Evaluación Estricta y Bootstrap (`04_evaluate_ner.py`) | **Completada (6/6 corridas)** | Evaluación `seqeval` estricta a nivel de entidad + bootstrap ($B = 1000$) para BETO y XLM-R. |
| **Fase 5** | Consolidación y Visualización (`notebooks/01_resultados.ipynb`) | **Completada (BETO + XLM-R)** | Notebook re-ejecutado con las 6 corridas, salidas embebidas, tablas `.csv`/`.tex` y figuras `.png` en `results/05_*`. |

---

## 2. Resultados Principales (BETO y XLM-R — `test.jsonl` original MEDDOCAN)

Evaluación estricta (`seqeval`, `IOB2` exact-match) sobre `test.jsonl` ($N = 250$ documentos, 5,661 entidades gold). Intervalos de confianza al 95% estimados con bootstrap por documento ($B = 1000$, `seed=42`):

| Modelo | Escenario | Dev F1 | Test Micro-P | Test Micro-R | Test Micro-F1 | IC 95% Micro-F1 | Test Macro-F1 | IC 95% Macro-F1 | Test Weighted-F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BETO** | **Baseline (0% Perú)** | 0.9401 | 0.9310 | 0.9456 | **0.9382** | `[0.9288, 0.9467]` | **0.7163** | `[0.7043, 0.7301]` | 0.9363 |
| **BETO** | **Mixto (50% Perú)** | 0.9363 | 0.9316 | 0.9410 | **0.9363** | `[0.9274, 0.9448]` | **0.7091** | `[0.6976, 0.7213]` | 0.9337 |
| **BETO** | **Full (100% Perú)** | 0.7827 | 0.7218 | 0.8419 | **0.7772** | `[0.7691, 0.7843]` | **0.5873** | `[0.5750, 0.6004]` | 0.8298 |
| **XLM-R** | **Baseline (0% Perú)** | 0.9264 | 0.9198 | 0.9406 | 0.9301 | `[0.9194, 0.9399]` | 0.6156 | `[0.6065, 0.6270]` | 0.9210 |
| **XLM-R** | **Mixto (50% Perú)** | 0.9309 | 0.9243 | 0.9406 | 0.9324 | `[0.9227, 0.9416]` | 0.6095 | `[0.6010, 0.6187]` | 0.9218 |
| **XLM-R** | **Full (100% Perú)** | 0.7490 | 0.6810 | 0.8143 | 0.7418 | `[0.7320, 0.7509]` | 0.4679 | `[0.4601, 0.4759]` | 0.7670 |

> **Interpretación metodológica:** El escenario **Mixto (50%)** conserva el rendimiento global en ambos modelos (BETO: $\Delta\text{Micro-F1} = -0.0019$; XLM-R: $+0.0023$), aunque con comportamiento heterogéneo por entidad. En BETO, `HOSPITAL` sube ligeramente de `0.7708` a `0.7762`, mientras `INSTITUCION` baja de `0.3810` a `0.2586`. El reemplazo total **Full (100%)** induce un desplazamiento de dominio severo sobre formatos regulatorios españoles (`ID_ASEGURAMIENTO`: BETO `0.9223 → 0.0052`, XLM-R `0.9875 → 0.0082`; `ID_TITULACION_PERSONAL_SANITARIO`: BETO `0.9175 → 0.0067`, XLM-R `0.9246 → 0.0036`). BETO conserva mejor los identificadores numéricos; XLM-R pierde Macro-F1 por el colapso de `ID_CONTACTO`, `NUMERO_TELEFONO` e `INSTITUCION`, no por un deterioro generalizado. La efectividad real sobre historias clínicas peruanas requerirá validación en un conjunto de prueba anotado localmente.

> **Trazabilidad del test:** el hash canónico de `test.jsonl` es `d596b103e92a38e8d7d45ea8fc9e674632aa3fd85e1a55cfca92368e129a680d` (LF). La copia Windows con CRLF produce `8b30b80360cce568a9087b754b9a38b33aa278a6be952107eab3c4ef1bc898e3`; el contenido es idéntico (250 documentos, 126,093 tokens, 5,661 entidades).

---

## 3. Estructura del Repositorio

```text
proyecto_tesis_nlp/
├── data/
│   ├── raw/                  # Corpus original MEDDOCAN en formato BRAT (.txt, .ann) [versionado en Git]
│   ├── processed/            # Corpus tokenizado en formato JSONL con esquema BIO [ignorado en Git]
│   ├── augmented_baseline/   # Escenario 0% (copia íntegra de processed) [ignorado en Git]
│   ├── augmented_50/         # Escenario Mixto 50% en train; dev y test intactos [ignorado en Git]
│   └── augmented_100/        # Escenario Full 100% en train; dev y test intactos [ignorado en Git]
├── docs/
│   ├── protocolo_experimental.md  # Diseño experimental, hiperparámetros, protocolo bootstrap y limitaciones
│   └── taxonomia_entidades.md     # Catálogo de 22 categorías PHI (21 con soporte en test), reglas de aumentación y soporte
├── models/
│   ├── README.md             # Estado de artefactos de entrenamiento y política de almacenamiento de pesos
│   ├── beto_baseline/        # Metadatos (config.json, labels.json, 03_train_summary.json)
│   ├── beto_mixed/           # Metadatos (config.json, labels.json, 03_train_summary.json)
│   ├── beto_full/            # Metadatos (config.json, labels.json, 03_train_summary.json)
│   ├── xlmr_baseline/        # Metadatos + tokenizer (config.json, labels.json, 03_train_summary.json)
│   ├── xlmr_mixed/           # Metadatos + tokenizer
│   └── xlmr_full/            # Metadatos + tokenizer
├── notebooks/
│   └── 01_resultados.ipynb   # Notebook ejecutado de Fase 5 (tablas comparativas, IC 95%, heatmaps)
├── results/
│   ├── README.md             # Inventario de métricas F4 y artefactos visuales/tabulares F5
│   ├── beto_baseline__augmented_baseline/
│   ├── beto_mixed__augmented_50/
│   ├── beto_full__augmented_100/
│   ├── xlmr_baseline__augmented_baseline/
│   ├── xlmr_mixed__augmented_50/
│   ├── xlmr_full__augmented_100/
│   └── 05_*                  # Tablas (.csv, .tex) y gráficos (.png) listos para el documento de tesis
├── scripts/
│   ├── 01_brat_to_bio.py     # Fase 1: Conversión BRAT a BIO JSONL con spaCy (es_core_news_sm)
│   ├── 02_augmentation.py    # Fase 2: Aumentación sintética peruana con preservación de integridad
│   ├── 03_train_ner.py       # Fase 3: Fine-tuning token classification con Hugging Face Trainer
│   └── 04_evaluate_ner.py    # Fase 4: Evaluación estricta seqeval + Bootstrap IC 95%
├── Bitacora_Desarrollo.md    # Registro cronológico y técnico detallado del proyecto
└── requirements.txt          # Dependencias reproducibles del pipeline
```

---

## 4. Guía Rápida de Reproducción

### Requisitos e Instalación
```bash
python -m venv venv
# En Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# En Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

### Ejecución del Pipeline por Fases
```bash
# Fase 1: Conversión BRAT -> BIO JSONL
python scripts/01_brat_to_bio.py

# Fase 2: Generación de los 3 escenarios (seed=42)
python scripts/02_augmentation.py --scenario baseline
python scripts/02_augmentation.py --scenario mixed
python scripts/02_augmentation.py --scenario full

# Fase 3: Entrenamiento (ejemplo para BETO Baseline)
python scripts/03_train_ner.py \
  --data-dir data/augmented_baseline \
  --output-dir models/beto_baseline \
  --model-name dccuchile/bert-base-spanish-wwm-cased

# Fase 4: Evaluación estricta sobre test.jsonl + Bootstrap IC 95%
python scripts/04_evaluate_ner.py \
  --model-dir models/beto_baseline \
  --data-dir data/augmented_baseline \
  --output-dir results/beto_baseline__augmented_baseline \
  --bootstrap 1000
```

---

## 5. Documentación Técnica Adicional

- [Bitacora_Desarrollo.md](Bitacora_Desarrollo.md): Historial completo de decisiones de diseño, auditorías de datos, resultados y tareas pendientes (`TODOs`).
- [docs/protocolo_experimental.md](docs/protocolo_experimental.md): Especificación formal de particiones, hiperparámetros, realineamiento de sub-tokens, protocolo bootstrap y amenazas a la validez.
- [docs/taxonomia_entidades.md](docs/taxonomia_entidades.md): Taxonomía de las 22 categorías PHI de MEDDOCAN (21 con soporte en `test`), estrategia de reemplazo sintético peruano y soporte estadístico por partición.
- [models/README.md](models/README.md): Trazabilidad de los modelos entrenados, política de exclusión de pesos binarios en Git y advertencia sobre checkpoints locales.
- [results/README.md](results/README.md): Descripción de los archivos generados por las Fases 4 y 5 (incluyendo tablas LaTeX listas para compilar).

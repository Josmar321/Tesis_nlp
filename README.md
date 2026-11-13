# Reconocimiento de Entidades Nombradas (NER) en Textos Clínicos: Adaptación al Contexto Peruano

Repositorio de investigación de tesis (Ingeniería de Sistemas) sobre **Desidentificación y Reconocimiento de Entidades Nombradas (NER) en Textos Clínicos en Español**, evaluando el impacto del reemplazo sintético progresivo de entidades culturales, geográficas y administrativas hacia el contexto hospitalario peruano sobre el corpus **MEDDOCAN** (IberLEF 2019).

---

## 1. Estado del Proyecto por Fases

| Fase | Componente | Estado | Detalle |
| :--- | :--- | :--- | :--- |
| **Fase 1** | Conversión BRAT → BIO (`01_brat_to_bio.py`) | **Cerrada y congelada** | 1,000 documentos (`train`: 500, `dev`: 250, `test`: 250), 473,538 tokens, 44 etiquetas BIO (21 tipos de entidad). |
| **Fase 2** | Aumentación Sintética Peruana (`02_augmentation.py`) | **Cerrada y congelada** | 3 escenarios (`augmented_baseline` 0%, `augmented_50` 50%, `augmented_100` 100%). `dev` y `test` permanecen intactos. |
| **Fase 3** | Entrenamiento Fine-Tuning (`03_train_ner.py`) | **Parcial (3/6 corridas)** | Completado en GPU (Colab T4) para **BETO** (`dccuchile/bert-base-spanish-wwm-cased`). Pendiente **XLM-RoBERTa** (`xlm-roberta-base`). |
| **Fase 4** | Evaluación Estricta y Bootstrap (`04_evaluate_ner.py`) | **Parcial (3/6 corridas)** | Evaluación `seqeval` estricta a nivel de entidad + bootstrap ($B = 1000$) completada para **BETO**. |
| **Fase 5** | Consolidación y Visualización (`notebooks/01_resultados.ipynb`) | **Completada (BETO)** | Notebook generalizado y ejecutado con salidas embebidas, tablas `.csv`/`.tex` y figuras `.png` en `results/05_*`. |

---

## 2. Resultados Principales (BETO — `test.jsonl` original MEDDOCAN)

Evaluación estricta (`seqeval`, `IOB2` exact-match) sobre `test.jsonl` ($N = 250$ documentos, 5,661 entidades gold). Intervalos de confianza al 95% estimados con bootstrap por documento ($B = 1000$, `seed=42`):

| Modelo | Escenario | Dev F1 | Test Micro-P | Test Micro-R | Test Micro-F1 | IC 95% Micro-F1 | Test Macro-F1 | IC 95% Macro-F1 | Test Weighted-F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BETO** | **Baseline (0% Perú)** | 0.9401 | 0.9310 | 0.9456 | **0.9382** | `[0.9288, 0.9467]` | **0.7163** | `[0.7043, 0.7301]` | 0.9363 |
| **BETO** | **Mixto (50% Perú)** | 0.9363 | 0.9316 | 0.9410 | **0.9363** | `[0.9274, 0.9448]` | **0.7091** | `[0.6976, 0.7213]` | 0.9337 |
| **BETO** | **Full (100% Perú)** | 0.7827 | 0.7218 | 0.8419 | **0.7772** | `[0.7691, 0.7843]` | **0.5873** | `[0.5750, 0.6004]` | 0.8298 |

> **Interpretación metodológica:** El escenario **Mixto (50%)** conserva el rendimiento global sobre el corpus MEDDOCAN original ($\Delta\text{Micro-F1} = -0.0019$), aunque con comportamiento heterogéneo por entidad (p. ej. `HOSPITAL` sube ligeramente de `0.7708` a `0.7762`, mientras `INSTITUCION` baja de `0.3810` a `0.2586`). En cambio, el reemplazo total **Full (100%)** induce un desplazamiento de dominio severo sobre formatos regulatorios españoles (`ID_ASEGURAMIENTO`: `0.9223 → 0.0052`; `ID_TITULACION_PERSONAL_SANITARIO`: `0.9175 → 0.0067`). La efectividad real sobre historias clínicas peruanas requerirá validación en un conjunto de prueba anotado localmente.

---

## 3. Estructura del Repositorio

```text
proyecto_tesis_nlp/
├── data/
│   ├── raw/                  # Corpus original MEDDOCAN en formato BRAT (.txt, .ann) [ignorado en Git]
│   ├── processed/            # Corpus tokenizado en formato JSONL con esquema BIO [ignorado en Git]
│   ├── augmented_baseline/   # Escenario 0% (copia íntegra de processed) [ignorado en Git]
│   ├── augmented_50/         # Escenario Mixto 50% en train; dev y test intactos [ignorado en Git]
│   └── augmented_100/        # Escenario Full 100% en train; dev y test intactos [ignorado en Git]
├── docs/
│   ├── protocolo_experimental.md  # Diseño experimental, hiperparámetros, protocolo bootstrap y limitaciones
│   └── taxonomia_entidades.md     # Catálogo de 21 entidades MEDDOCAN, reglas de aumentación y soporte
├── models/
│   ├── README.md             # Estado de artefactos de entrenamiento y política de almacenamiento de pesos
│   ├── beto_baseline/        # Metadatos (config.json, labels.json, 03_train_summary.json)
│   ├── beto_mixed/           # Metadatos (config.json, labels.json, 03_train_summary.json)
│   └── beto_full/            # Metadatos (config.json, labels.json, 03_train_summary.json)
├── notebooks/
│   └── 01_resultados.ipynb   # Notebook ejecutado de Fase 5 (tablas comparativas, IC 95%, heatmaps)
├── results/
│   ├── README.md             # Inventario de métricas F4 y artefactos visuales/tabulares F5
│   ├── beto_baseline__augmented_baseline/
│   ├── beto_mixed__augmented_50/
│   ├── beto_full__augmented_100/
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
python scripts/02_augmentation.py

# Fase 3: Entrenamiento (ejemplo para BETO Baseline)
python scripts/03_train_ner.py \
  --data-dir data/augmented_baseline \
  --output-dir models/beto_baseline \
  --model-name dccuchile/bert-base-spanish-wwm-cased

# Fase 4: Evaluación estricta sobre test.jsonl + Bootstrap IC 95%
python scripts/04_evaluate_ner.py \
  --model-dir models/beto_baseline \
  --data-dir data/augmented_baseline \
  --output-dir results/beto_baseline__augmented_baseline
```

---

## 5. Documentación Técnica Adicional

- [Bitacora_Desarrollo.md](Bitacora_Desarrollo.md): Historial completo de decisiones de diseño, auditorías de datos, resultados y tareas pendientes (`TODOs`).
- [docs/protocolo_experimental.md](docs/protocolo_experimental.md): Especificación formal de particiones, hiperparámetros, realineamiento de sub-tokens, protocolo bootstrap y amenazas a la validez.
- [docs/taxonomia_entidades.md](docs/taxonomia_entidades.md): Taxonomía de las 21 clases PHI de MEDDOCAN, estrategia de reemplazo sintético peruano y soporte estadístico por partición.
- [models/README.md](models/README.md): Trazabilidad de los modelos entrenados, política de exclusión de pesos binarios en Git y advertencia sobre checkpoints locales.
- [results/README.md](results/README.md): Descripción de los archivos generados por las Fases 4 y 5 (incluyendo tablas LaTeX listas para compilar).

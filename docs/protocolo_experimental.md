# Protocolo Experimental — Reconocimiento de Entidades Nombradas (NER) Clínico

Documento de especificación metodológica y de reproducibilidad para la adaptación del corpus **SPACCC_MEDDOCAN** al contexto clínico peruano mediante modelos Transformer.

---

## 1. Datos y Particiones (Splits)

El corpus base proviene de la tarea compartida **MEDDOCAN (Medical Document Anonymization Track — IberLEF 2019)** en formato BRAT (`.txt` y `.ann`), ubicado en `data/raw/corpus/{train,dev,test}/brat/`.

Mediante [01_brat_to_bio.py](file:///c:/Users/kotem/Documents/Universidad/26.2/Tesis/github/proyecto_tesis_nlp/scripts/01_brat_to_bio.py) se tokeniza con `spaCy` (`es_core_news_sm`), se alinea por solapamiento a nivel de carácter (dividiendo tokens con múltiples entidades o espacios internos como `"EE. UU."`) y se exporta a esquema **IOB2 (BIO)** en formato `.jsonl` (`id`, `tokens`, `tags`):

| Split | Documentos | Tokens (`processed`) | Menciones BRAT (`B-`) | Perdidas | `I-` huérfanas | Ruido de límites (1 entidad parcial) |
|---|---:|---:|---:|---:|---:|---:|
| `train` | 500 | 251.206 | 11.333 | 0 | 0 | 973 (0,39 %) |
| `dev` | 250 | 132.665 | 5.801 | 0 | 0 | 497 (0,37 %) |
| `test` | 250 | 126.093 | 5.661 | 0 | 0 | 482 (0,38 %) |
| **Total** | **1.000** | **509.964** | **22.795** | **0** | **0** | **1.952 (0,38 %)** |

- **Etiquetas BIO totales en el vocabulario (`train` ∪ `dev`):** **44 clases** (`O` + 22 prefijos `B-` + 21 prefijos `I-`, dado que `ID_CONTACTO_ASISTENCIAL` solo ocurre como token único en `train`/`dev`).

---

## 2. Escenarios de Aumentación (Fase 2 Congelada)

El script [02_augmentation.py](file:///c:/Users/kotem/Documents/Universidad/26.2/Tesis/github/proyecto_tesis_nlp/scripts/02_augmentation.py) (`--seed 42`) genera tres escenarios experimentales combinando `Faker('es_MX')` y *gazetteers* clínicos peruanos con longitud libre de span:

| Escenario | Directorio | Tasa (`--rate`) | Splits modificados | Tokens `train` | Entidades `train` reemplazadas | `dev` / `test` |
|---|---|---:|---|---:|---:|---|
| **Baseline (0 %)** | `data/augmented_baseline/` | `0.0` | Ninguno | 251.206 | 0 / 11.333 (0,0 %) | Copia exacta del original |
| **Mixto (50 %)** | `data/augmented_50/` | `0.5` | Solo `train` | 250.057 | 3.973 / 11.333 (35,1 %) | Copia exacta del original |
| **Full (100 %)** | `data/augmented_100/` | `1.0` | Solo `train` | 248.638 | 8.091 / 11.333 (71,4 %) | Copia exacta del original |

### Reglas de dominio e invariantes de Fase 2
1. **Categorías cerradas excluidas de la aumentación (3.200 menciones en `train`):** `FECHAS` (1.231), `EDAD_SUJETO_ASISTENCIA` (1.035), `SEXO_SUJETO_ASISTENCIA` (925) y `OTROS_SUJETO_ASISTENCIA` (9) nunca se reemplazan para preservar su distribución gramatical y clínica original.
2. **Descriptores sin dígitos en `ID_SUJETO_ASISTENCIA` (42 menciones en `train`):** Menciones no numéricas heredadas del corpus original (`"lactante"`, `"casada"`, `"raza caucásica"`) no se sustituyen por DNI peruanos. Por ello, el techo de reemplazo en `Full (100 %)` es $11.333 - 3.200 - 42 = 8.091$ entidades ($71{,}4\ \%$).
3. **Concordancia de género clínica:** En MEDDOCAN `H.` = hombre y `M.` = mujer. Para `NOMBRE_SUJETO_ASISTENCIA` prevalece el sexo documentado en `SEXO_SUJETO_ASISTENCIA` sobre el género del nombre original (concordancia externa verificada: 100 %).

---

## 3. Aislamiento del Conjunto de Prueba (Anti-Leakage)

- El conjunto `test.jsonl` (250 documentos, 126.093 tokens, 5.661 entidades) **jamás es accedido por `03_train_ner.py`** y no recibe aumentación en ninguno de los escenarios.
- Hash de integridad verificable de `test.jsonl` en las cuatro rutas (`data/processed/test.jsonl`, `data/augmented_baseline/test.jsonl`, `data/augmented_50/test.jsonl`, `data/augmented_100/test.jsonl`):
  ```text
  SHA256 canónico (LF): d596b103e92a38e8d7d45ea8fc9e674632aa3fd85e1a55cfca92368e129a680d
  SHA256 variante CRLF (Windows): 8b30b80360cce568a9087b754b9a38b33aa278a6be952107eab3c4ef1bc898e3
  ```
  Ambos hashes corresponden al mismo contenido (250 documentos, 126.093 tokens, 5.661 entidades); la diferencia es el final de línea. Los summaries de BETO registran la variante CRLF y los de XLM-R la variante LF.
- El script [04_evaluate_ner.py](file:///c:/Users/kotem/Documents/Universidad/26.2/Tesis/github/proyecto_tesis_nlp/scripts/04_evaluate_ner.py) bloquea por defecto cualquier archivo cuyo nombre no sea `test.jsonl` (salvo que se active `--allow-non-test` para evaluar un futuro *gold set* peruano externo) y registra `test_file_sha256` en `04_eval_summary.json`.

---

## 4. Arquitecturas e Hiperparámetros de Entrenamiento (Fase 3)

### Modelos evaluados
1. **BETO (Completado — 3/3 escenarios):** `dccuchile/bert-base-spanish-wwm-cased`
2. **XLM-RoBERTa (Completado — 3/3 escenarios):** `xlm-roberta-base`

### Hiperparámetros estandarizados (`03_train_ner.py`)

| Parámetro | Valor | Notas |
|---|---|---|
| `epochs` | `5.0` | `1.0` en modo `--smoke-test` (50 docs) |
| `batch_size` | `16` | Por dispositivo (`train` y `eval`) |
| `learning_rate` | `2e-5` | Optimizador `AdamW` |
| `weight_decay` | `0.01` | Regularización L2 |
| `warmup_ratio` | `0.0` | Flag opt-in `--warmup-ratio` disponible |
| `early_stopping_patience` | `None` | Flag opt-in `--early-stopping-patience` disponible |
| `max_length` | `512` | Ventana de sub-tokens |
| `stride` | `128` | Solapamiento entre ventanas (`return_overflowing_tokens=True`) |
| `fp16` | `True` (en CUDA) | Precisión mixta automática en GPU |
| `metric_for_best_model` | `f1` (en `dev.jsonl`) | `load_best_model_at_end=True` (`save_strategy="epoch"`) |
| `seed` / `data_seed` | `42` | Fijada en `random`, `numpy`, `torch` y `transformers.set_seed` |

### Alineación sub-token y ventana deslizante
- Sólo el **primer sub-token** de cada palabra recibe la etiqueta BIO (`label2id[tag]`); los sub-tokens secundarios y tokens especiales (`[CLS]`/`<s>`, `[SEP]`/`</s>`, `[PAD]`) reciben `-100` (`IGNORE_INDEX`).
- En documentos que exceden 512 sub-tokens y se dividen con `stride=128`, las palabras del contexto solapado ya cubiertas en la ventana previa reciben `-100` durante entrenamiento y validación (BETO: `train_windows` 1.023/1.024/1.032 en baseline/mixto/full y `dev_windows` 538; XLM-R: `train_windows` 1.136/1.142/1.142 y `dev_windows` 599).

---

## 5. Evaluación y Bootstrap (Fase 4)

Ejecutada mediante [04_evaluate_ner.py](file:///c:/Users/kotem/Documents/Universidad/26.2/Tesis/github/proyecto_tesis_nlp/scripts/04_evaluate_ner.py):
- **Reconstrucción documental (`--aggregation first`):** Cada palabra del documento recibe la predicción del primer sub-token en la primera ventana que la contiene (idéntico al criterio de `03_train_ner.py`).
- **Reparación BIO en inferencia:** Si el modelo predice una etiqueta `I-X` huérfana (precedida de `O` o de `I-Y` con $Y \neq X$), se repara a `B-X` y se contabiliza (`bio_repairs`: BETO 119/110/167 y XLM-R 87/78/698 en baseline/mixto/full). Las etiquetas *gold* se validan estrictamente y nunca se alteran.
- **Métricas oficiales (`seqeval`):** Cálculo a nivel de **entidad exacta** (inicio, fin y tipo idénticos). Se reportan `Micro-F1` (métrica primaria), `Macro-F1` (sobre las 21 etiquetas con soporte o predicción en `test.jsonl`) y `Weighted-F1`.
- **Intervalos de confianza Bootstrap (`--bootstrap 1000 --bootstrap-seed 42`):** Remuestreo con reemplazo a nivel de **documento** ($N=250$ documentos por réplica, $B=1.000$ iteraciones), reportando percentiles $2{,}5\ \%$ y $97{,}5\ \%$ (IC 95 %).

---

## 6. Entorno de Ejecución Registrado

### Entorno de entrenamiento y evaluación en GPU (Google Colab — Fase 3 y 4 de BETO y XLM-R)
- **Plataforma:** `Linux-6.6.159+-x86_64-with-glibc2.39`
- **GPU:** NVIDIA Tesla T4 (`device: cuda`)
- **Python:** `3.13.16`
- **PyTorch:** `2.14.1+cu130`
- **Transformers:** `4.57.6`

### Entorno local de ingeniería de datos y visualización (Windows — Fases 1, 2 y 5)
- **OS:** Windows 10/11 (`venv` local)
- **Python:** `3.11.x`
- **spaCy:** `3.8.18` (`es_core_news_sm`)
- **Faker:** `40.44.0`

---

## 7. Comandos de Reproducción End-to-End

```bash
# Fase 1: Conversión BRAT -> BIO JSONL
python scripts/01_brat_to_bio.py

# Fase 2: Generación determinista de los 3 escenarios (seed=42)
python scripts/02_augmentation.py --scenario baseline
python scripts/02_augmentation.py --scenario mixed
python scripts/02_augmentation.py --scenario full

# Fase 3: Entrenamiento (ejemplo BETO en los 3 escenarios)
python scripts/03_train_ner.py --data-dir data/augmented_baseline --output-dir models/beto_baseline --model-name dccuchile/bert-base-spanish-wwm-cased --epochs 5 --batch-size 16 --seed 42
python scripts/03_train_ner.py --data-dir data/augmented_50       --output-dir models/beto_mixed    --model-name dccuchile/bert-base-spanish-wwm-cased --epochs 5 --batch-size 16 --seed 42
python scripts/03_train_ner.py --data-dir data/augmented_100      --output-dir models/beto_full     --model-name dccuchile/bert-base-spanish-wwm-cased --epochs 5 --batch-size 16 --seed 42

# Fase 3: Entrenamiento XLM-RoBERTa en los 3 escenarios
python scripts/03_train_ner.py --data-dir data/augmented_baseline --output-dir models/xlmr_baseline --model-name xlm-roberta-base --epochs 5 --batch-size 16 --seed 42
python scripts/03_train_ner.py --data-dir data/augmented_50       --output-dir models/xlmr_mixed    --model-name xlm-roberta-base --epochs 5 --batch-size 16 --seed 42
python scripts/03_train_ner.py --data-dir data/augmented_100      --output-dir models/xlmr_full     --model-name xlm-roberta-base --epochs 5 --batch-size 16 --seed 42

# Fase 4: Evaluación estricta sobre test.jsonl con Bootstrap (1000 iteraciones)
python scripts/04_evaluate_ner.py --model-dir models/beto_baseline --data-dir data/augmented_baseline --output-dir results/beto_baseline__augmented_baseline --bootstrap 1000
python scripts/04_evaluate_ner.py --model-dir models/beto_mixed    --data-dir data/augmented_50       --output-dir results/beto_mixed__augmented_50       --bootstrap 1000
python scripts/04_evaluate_ner.py --model-dir models/beto_full     --data-dir data/augmented_100      --output-dir results/beto_full__augmented_100      --bootstrap 1000
python scripts/04_evaluate_ner.py --model-dir models/xlmr_baseline --data-dir data/augmented_baseline --output-dir results/xlmr_baseline__augmented_baseline --bootstrap 1000
python scripts/04_evaluate_ner.py --model-dir models/xlmr_mixed    --data-dir data/augmented_50       --output-dir results/xlmr_mixed__augmented_50       --bootstrap 1000
python scripts/04_evaluate_ner.py --model-dir models/xlmr_full     --data-dir data/augmented_100      --output-dir results/xlmr_full__augmented_100      --bootstrap 1000

# Fase 5: Generación de tablas LaTeX/CSV y figuras en notebooks/01_resultados.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/01_resultados.ipynb
```

---

## 8. Política de Artefactos y Limitaciones

1. **Política de versionado en Git:**
   - **Versionados:** `scripts/`, `notebooks/01_resultados.ipynb` (con outputs), `docs/`, resúmenes JSON (`data/processed/01_summary.json`, `data/augmented_*/02_summary.json`, `models/*/03_train_summary.json`), metadatos ligeros del modelo (`config.json`, `labels.json`, `tokenizer*.json`, `sentencepiece.bpe.model`) y todos los reportes/tablas/figuras de `results/`.
   - **No versionados (`.gitignore`):** archivos `.jsonl` de `data/processed/` y `data/augmented_*/`, respaldos `data/archive/`, corridas de prueba `models/smoke_*/`, checkpoints intermedios `models/**/checkpoints/` y pesos binarios (`model.safetensors`, `pytorch_model.bin`).
2. **Limitaciones metodológicas clave:**
   - Evaluación realizada sobre `test.jsonl` de MEDDOCAN (español peninsular); no mide rendimiento sobre un *gold set* clínico peruano real.
   - Una sola semilla (`seed=42`) por configuración.
   - Bootstrap a nivel documento ($N=250$); la superposición de intervalos de confianza no reemplaza un test formal de hipótesis pareado.
   - Cuatro etiquetas de soporte muy bajo en `test` (`CENTRO_SALUD`, `NUMERO_FAX`, `OTROS_SUJETO_ASISTENCIA`, `PROFESION`) presentan $\text{F1}=0.0000$ en las 6 corridas (BETO y XLM-R), reduciendo el `Macro-F1`.

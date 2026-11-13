# Directorio de Resultados (`results/`)

Contiene todos los artefactos de evaluación estricta sobre `test.jsonl` generados por la **Fase 4** ([04_evaluate_ner.py](file:///c:/Users/kotem/Documents/Universidad/26.2/Tesis/github/proyecto_tesis_nlp/scripts/04_evaluate_ner.py)) y las tablas/figuras comparativas generadas por la **Fase 5** ([01_resultados.ipynb](file:///c:/Users/kotem/Documents/Universidad/26.2/Tesis/github/proyecto_tesis_nlp/notebooks/01_resultados.ipynb)).

Todos los archivos de este directorio son ligeros y **se versionan en Git** (con la única excepción de `*/04_predictions.jsonl`, si se generase con `--save-predictions`).

---

## 1. Estructura Actual

```text
results/
├── beto_baseline__augmented_baseline/   # Fase 4: BETO entrenado en Baseline (0%)
│   ├── 04_eval_summary.json             # Resumen completo (Micro/Macro/Weighted F1, conteos, entorno)
│   ├── 04_bootstrap.json                # IC 95% Bootstrap (1000 iteraciones, seed=42)
│   ├── 04_classification_report.txt     # Reporte oficial seqeval en texto plano
│   ├── 04_classification_report.json    # Reporte oficial seqeval estructurado
│   ├── 04_per_entity.csv                # Precision, Recall, F1, soporte, TP, FP, FN por entidad
│   ├── 04_errors.csv                    # Análisis exploratorio de errores con contexto
│   └── 04_confusions.csv                # Pares de confusión de tipo y errores de frontera
├── beto_mixed__augmented_50/            # Fase 4: BETO entrenado en Mixto (50%)
│   └── (mismos 7 archivos 04_*)
├── beto_full__augmented_100/            # Fase 4: BETO entrenado en Full (100%)
│   └── (mismos 7 archivos 04_*)
├── xlmr_baseline__augmented_baseline/   # Fase 4: XLM-R entrenado en Baseline (0%)
│   └── (mismos 7 archivos 04_*)
├── xlmr_mixed__augmented_50/            # Fase 4: XLM-R entrenado en Mixto (50%)
│   └── (mismos 7 archivos 04_*)
├── xlmr_full__augmented_100/            # Fase 4: XLM-R entrenado en Full (100%)
│   └── (mismos 7 archivos 04_*)
├── 05_resultados_comparativos.csv       # Fase 5: Tabla consolidada de todas las corridas
├── 05_tabla_resultados.tex              # Fase 5: Tabla principal en LaTeX para la tesis
├── 05_tabla_por_entidad.tex             # Fase 5: Tabla LaTeX desglosada por las 21 entidades con soporte en test
├── 05_tabla_micro_f1.tex                # Fase 5: Tabla pivote resumida de Micro-F1
├── 05_f1_por_escenario.png              # Fase 5: Gráfico de barras Micro-F1 y Macro-F1
├── 05_f1_entidades_clave.png            # Fase 5: Mapa de calor de F1 por entidad y escenario
├── 05_bootstrap_micro_f1.png            # Fase 5: Gráfico de Micro-F1 con IC 95% bootstrap
└── 05_bootstrap_macro_f1.png            # Fase 5: Gráfico de Macro-F1 con IC 95% bootstrap
```

---

## 2. Notas de Trazabilidad sobre las Corridas Actuales

- **Metadato `data_dir` en `beto_mixed__augmented_50` y `beto_full__augmented_100`:**
  Durante la ejecución en Google Colab se invocó `04_evaluate_ner.py` pasando explícitamente `--test-file data/augmented_50/test.jsonl` y `--test-file data/augmented_100/test.jsonl` sin sobrescribir `--data-dir`, por lo que el campo informativo `"data_dir"` quedó con su valor por defecto (`"data/augmented_baseline"`). Sin embargo, los campos `"test_file"` y `"scenario"` son exactos y los cuatro archivos `test.jsonl` son idénticos en contenido (250 documentos, 126.093 tokens, 5.661 entidades). El notebook de Fase 5 clasifica cada corrida usando exclusivamente `"scenario"` y el directorio padre de `"test_file"`.
- **Hash del test (CRLF vs. LF):**
  Los summaries de BETO registran `8b30b803...` (copia Windows con CRLF); los de XLM-R registran `d596b103...` (copia Linux/Colab con LF). Al normalizar CRLF→LF, el SHA256 es `d596b103...`; el contenido es idéntico. Se adopta `d596b103...` como hash canónico y se documenta la variante CRLF.
- **XLM-R integrado:**
  Los 3 subdirectorios `xlmr_*` están presentes y versionados. La matriz experimental está completa (6/6 corridas: BETO + XLM-R × Baseline/Mixto/Full). Los pesos `model.safetensors` siguen fuera de Git.

---

## 3. Cómo Regenerar los Artefactos

### Regenerar las tablas y figuras de Fase 5 (no requiere GPU ni pesos)
Basta con ejecutar el notebook [01_resultados.ipynb](file:///c:/Users/kotem/Documents/Universidad/26.2/Tesis/github/proyecto_tesis_nlp/notebooks/01_resultados.ipynb), que lee los archivos `04_*` ya versionados en `results/*/`:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_resultados.ipynb
```

### Regenerar las evaluaciones de Fase 4 (requiere los pesos entrenados en `models/`)
```bash
python scripts/04_evaluate_ner.py --model-dir models/beto_baseline --data-dir data/augmented_baseline --output-dir results/beto_baseline__augmented_baseline --bootstrap 1000
python scripts/04_evaluate_ner.py --model-dir models/beto_mixed    --data-dir data/augmented_50       --output-dir results/beto_mixed__augmented_50       --bootstrap 1000
python scripts/04_evaluate_ner.py --model-dir models/beto_full     --data-dir data/augmented_100      --output-dir results/beto_full__augmented_100      --bootstrap 1000
python scripts/04_evaluate_ner.py --model-dir models/xlmr_baseline --data-dir data/augmented_baseline --output-dir results/xlmr_baseline__augmented_baseline --bootstrap 1000
python scripts/04_evaluate_ner.py --model-dir models/xlmr_mixed    --data-dir data/augmented_50       --output-dir results/xlmr_mixed__augmented_50       --bootstrap 1000
python scripts/04_evaluate_ner.py --model-dir models/xlmr_full     --data-dir data/augmented_100      --output-dir results/xlmr_full__augmented_100      --bootstrap 1000
```

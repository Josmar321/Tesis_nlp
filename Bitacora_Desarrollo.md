# Bitácora de Desarrollo

Registro central del proyecto de tesis: decisiones, avances, experimentos y problemas.

## 1. Datos del proyecto

| Campo | Valor |
|---|---|
| Título | _(pendiente)_ |
| Autor | _(pendiente)_ |
| Asesor | _(pendiente)_ |
| Repositorio | `proyecto_tesis_nlp` |
| Inicio | 2026-10-29 |

## 2. Objetivo

Construir un modelo NER para texto clínico en español adaptado al contexto peruano
(anonimización de datos personales), partiendo de SPACCC_MEDDOCAN y aplicando
aumento de datos con entidades peruanas (nombres, DNI, direcciones, centros de salud)
mediante Faker (`es_MX`) + gazetteers peruanos propios. Modelos objetivo:
XLM-RoBERTa y BERT en español (BETO, `dccuchile/bert-base-spanish-wwm-cased`).

## 3. Estructura del proyecto

```
proyecto_tesis_nlp/
├── data/
│   ├── raw/                     # SPACCC_MEDDOCAN clonado (txt y ann originales; no versionado)
│   ├── processed/               # {train,dev,test}.jsonl en BIO (no versionados) + 01_summary.json
│   ├── augmented_baseline/      # Escenario 1: copia exacta (rate=0; .jsonl no versionados, 02_summary.json sí)
│   ├── augmented_100/           # Escenario 2: train aumentado al 100 %; dev/test originales
│   ├── augmented_50/            # Escenario 3: train aumentado al 50 %; dev/test originales
│   ├── archive/                 # Respaldos históricos (ignorado en Git)
│   └── augmented/               # (obsoleta; solo .gitkeep)
├── docs/
│   ├── protocolo_experimental.md # Protocolo experimental, hiperparámetros, bootstrap y limitaciones
│   └── taxonomia_entidades.md    # Catálogo de 21 entidades MEDDOCAN, aumentación y soporte
├── notebooks/                   # EDA, resultados ejecutados (01_resultados.ipynb)
├── scripts/
│   ├── 01_brat_to_bio.py        # BRAT + spaCy -> BIO (JSONL)
│   ├── 02_augmentation.py       # Aumentación: Faker es_MX + gazetteers peruanos
│   ├── 03_train_ner.py          # Entrenamiento con Hugging Face
│   └── 04_evaluate_ner.py       # Evaluación estricta sobre test.jsonl (seqeval)
├── models/                      # Metadatos versionados (config.json, labels.json, 03_train_summary.json); pesos binarios no versionados
├── results/                     # Salidas ligeras de Fase 4 y Fase 5 (JSON/CSV/TXT/TEX/PNG versionados)
├── README.md
├── requirements.txt
└── Bitacora_Desarrollo.md
```

## 4. Estado actual

- [x] Estructura inicial del repositorio creada.
- [x] Corpus SPACCC_MEDDOCAN disponible en `data/raw/corpus/` (500 train / 250 dev / 250 test).
- [x] Ejecutar `01_brat_to_bio.py` y validar el corpus BIO.
- [x] Crear/adaptar `02_augmentation.py` a la salida real de 01.
- [x] Reemplazar `02_augmentation.py` por una versión basada en Faker + `ProviderPeruano`.
- [x] Ejecutar `02_augmentation.py`; genera 3 archivos en `data/augmented/`.
- [x] Corregir los defectos de la versión Faker (ver §5 y §8). _(2026-11-04)_
- [x] Re-ejecutar la aumentación solo sobre `train` y en formato JSONL. _(2026-11-04)_
- [x] Corregir la pérdida de entidades por puntuación en 01 y regenerar `data/processed/`. _(0 pérdidas)_
- [x] Generar los tres escenarios experimentales (`augmented_baseline`, `augmented_100`, `augmented_50`).
- [x] Ajustes de cierre de 02 (mapeo `H.`/`M.`, `ID_SUJETO` sin dígitos) y **congelamiento de la Fase 2** (seed 42).
- [ ] EDA en `notebooks/` (pendiente; el notebook de resultados de Fase 5 ya está ejecutado).
- [x] Fase 3: `03_train_ner.py` con rutas por defecto a `data/augmented_baseline`, ventana 512/128, validación estricta de `label2id` y soporte opcional para warmup/early stopping. _(2026-11-11 / 2026-11-12)_
- [x] Entrenar las 3 combinaciones de BETO (Baseline, Mixto 50%, Full 100%) en Google Colab (GPU Tesla T4). _(2026-11-11)_
- [ ] **TODO (decisión humana — 3/6 corridas pendientes):** Entrenar y evaluar las 3 combinaciones de `xlm-roberta-base` en GPU, o acotar formalmente el alcance de §2 exclusivamente a BETO.
- [x] Fase 4: `04_evaluate_ner.py` implementado (test-only, `seqeval`, bootstrap, análisis de errores y hash `SHA256` del test). _(2026-11-11 / 2026-11-12)_
- [x] Ejecutar las evaluaciones con bootstrap (IC 95%) y documentar resultados reales de BETO. _(2026-11-11)_
- [x] Fase 5: `notebooks/01_resultados.ipynb` generalizado, ejecutado con salidas guardadas y artefactos `results/05_*` exportados (`.csv`, `.tex`, `.png`). _(2026-11-12)_
- [x] Documentar protocolo experimental (`docs/protocolo_experimental.md`), taxonomía de entidades (`docs/taxonomia_entidades.md`), trazabilidad de modelos/resultados (`models/README.md`, `results/README.md`) y política de versionado (`.gitignore`). _(2026-11-12)_

## 5. Registro de sesiones

### 2026-10-29 — Creación de la estructura

**Objetivo:** montar el esqueleto del proyecto.

**Avances:**
- Carpetas `data/{raw,processed,augmented}`, `notebooks/`, `scripts/`, `models/`.
- Scripts base: conversión BRAT→BIO, aumento peruano y entrenamiento HF.
- `requirements.txt` y esta bitácora.

**Decisiones:**
- Formato intermedio JSONL (`tokens` + `tags`) por ser compatible con `datasets`.
- Tokenización con spaCy y alineación por carácter para no perder límites de entidad.

**Problemas:** _(ninguno)_

**Próximos pasos:**
1. Clonar SPACCC_MEDDOCAN en `data/raw/SPACCC_MEDDOCAN`.
2. Ejecutar el script 01 y revisar el conteo de entidades.

### 2026-10-29 — Ejecución de `01_brat_to_bio.py`

**Objetivo:** convertir BRAT + spaCy a BIO y validar los tres archivos de salida.

**Comando:**
```bash
python scripts/01_brat_to_bio.py
```
(ejecutado desde `proyecto_tesis_nlp/` con el `venv` del proyecto)

**Salidas en `data/processed/`:**

| Archivo | Documentos | Tokens | Menciones `B-` | Etiquetas BIO | Tamaño |
|---|---:|---:|---:|---:|---:|
| `corpus_bio_train.json` | 500 | 251.186 | 10.672 | 22.293 | 7,10 MB |
| `corpus_bio_dev.json` | 250 | 132.663 | 5.463 | 11.479 | 3,74 MB |
| `corpus_bio_test.json` | 250 | 126.085 | 5.324 | 11.103 | 3,56 MB |
| **Total** | **1.000** | **509.934** | **21.459** | **44.875** | **14,40 MB** |

**Validación:**
- `len(tokens) == len(etiquetas_ner)` en los 1.000 documentos.
- Sin IDs duplicados ni etiquetas BIO inválidas.
- 22 tipos de entidad presentes; 44 etiquetas distintas en la salida, incluida `O`.
- **Hallazgo:** el BRAT original tiene 22.795 menciones, pero la conversión conserva 21.459 `B-`; se pierden 1.336 (5,9 %). La causa principal es que spaCy pega la puntuación al token (p. ej. `H.` para la entidad `SEXO` = `H`) y la condición `token_fin <= fin_entidad` descarta la entidad. `SEXO_SUJETO_ASISTENCIA` pierde 925 de 1.841 menciones (50,2 %). Se recomienda corregir la alineación de 01 antes del entrenamiento final.

**Decisión:**
- Conservar temporalmente la salida JSON de 01 (`id_documento`, `tokens`, `etiquetas_ner`) y adaptar 02/03, en lugar de reescribir 01 en esta sesión.

### 2026-11-03 — Preparación de `02_augmentation.py`

**Objetivo:** dejar listo el aumento peruano compatible con la salida real de 01.

**Avances:**
- Reemplazado el borrador de `scripts/02_augmentation.py`, que esperaba `.jsonl` con `id`/`tags`, por una versión que:
  - lee `data/processed/corpus_bio_{split}.json` y también acepta `.jsonl`;
  - normaliza `id_documento`/`id` y `etiquetas_ner`/`tags`;
  - aumenta solo `train` por defecto (`--splits train`);
  - reemplaza nombres, DNI, teléfonos, correos, fechas, edades, direcciones, lugares, hospitales y centros de salud por valores peruanos;
  - conserva la secuencia BIO y, cuando aplica, el número de tokens del span;
  - escribe `data/augmented/{split}.jsonl` con `id`, `tokens`, `tags`, compatible con `03_train_ner.py`;
  - soporta `--rate`, `--copies`, `--seed`, `--limit` y `--dry-run`.
- Prueba de humo: `--dry-run` sobre los 500 documentos de train y salida temporal de 5 documentos; en ambos casos `len(tokens) == len(tags)`.

**Decisión:**
- Aumentar solo `train`; `dev` y `test` se mantienen sin aumentar para evaluar sobre la distribución original.
- Ignorar `I-` huérfanos en 02 para no crear entidades parciales; la corrección de fondo corresponde a 01.

**Próximos pasos:**
1. Ejecutar `python scripts/02_augmentation.py`.
2. Revisar una muestra de `data/augmented/train.jsonl`.
3. Corregir la alineación de 01 (puntuación) y regenerar `data/processed/` antes del entrenamiento final.
4. Adaptar `03_train_ner.py` para leer `data/augmented/train.jsonl` y un `dev` sin aumentar.

### 2026-11-03 — Implementación y ejecución de `02_augmentation.py` con Faker

**Objetivo:** sustituir los diccionarios estáticos por Faker + un provider peruano y generar el corpus aumentado.

**Implementación:**
- `ProviderPeruano(BaseProvider)` con listas propias: 50 apellidos, 36 nombres, 28 ciudades y 19 hospitales.
- `Faker()` sin locale explícito (el default es `en_US`); no se usan métodos nativos de Faker, solo `random_element` del provider.
- Reemplazo token a token, sin re-tokenizar el valor generado:
  - `ID_SUJETO*` / `ID_TITULAR*` → DNI de 8 dígitos;
  - `NUMERO_TELEFONO` → 9 dígitos;
  - `NOMBRE_*`: `B-` → nombre, `I-` → apellido;
  - `TERRITORIO` / `PAIS` / `CALLE`: `B-` → ciudad;
  - `HOSPITAL` / `CENTRO_SALUD`: `B-` → primera palabra del hospital, `I-` → palabra aleatoria.
- Procesa todos los `.json` de `data/processed` y escribe en `data/augmented` con el mismo nombre (`corpus_bio_{split}.json`).
- Salida en JSON (`id_documento`, `tokens`, `etiquetas_ner`), no en JSONL.

**Ejecución:**
```bash
venv\Scripts\python.exe scripts\02_augmentation.py
```
Faker 40.44.0 instalado en el `venv`; no figura en `requirements.txt`.

**Salidas:**

| Archivo | Documentos | Tokens | Tokens modificados |
|---|---:|---:|---:|
| `data/augmented/corpus_bio_train.json` | 500 | 251.186 | 9.722 |
| `data/augmented/corpus_bio_dev.json` | 250 | 132.663 | 4.897 |
| `data/augmented/corpus_bio_test.json` | 250 | 126.085 | 4.881 |

**Validación:**
- `len(tokens) == len(etiquetas_ner)` en los 1.000 documentos; las etiquetas no cambiaron.
- Solo se modificaron 9 de los 22 tipos de entidad del corpus (21 presentes en `train`); quedaron intactos 13: `FECHAS`, `EDAD_SUJETO_ASISTENCIA`, `SEXO_SUJETO_ASISTENCIA`, `CORREO_ELECTRONICO`, `ID_ASEGURAMIENTO`, `ID_TITULACION_PERSONAL_SANITARIO`, `FAMILIARES_SUJETO_ASISTENCIA`, `INSTITUCION`, `ID_CONTACTO_ASISTENCIAL`, `PROFESION`, `NUMERO_FAX`, `OTROS_SUJETO_ASISTENCIA` e `ID_EMPLEO_PERSONAL_SANITARIO`.
- `B-HOSPITAL` colapsa a 4 valores: `Hospital` (184), `Clínica` (29), `Instituto` (28) y `Centro` (14). Es una fuga de etiqueta: el modelo puede aprender `B-HOSPITAL` solo por la palabra inicial.
- Reemplazos multi-token incoherentes: un teléfono de 3–4 tokens queda con 3–4 números distintos; un `ID_SUJETO` multi-token con varios DNI; `CALLE` solo cambia la primera palabra y conserva el resto de la dirección; `TERRITORIO` conserva sufijos («Tarapoto Coruña»); `PAIS` se convierte en una ciudad.
- 351 tokens de `train` contienen espacios (`Luis Alberto`, `María Elena`, `Cerro de Pasco`, `Puerto Maldonado`), lo que rompe la unidad token/etiqueta.
- `ID_SUJETO_ASISTENCIA` incluye spans no numéricos heredados de la anotación/01 (`raza caucásica`, `cabello amarillento`); el script los deja intactos.
- Sin semilla: `random` y Faker no se siembran; la salida no es reproducible.
- `dev` y `test` también fueron aumentados, contra la decisión de mantenerlos intactos.
- La salida no es JSONL y no tiene `id`/`tags`; `03_train_ner.py` no puede leerla.

**Decisión:**
- Se acepta Faker + provider peruano como base, pero esta implementación **no se considera válida para el experimento final** hasta corregir: train-only, JSONL, semilla, re-tokenización, semántica por etiqueta y formato de los identificadores.
- Mantener `dev`/`test` sin aumentar y regenerarlos desde `data/processed`.

**Próximos pasos:**
1. Corregir `02_augmentation.py` (o crear `02_augmentation_faker.py`) con los puntos anteriores.
2. Fijar `Faker==40.44.0` en `requirements.txt`.
3. Re-ejecutar solo `train` y validar `len(tokens) == len(tags)` y ausencia de tokens con espacios.
4. Adaptar `03_train_ner.py` al JSONL y a un `dev` sin aumentar.
5. Corregir 01 y regenerar `data/processed/` antes del entrenamiento final.

### 2026-11-04 — Auditoría de las Fases 1 y 2

**Objetivo:** verificar el estado de 01 (ya reescrito con alineación por solapamiento y salida JSONL) y de 02 (Faker `es_MX` + gazetteers, exclusión de FECHAS/EDAD/SEXO/OTROS).

**Método:** script de validación independiente sobre `data/processed/` y ejecución de 02 (`--rate 1.0`) en un directorio temporal, comparando cada documento aumentado contra su original.

**Resultados Fase 1 (01):**

| Split | Docs | Entidades BRAT | `B-` | Perdidas | `I-` huérfanas |
|---|---:|---:|---:|---:|---:|
| train | 500 | 11.333 | 11.333 | 0 | 0 |
| dev | 250 | 5.801 | 5.801 | 0 | 0 |
| test | 250 | 5.661 | 5.661 | 0 | 0 |

- Se resolvió la pérdida histórica de 1.336 menciones (5,9 %).
- **Hallazgo:** 2 tokens con espacio en train (`"EE. UU."`, excepción del tokenizador de spaCy). La función `_emitir_token` existía, pero no se invocaba (código muerto).
- **Hallazgo:** ruido en los límites cuando una entidad cubre solo parte de un token (el token completo hereda la etiqueta).

**Resultados Fase 2 (02):**
- Integridad BIO correcta; contexto `O` intacto; dev/test sin cambios; categorías excluidas sin cambios.
- **Defectos:**
  1. Recorte/relleno para igualar la longitud original → entidades rotas (`'Calle Las Begonias Dpto. 6 315 Dpto.'`, `"Villa El"`, `"San Martín de"`, relleno con `"directo"`).
  2. Sin concordancia de género (`'Ernesto'` → `'Silvia'`, con la mención SEXO intacta en el texto).
  3. `ID_CONTACTO_ASISTENCIAL` generaba celulares (`'Episodio:3629184'` → `'995594493'`).
  4. Prefijos quemados y dígitos fragmentados (`'CMP 93 428'`, `'ESSALUD 19247 96048'`).
  5. Baja entropía: 7 hospitales, 3 centros, 3 instituciones, 5 calles; `PAIS` siempre `"Perú"`.
  6. `faker` ausente de `requirements.txt`; rutas relativas al directorio de trabajo.
- **Precisión conceptual:** excluir FECHAS/EDAD/SEXO/OTROS preserva su distribución original, pero no es en sí una medida anti-*leakage*; lo que evita la fuga es no aumentar dev/test.

### 2026-11-04 — Refactorización de 01 y 02, y generación de los tres escenarios

**Objetivo:** corregir los hallazgos de la auditoría y producir los datos de los experimentos.

**Cambios en `01_brat_to_bio.py`:**
- `_emitir_token` ahora se usa en ambas rutas de emisión (token completo y subtokens). `"EE. UU."` → `EE.`/`B-PAIS` + `UU.`/`I-PAIS`.
- El docstring y un comentario en línea documentan el ruido en los límites. Se mantiene la lógica, porque garantiza el 100 % de recuperación.

**Cambios en `02_augmentation.py` (reescrito):**
- **Generación libre de longitud:** se eliminaron el recorte, el relleno y `split_digits`. El valor se inserta con su longitud natural y el span se reetiqueta como `B-` + `I-`×(n−1).
- **Concordancia de género:**
  - Léxico de nombres de pila de Faker (`es_ES`, `es_MX`, `es_CO`, `es_AR`, `es_CL`) más nombres peruanos.
  - Para el paciente, si el nombre original no lo permite, se usa la mención `SEXO_SUJETO_ASISTENCIA` del documento.
  - Se replica la estructura (nombres o iniciales + hasta 2 apellidos).
  - Se forzaron `José`/`Jesús` → M y `María`/`Rosario`/`Guadalupe` → F (Faker `es_ES` los lista en ambos géneros por los nombres compuestos).
  - El parentesco también concuerda en género.
- **Formatos sin prefijos:**
  - `ID_CONTACTO_ASISTENCIAL`: episodio de 7 dígitos.
  - `ID_SUJETO_ASISTENCIA`: DNI (8) o historia clínica (6–7).
  - `ID_ASEGURAMIENTO`: DNI o código autogenerado tipo EsSalud.
  - `ID_TITULACION`: colegiatura de 5–6 dígitos.
  - `ID_EMPLEO`: código de 6–8 dígitos.
  - `NUMERO_FAX`: solo fijos (Lima `01` / provincias `0XX`).
  - `NUMERO_TELEFONO`: 70 % celular / 30 % fijo.
- **Gazetteers ampliados:**

  | Lista | Antes | Ahora |
  |---|---|---|
  | Hospitales y clínicas (MINSA, EsSalud, institutos, FF. AA., regionales, privadas) | 7 | ~80 |
  | Centros de salud | 3 | ~43 |
  | Instituciones | 3 | ~45 |
  | Lugares (departamentos, ciudades, distritos) | — | ~115 |
  | Calles | 5 | 60 vías × 5 formatos (Av./Jr., Dpto., Int., cdra., Mz./Lt. + Urb./AA.HH.) |
  | Países | solo `"Perú"` | 27 entradas, 16 países distintos; P(Perú) = 12/27 = 44,4 % |
  | Profesiones | — | 60 % lista peruana / 40 % Faker (≤ 4 palabras) |
  | Correos | — | dominios peruanos (`minsa.gob.pe`, `unmsm.edu.pe`, …) |

- **Infraestructura:**
  - `BASE_DIR` con `pathlib`; las rutas relativas se resuelven contra la raíz del proyecto.
  - Presets `--scenario baseline|full|mixed`.
  - Faker con `seed_instance` por split/copia y `random.Random(seed)`.
  - Validación de invariantes al final de cada documento.
  - `02_summary.json` por carpeta, con conteos por etiqueta.
- `requirements.txt`: se añadió `faker>=24.0` (instalado: 40.44.0) y la instrucción `python -m spacy download es_core_news_sm`.

**Comandos:**
```bash
venv\Scripts\python.exe scripts\01_brat_to_bio.py
venv\Scripts\python.exe scripts\02_augmentation.py --scenario baseline
venv\Scripts\python.exe scripts\02_augmentation.py --scenario full
venv\Scripts\python.exe scripts\02_augmentation.py --scenario mixed
```

**Volumetría — Fase 1 (`data/processed/`):**

| Split | Docs | Tokens | Entidades BRAT | `B-` | Perdidas | Tokens divididos | Tokens con espacio |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 500 | 251.206 | 11.333 | 11.333 | 0 | 9 | 0 |
| dev | 250 | 132.665 | 5.801 | 5.801 | 0 | 1 | 0 |
| test | 250 | 126.093 | 5.661 | 5.661 | 0 | 4 | 0 |

**Volumetría — Fase 2 (seed = 42):**

| Escenario | Carpeta | Split | Tokens in → out | Entidades | Reemplazadas |
|---|---|---|---|---:|---:|
| Baseline (rate 0) | `augmented_baseline/` | train / dev / test | 251.206 / 132.665 / 126.093 (sin cambio) | 11.333 / 5.801 / 5.661 | 0 |
| Aumentado (rate 1.0) | `augmented_100/` | train | 251.206 → 248.824 (−2.382) | 11.333 | 8.133 (71,8 %) |
| | | dev / test | sin cambio (originales) | 5.801 / 5.661 | 0 |
| Mixto (rate 0.5) | `augmented_50/` | train | 251.206 → 250.052 (−1.154) | 11.333 | 4.032 (35,6 %) |
| | | dev / test | sin cambio (originales) | 5.801 / 5.661 | 0 |

- 71,8 % es el techo con rate 1.0: las 3.200 entidades restantes son de las categorías excluidas (FECHAS 1.231, EDAD 1.035, SEXO 925, OTROS 9).
- La reducción de tokens se debe a formatos peruanos más compactos que los españoles (p. ej. `'28 28 70973'` → `'130866'`; `'c/ del Abedul 5-7 , 2º dcha'` → `'Jr. Callao 2259'`).

**Validación independiente (sobre las salidas finales):**
- Integridad BIO en todas las carpetas: 0 desalineaciones, 0 `I-` huérfanas, 0 tokens vacíos o con espacios.
- `augmented_baseline` es idéntico a `processed` (500/250/250 docs).
- dev/test son idénticos al original en los tres escenarios.
- Tokens `O` alterados: 0. Secuencia de tipos de entidad alterada: 0. Categorías excluidas modificadas: 0.
- Concordancia de género en nombres: 1.432/1.432 (`augmented_100`) y 720/720 (`augmented_50`).
  - **Corrección (sesión de cierre):** esta cifra es **interna** (género del nombre original vs. nombre nuevo) y no detectaba el bug de mapeo `H.`/`M.`. Contra el sexo real del documento, la concordancia era del 93,1 % (484/520). Ver «Cierre de la Fase 2».

**Ejemplos (`augmented_100/train`):**

| Etiqueta | Original | Aumentado |
|---|---|---|
| NOMBRE_SUJETO_ASISTENCIA | `Jose` | `Wilfredo` |
| NOMBRE_PERSONAL_SANITARIO | `Ignacio Navarro Cuéllar` | `Lorenzo Atao Montemayor` |
| CALLE | `c/ del Abedul 5-7 , 2º dcha` | `Jr. Callao 2259` |
| HOSPITAL | `Hospital Universitario 12 de octubre` | `Hospital Nacional Almanzor Aguinaga Asenjo` |
| ID_ASEGURAMIENTO | `26 37482910 04` | `8110057WJHUS005` |
| ID_CONTACTO_ASISTENCIAL | `Episodio:3629184` | `1250905` |
| NUMERO_FAX | `948 296 500` | `052 325995` |
| FAMILIARES_SUJETO_ASISTENCIA | `prima materna` | `sobrina` |

**Limitaciones conocidas:**
- Si un apellido coincide con un nombre de pila (`Luna`, `Cruz`), puede interpretarse como nombre. En la validación no produjo discordancias.
- Se mantiene el ruido en los límites de 01 (decisión documentada).

**Próximos pasos (Fase 3):**
1. Cambiar los valores por defecto de `03_train_ner.py` (hoy `data/augmented/train.jsonl`) por las carpetas `augmented_*/`.
2. Entrenar los tres escenarios con la misma semilla y los mismos hiperparámetros; evaluar sobre el test original.

> **Nota:** la volumetría y las cifras de género de esta sesión quedan **superadas** por la sesión «Cierre de la Fase 2», que regeneró los datos.

### 2026-11-04 — Cierre de la Fase 2 (ajustes tras revisión externa)

**Contexto:** una revisión externa confirmó que la etapa vigente es 01 + 02. Los riesgos señalados en 03 (truncación, uso del test, métricas por etiqueta) quedan para la siguiente etapa. Antes de congelar 02 se pidieron estos ajustes:

| Punto | ¿Bloquea el cierre? | Acción tomada |
|---|---|---|
| Mapeo `H.`/`M.` del sexo | Sí (bug) | Corregido y regenerado |
| `ID_SUJETO` con descriptores sin dígitos | No | Opción A: no se aumentan |
| Ruido de límites | No | Cifra medida y documentada |
| Texto de PAIS (27 entradas / 16 países) | No | Corregido; distribución reportada |
| 03 | Fuera de alcance | Siguiente etapa |

**1. Bug de mapeo de sexo (corregido):**
- En MEDDOCAN `H.` = hombre y `M.` = mujer. El código interpretaba `m` como masculino y no reconocía `h`.
- Valores de `SEXO_SUJETO_ASISTENCIA` en train: `H.` 264, `varón` 214, `M.` 196, `mujer` 149, `niña` 31, `niño` 26, `hombre` 12, `masculino` 9, `femenino` 9, `f.` 5, `femenina` 4, `h` 3, `anciana` 1, `recién nacido` 1.
- Mapeo nuevo:
  - Femenino: `m`, `f`, `mujer`, `femenino/a`, `hembra`, `niña`, `señora`, `anciana`, `nacida`, `gestante`, `puérpera`, `embarazada`, `esposa`.
  - Masculino: `h`, `v`, `varón`, `hombre`, `masculino`, `niño`, `señor`, `anciano`, `nacido`, `esposo`.
- **Prioridad de género para el paciente:** primero el SEXO del documento y después el nombre original (antes era al revés). Motivo: los nombres sustitutos de MEDDOCAN no siempre concuerdan con el sexo del paciente, mientras que el texto (pronombres, «la paciente») sí. Para el personal sanitario se mantiene el género del nombre original.
- **Apellidos ambiguos:** los slots de apellido descartan apellidos que también son nombres de pila (`Luna`, `Santiago`, `Tito`, `Alba`, `Rosario`, `León`, `de Jesús`). Antes, un span de solo apellidos podía leerse como un nombre del género opuesto.

**2. `ID_SUJETO_ASISTENCIA` sin dígitos (Opción A):**
- 42 spans en train (24 en dev, 13 en test) son descriptores, no identificadores: `Lactante`/`lactante` (8), `raza caucásica`, `casada`, `joven`, `boliviano`, `viudo`, `peruano`, `estudiante`, `niña adoptada`, `la sexta de nueve hermanos`, etc.
- Ya no se convierten en DNI; se conservan sin cambios. Los reemplazos en `full` bajan de 8.133 a 8.091 (−42).
- **Limitación:** son ruido de anotación del corpus original; el modelo los verá como `ID_SUJETO_ASISTENCIA`.

**3. Ruido de límites (01), medido:**

| Split | Tokens parcialmente cubiertos por 1 entidad | Total de tokens spaCy | % |
|---|---:|---:|---:|
| train | 973 | 251.186 | 0,39 % |
| dev | 497 | 132.663 | 0,37 % |
| test | 482 | 126.085 | 0,38 % |
| **Total** | **1.952** | **509.934** | **0,38 %** |

Son tokens que heredan la etiqueta completa aunque la entidad BRAT cubre solo una parte (p. ej. `Madrid,`, `NHC:2569870`).

**4. PAIS:** la lista tiene 27 entradas y 16 países distintos (Perú repetido 12 veces, P = 44,4 %). Distribución observada en `augmented_100/train` (713 menciones): Perú 299 (41,9 %); Ecuador, Japón y Colombia 34 cada uno; Haití, Venezuela y EE. UU. 30; Argentina y Chile 28; Italia y Bolivia 25; España 24; Cuba, China, México y Brasil 23.

**Regeneración (seed = 42):**

| Escenario | Split | Tokens in → out | Entidades | Reemplazadas |
|---|---|---|---:|---:|
| `augmented_baseline` | train / dev / test | 251.206 / 132.665 / 126.093 (sin cambio) | 11.333 / 5.801 / 5.661 | 0 |
| `augmented_100` | train | 251.206 → 248.638 (−2.568) | 11.333 | 8.091 (71,4 %) |
| | dev / test | sin cambio | 5.801 / 5.661 | 0 |
| `augmented_50` | train | 251.206 → 250.057 (−1.149) | 11.333 | 3.973 (35,1 %) |
| | dev / test | sin cambio | 5.801 / 5.661 | 0 |

**Conteos por etiqueta (train):**

| Etiqueta | Menciones | Reempl. 100 % | Tokens in→out (100 %) | Reempl. 50 % |
|---|---:|---:|---|---:|
| `CALLE` | 862 | 862 | 5.472→4.604 | 422 |
| `CENTRO_SALUD` | 6 | 6 | 29→28 | 3 |
| `CORREO_ELECTRONICO` | 469 | 469 | 472→469 | 234 |
| `EDAD_SUJETO_ASISTENCIA` | 1.035 | 0 (excluida) | — | 0 |
| `FAMILIARES_SUJETO_ASISTENCIA` | 243 | 243 | 376→274 | 124 |
| `FECHAS` | 1.231 | 0 (excluida) | — | 0 |
| `HOSPITAL` | 255 | 255 | 1.090→1.086 | 136 |
| `ID_ASEGURAMIENTO` | 391 | 391 | 1.150→391 | 196 |
| `ID_CONTACTO_ASISTENCIAL` | 77 | 77 | 77→77 | 40 |
| `ID_SUJETO_ASISTENCIA` | 567 | 525 (42 sin dígitos conservados) | 529→525 | 253 |
| `ID_TITULACION_PERSONAL_SANITARIO` | 471 | 471 | 1.382→471 | 224 |
| `INSTITUCION` | 98 | 98 | 308→414 | 53 |
| `NOMBRE_PERSONAL_SANITARIO` | 1.000 | 1.000 | 3.209→3.053 | 501 |
| `NOMBRE_SUJETO_ASISTENCIA` | 1.009 | 1.009 | 1.561→1.539 | 503 |
| `NUMERO_FAX` | 15 | 15 | 33→29 | 8 |
| `NUMERO_TELEFONO` | 58 | 58 | 136→118 | 31 |
| `OTROS_SUJETO_ASISTENCIA` | 9 | 0 (excluida) | — | 0 |
| `PAIS` | 713 | 713 | 727→743 | 326 |
| `PROFESION` | 24 | 24 | 52→35 | 16 |
| `SEXO_SUJETO_ASISTENCIA` | 925 | 0 (excluida) | — | 0 |
| `TERRITORIO` | 1.875 | 1.875 | 2.092→2.271 | 903 |
| **Total** | **11.333** | **8.091** | **18.695→16.127** | **3.973** |

**Validación final:**
- Integridad BIO en todas las carpetas: 0 desalineaciones, 0 `I-` huérfanas, 0 tokens vacíos o con espacios.
- `augmented_baseline` es idéntico a `processed` (500/250/250). dev/test son idénticos en los tres escenarios.
- Tokens `O` alterados: 0. Secuencia de tipos de entidad alterada: 0. Categorías excluidas modificadas: 0. `ID_SUJETO` sin dígitos modificados: 0.
- **Concordancia de género externa** (nombre nuevo del paciente vs. SEXO real del documento, en docs con sexo inequívoco y nombre con género reconocible):

  | Escenario | Antes del fix | Después del fix |
  |---|---:|---:|
  | `augmented_100` | 484/520 (93,1 %) | **489/489 (100 %)** |
  | `augmented_50` | 251/271 (92,6 %) | **258/258 (100 %)** |

- La concordancia **interna** (nombre original vs. nuevo) baja a 1.404/1.432 (98,0 %) en `augmented_100`. Es esperado: en 28 casos el nombre sustituto original de MEDDOCAN no concuerda con el sexo del paciente, y ahora manda el sexo. Coincide con los ~29 casos (2,0 %) que señaló la revisión externa.

**Estado:** ✅ **Fase 2 cerrada y congelada** (seed 42). Los datos de `processed/` y `augmented_*/` no deben regenerarse sin registrarlo en esta bitácora.

**Pendiente para la siguiente etapa (03, fuera de alcance aquí):** rutas por defecto, truncación a 512 subtokens (documentos largos), uso exclusivo del test para la evaluación final y métricas por etiqueta con `seqeval`.

### 2026-11-11 — Fase 3: diseño final de `03_train_ner.py`

**Contexto:** la Fase 2 quedó congelada el 2026-11-04. El 2026-11-11 se retomó el pipeline para cerrar el diseño experimental de la Fase 3. El script `03_train_ner.py` ya existía, pero sus valores por defecto apuntaban a `data/augmented/`; se ajustaron a los tres escenarios congelados.

**Objetivo:** entrenar BETO y XLM-RoBERTa en los tres escenarios (`augmented_baseline`, `augmented_50`, `augmented_100`) y seleccionar el mejor checkpoint por F1 de validación, manteniendo `test.jsonl` aislado para la Fase 4.

**Decisiones de diseño:**
- Modelos: `dccuchile/bert-base-spanish-wwm-cased` (BETO) y `xlm-roberta-base`, con `AutoModelForTokenClassification`.
- Entrada: `--data-dir` (default `data/augmented_baseline`); solo `train.jsonl` y `dev.jsonl`. El script no abre `test.jsonl`.
- Salida: `--output-dir` (default `models/ner_model`), con pesos, tokenizador, `labels.json` y `03_train_summary.json`.
- Ventana: `max_length=512`, `stride=128`, `is_split_into_words=True`, `return_overflowing_tokens=True`.
- Alineación BIO: el primer sub-token de cada palabra recibe la etiqueta; los sub-tokens siguientes y los especiales reciben `-100`; las palabras del solapamiento ya cubiertas por una ventana anterior también reciben `-100` (`max_word_id_seen`).
- Etiquetas: se construyen desde train+dev; el vocabulario real es de **44 clases** (22 tipos + `O`, sin `I-ID_CONTACTO_ASISTENCIAL`). El comentario del script dice 45; se corregirá en la próxima edición.
- Trainer: 5 épocas, batch 16, lr 2e-5, weight decay 0.01, evaluación por época, `load_best_model_at_end`, `metric_for_best_model="f1"`, `fp16` solo si CUDA, semilla 42.
- Smoke test: `--smoke-test` recorta train/dev a 8 documentos y ejecuta 1 época.

**Estado:** diseño cerrado, **no ejecutado** _(Nota: estado en el momento del diseño; ver sesiones posteriores de ejecución en Colab y Fase 5)_. En `models/` solo hay `.gitkeep`; no existen `03_train_summary.json` ni checkpoints. El `venv` actual no tiene `torch`, `transformers`, `datasets`, `accelerate` ni `seqeval`, y la máquina no tiene GPU NVIDIA (AMD Radeon 780M), por lo que el entrenamiento sería CPU-only y lento.

**Pendiente:** instalar el stack, ejecutar `--smoke-test`, entrenar las 6 combinaciones (3 escenarios × 2 modelos) en GPU/Colab, corregir el comentario de 45 clases, hacer estricto `label2id` (no mapear etiquetas desconocidas a `O`) y añadir warmup/early stopping.

### 2026-11-11 — Fase 4: diseño de la evaluación estricta (`04_evaluate_ner.py`)

**Contexto:** la Fase 4 no existía. Se creó `scripts/04_evaluate_ner.py` para evaluar exclusivamente `test.jsonl` y comparar los 6 modelos de la Fase 3.

**Objetivo:** medir el rendimiento oficial a nivel de entidad con `seqeval`, reconstruyendo las ventanas de sub-tokens y agregando las predicciones a nivel de documento/palabra.

**Decisiones de diseño:**
- **Aislamiento del test:** el script solo acepta un archivo llamado `test.jsonl`; `--allow-non-test` queda para un futuro gold set peruano.
- **Carga:** pesos, tokenizador y `labels.json` desde `--model-dir`; `max_length` y `stride` se heredan de `03_train_summary.json` si existe (512/128 por defecto).
- **Ventanas:** misma tokenización con overflow que 03. Agregación `first` por defecto (primer sub-token de la primera ventana que contiene la palabra), que replica la regla de alineación del entrenamiento; `mean` opcional para promediar probabilidades de ventanas solapadas.
- **BIO:** las etiquetas gold se validan y no se reparan; las predicciones se reparan (`I-` huérfana → `B-`) y se reporta el número de reparaciones.
- **Métricas:** `seqeval` a nivel entidad; micro, macro (etiquetas presentes en gold o predicción, igual que `seqeval`) y weighted; `classification_report` completo; TP/FP/FN y soporte por etiqueta; entidades destacadas `PAIS`, `HOSPITAL`, `SEXO_SUJETO_ASISTENCIA`, `FECHAS` e `ID_*`.
- **Bootstrap:** opcional por documentos (`--bootstrap N`) para IC del micro-F1 y macro-F1.
- **Análisis de errores:** falsos negativos, falsos positivos, confusiones de tipo y errores de frontera, con contexto textual.

**Salidas:** `04_eval_summary.json`, `04_classification_report.txt`, `04_classification_report.json`, `04_per_entity.csv`, `04_bootstrap.json`, `04_errors.csv`, `04_confusions.csv`, `04_predictions.jsonl` (opcional).

**Validación realizada:** sintaxis (`py_compile`), `pyflakes` sin nombres indefinidos, pruebas de lógica pura con stubs (validación/reparación BIO, spans, análisis de errores y bootstrap), contraste exacto contra `seqeval` real en micro/macro/weighted (incluido el caso de etiquetas con soporte 0) y reconstrucción de ventanas con el tokenizador real de BETO sobre 12 documentos del test: cobertura completa de palabras, ventanas contiguas y agregación `first`/`mean` sin palabras faltantes. Todas pasaron.

**Estado:** implementado y validado a nivel de lógica; **no ejecutado end-to-end** porque no hay checkpoints y faltan `torch`/`transformers`/`seqeval` en el entorno. _(Nota: ver sesiones posteriores de ejecución en Colab y Fase 5)_.

**Pendiente:** instalar dependencias, entrenar, ejecutar las 6 evaluaciones con `--bootstrap 1000`, revisar el reporte por entidad y decidir si el test final será MEDDOCAN o un gold set peruano.

### 2026-11-11 — Ejecución en Colab y Generación de Resultados (BETO)

**Contexto:** Se llevó el código a Google Colab (GPU Tesla T4) para ejecutar el pipeline de entrenamiento (Fase 3) y evaluación estricta con Bootstrap ($B = 1000$, `seed = 42`) sobre el modelo **BETO** (`dccuchile/bert-base-spanish-wwm-cased`).
- **Entorno de ejecución verificado (`03_train_summary.json` / `04_eval_summary.json`):** Linux (`x86_64`), `Python 3.13.16`, `torch 2.14.1+cu130`, `transformers 4.57.6`, dispositivo `cuda` (`fp16=True`).
- **Nota de trazabilidad sobre `04_eval_summary.json`:** En las corridas `beto_mixed__augmented_50` y `beto_full__augmented_100`, el argumento CLI `--data-dir` se dejó por defecto en `data/augmented_baseline`, pero `test_file` apuntó a `data/augmented_50/test.jsonl` y `data/augmented_100/test.jsonl` respectivamente (archivos bit-a-bit idénticos con `SHA256 = 8b30b80360cce568a9087b754b9a38b33aa278a6be952107eab3c4ef1bc898e3`).

**Resultados Globales Verificados (`test.jsonl`, $N = 250$ docs, 5.661 entidades gold):**

| Escenario | Dev F1 | Test Micro-P | Test Micro-R | Test Micro-F1 | IC 95% Micro-F1 | Test Macro-P | Test Macro-R | Test Macro-F1 | IC 95% Macro-F1 | Test Weighted-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (0%)** | 0.9401 | 0.9310 | 0.9456 | **0.9382** | `[0.9288, 0.9467]` | 0.7074 | 0.7271 | **0.7163** | `[0.7043, 0.7301]` | 0.9363 |
| **Mixto (50%)** | 0.9363 | 0.9316 | 0.9410 | **0.9363** | `[0.9274, 0.9448]` | 0.7023 | 0.7183 | **0.7091** | `[0.6976, 0.7213]` | 0.9337 |
| **Full (100%)** | 0.7827 | 0.7218 | 0.8419 | **0.7772** | `[0.7691, 0.7843]` | 0.5794 | 0.5994 | **0.5873** | `[0.5750, 0.6004]` | 0.8298 |

**Hallazgos clave por entidad (Domain Shift — progresión `Baseline → Mixto → Full`):**
- **Identificadores regulatorios europeos (colapso en `Full`):**
  - `ID_ASEGURAMIENTO` ($n=199$): `0.9223 → 0.9223 → 0.0052`
  - `ID_TITULACION_PERSONAL_SANITARIO` ($n=232$): `0.9175 → 0.9133 → 0.0067`
- **Entidades institucionales y geográficas (degradación progresiva o abrupta):**
  - `INSTITUCION` ($n=53$): `0.3810 → 0.2586 → 0.1261` (degradación ya visible en el escenario Mixto).
  - `CALLE` ($n=417$): `0.8915 → 0.8715 → 0.6710`
  - `HOSPITAL` ($n=129$): `0.7708 → 0.7762 → 0.5841` (ligera mejora en Mixto, caída en Full).
  - `PAIS` ($n=341$): `0.9656 → 0.9576 → 0.8958`
  - `TERRITORIO` ($n=930$): `0.9612 → 0.9570 → 0.9146`
  - `FAMILIARES_SUJETO_ASISTENCIA` ($n=118$): `0.6480 → 0.6441 → 0.6211`
- **Clases raras con `F1 = 0.0000` en los 3 escenarios:** `CENTRO_SALUD` ($n=6$ en test), `NUMERO_FAX` ($n=7$), `OTROS_SUJETO_ASISTENCIA` ($n=7$) y `PROFESION` ($n=9$). Además, `ID_EMPLEO_PERSONAL_SANITARIO` tiene soporte $0$ en `train` y `test` ($1$ mención en `dev`).

**Limitaciones estadísticas y metodológicas:**
1. **Semilla única (`seed=42`):** No se cuenta con varianza inter-semilla de inicialización/orden de lotes.
2. **Bootstrap a nivel de documento ($B=1000$):** Captura la incertidumbre muestral sobre los $N=250$ documentos de prueba, pero **la superposición de los intervalos de confianza al 95% entre Baseline y Mixto no constituye por sí misma una prueba formal de hipótesis de equivalencia o no-inferioridad**; para contrastar formalmente la diferencia se requeriría un test bootstrap pareado sobre $\Delta\text{F1} = \text{F1}_{\text{mixto}} - \text{F1}_{\text{baseline}}$ documento a documento.
3. **Inestabilidad del Macro-F1:** Las 4 clases de soporte ultra-bajo con `F1 = 0.0000` imponen un techo artificial al Macro-F1 (~0.716).
4. **Ausencia de un conjunto de prueba clínico peruano:** `test.jsonl` contiene exclusivamente entidades españolas originales de MEDDOCAN.

**Conclusión (corregida y ajustada a la evidencia):**
El escenario **Mixto (50%)** conserva el rendimiento global sobre el corpus MEDDOCAN original (Micro-F1 `0.9363` vs. `0.9382` en Baseline, con comportamiento heterogéneo en ciertas entidades como `INSTITUCION`, que cae de `0.3810` a `0.2586`); sin embargo, **la adaptación efectiva a entidades clínicas peruanas reales no está medida empíricamente por ausencia de un conjunto de prueba peruano (`gold test set`)**.

### 2026-11-12 — Ejecución del notebook de F5 y auditoría de cierre (tesis-safe)

**Contexto:** Auditoría integral de ingeniería de ML y revisión metodológica para dejar el repositorio en estado reproducible y verificable («tesis-safe») en la rama `fix/cierre-fase5`, sin alterar datos ni métricas existentes.

**Avances realizados:**
1. **Generalización y ejecución de `notebooks/01_resultados.ipynb`:**
   - `load_runs_strict()` generalizado para descubrir automáticamente cualquier corrida en `results/*/04_eval_summary.json` + `04_bootstrap.json` (preparado para incorporar `XLM-RoBERTa` sin cambios de código).
   - Derivación del escenario exclusivamente desde `scenario` y `Path(test_file).parent.name` (ignorando el campo `data_dir` erróneo de los summaries de `mixed` y `full`).
   - Notebook ejecutado de principio a fin (`execution_count` 1..6) con tablas HTML y figuras PNG embebidas en las salidas de celda, y celda final de limitaciones metodológicas añadida.
2. **Artefactos de Fase 5 generados en `results/`:**
   - `results/05_resultados_comparativos.csv`: tabla consolidada plana de métricas globales e IC 95%.
   - `results/05_tabla_resultados.tex`, `results/05_tabla_micro_f1.tex`, `results/05_tabla_por_entidad.tex`: tablas LaTeX exportadas con `escape=True` y porcentajes escapados (`95\%`).
   - `results/05_f1_por_escenario.png`: barras comparativas Micro-F1 vs. Macro-F1 con barras de error asimétricas (IC 95%).
   - `results/05_f1_entidades_clave.png`: mapa de calor completo de las **21 entidades evaluadas** en `test.jsonl`.
   - `results/05_bootstrap_micro_f1.png` y `results/05_bootstrap_macro_f1.png`: distribuciones bootstrap ($B=1000$).
3. **Endurecimiento de scripts (`Fase 3` y `Fase 4`):**
   - `scripts/03_train_ner.py`: corregido comentario de vocabulario (`44` etiquetas en MEDDOCAN); validación estricta en `align_labels_with_tokens` que lanza `ValueError` ante cualquier etiqueta desconocida (eliminando el mapeo silencioso a `"O"`); añadidos flags opcionales `--warmup-ratio` (default `0.0`) y `--early-stopping-patience` (default `None`, usando `EarlyStoppingCallback`), registrados en `03_train_summary.json` para futuras corridas.
   - `scripts/04_evaluate_ner.py`: añadido warning explícito cuando `--data-dir` no coincide con el directorio padre de `--test-file`; añadido cálculo y registro de `test_file_sha256` (`hashlib.sha256`) en `04_eval_summary.json` para futuras evaluaciones.
4. **Política de versionado y documentación:**
   - `.gitignore`: actualizado para ignorar `.jsonl` pesados, `data/archive/`, `models/smoke_*/`, checkpoints intermedios y pesos binarios (`*.safetensors`, `*.bin`, `*.pt`), manteniendo versionados los resúmenes JSON y todo `results/`.
   - Creados `docs/protocolo_experimental.md`, `docs/taxonomia_entidades.md`, `models/README.md`, `results/README.md` y `README.md` raíz.

## 6. Decisiones técnicas (ADR)

| Fecha | Decisión | Alternativas | Motivo |
|---|---|---|---|
| 2026-10-29 | JSONL + CoNLL como formatos de salida | CSV, XML | Compatible con HF y fácil de inspeccionar |
| 2026-10-29 | Modelo base BETO/BERTIN | mBERT, RoBERTa-es | Buen rendimiento en español clínico |
| 2026-10-29 | Mantener la salida JSON de 01 (`id_documento`, `tokens`, `etiquetas_ner`) | Reescribir 01 para emitir JSONL | El conversor ya fue ejecutado y validado; 02 adapta el formato |
| 2026-11-03 | Aumentar solo `train`; `dev`/`test` intactos | Aumentar los tres splits | Evitar fuga de información y evaluar sobre la distribución original; la versión Faker actual no lo cumple |
| 2026-11-03 | 02 debe emitir JSONL (`id`, `tokens`, `tags`) | JSON, CoNLL | Compatible con `03_train_ner.py` y `datasets`; la versión Faker actual aún no lo cumple |
| 2026-11-03 | Faker + `ProviderPeruano` como base de aumentación | Diccionarios estáticos, generación libre | Más variedad y mantenimiento; el provider peruano aporta el dominio |
| 2026-11-03 | Re-tokenizar cada reemplazo y reconstruir BIO | Conservar el número de tokens del span | Evita tokens con espacios y valores truncados |
| 2026-11-04 | 01 emite JSONL (`id`, `tokens`, `tags`) directamente; reemplaza la decisión del 2026-10-29 | Mantener JSON + adaptador | Un único formato en todo el pipeline |
| 2026-11-04 | Alineación por solapamiento, etiquetando el token completo; subdivisión solo si hay 2+ entidades | Descartar entidades parciales; subdividir siempre | 100 % de recuperación; el ruido en los límites se documenta como limitación |
| 2026-11-04 | Generación libre de longitud en 02 | Igualar la longitud original (recorte/relleno) | El recorte producía entidades incoherentes; las etiquetas se recalculan igual |
| 2026-11-04 | Excluir FECHAS, EDAD, SEXO y OTROS de la aumentación | Generadores para todas las etiquetas | Preservar el vocabulario y la distribución originales (reemplaza la propuesta de §8 de aumentarlas) |
| 2026-11-04 | Faker `es_MX` + gazetteers peruanos (en lugar de `ProviderPeruano` con `Faker()` en `en_US`) | Solo gazetteers; solo Faker | Faker aporta entropía a nombres/profesiones/correos; los gazetteers, el dominio peruano |
| 2026-11-04 | Concordancia de género por nombre original, con respaldo en la mención SEXO del documento | Género aleatorio | Coherencia entre el nombre y el sexo del paciente en el texto |
| 2026-11-04 | Identificadores sin prefijos de texto (sin "ESSALUD"/"CMP") | Prefijos dentro del span | Evitar que el modelo aprenda una palabra clave en lugar del patrón numérico |
| 2026-11-04 | Tres escenarios en carpetas separadas (`augmented_baseline`, `augmented_100`, `augmented_50`) | Una sola carpeta `augmented/` | Experimentos comparables y reproducibles |
| 2026-11-04 | Género del nombre del paciente: el SEXO del documento tiene prioridad sobre el nombre original (reemplaza la prioridad inversa) | Priorizar el nombre original | Los nombres sustitutos de MEDDOCAN no siempre concuerdan con el sexo; el texto (pronombres, «la paciente») sí |
| 2026-11-04 | No aumentar `ID_SUJETO_ASISTENCIA` sin dígitos (Opción A) | Aumentarlos y solo documentarlo (Opción B) | Convertir «lactante» en un DNI cambia la semántica del span |
| 2026-11-04 | Congelar la Fase 2 (seed 42); 03 pasa a la siguiente etapa | Seguir iterando 02 en paralelo con 03 | Datos estables para experimentos comparables |
| 2026-11-12 | Versionar en Git solo metadatos (`01/02/03_summary.json`, `config.json`, `labels.json`) y artefactos de evaluación (`results/`), excluyendo `.jsonl` y pesos binarios (`*.safetensors`) | Versionar datasets y pesos con Git LFS o en el repo | Mantener el repositorio ligero y auditable; los `.jsonl` son 100% reproducibles con `seed=42` |

## 7. Experimentos

| ID | Fecha | Modelo | Datos | Épocas | Dev F1 | Test Micro-F1 (IC 95%) | Test Macro-F1 (IC 95%) | Test Weighted-F1 | Notas |
|---|---|---|---|---:|---:|---|---|---:|---|
| E01 | 2026-11-11 | BETO | Baseline (0%) | 5 | 0.9401 | 0.9382 `[0.9288, 0.9467]` | 0.7163 `[0.7043, 0.7301]` | 0.9363 | Baseline oficial sobre `test.jsonl` intacto |
| E02 | 2026-11-11 | BETO | Mixto (50%) | 5 | 0.9363 | 0.9363 `[0.9274, 0.9448]` | 0.7091 `[0.6976, 0.7213]` | 0.9337 | Rendimiento global conservado; caída puntual en `INSTITUCION` (`0.381→0.259`) |
| E03 | 2026-11-11 | BETO | Full (100%) | 5 | 0.7827 | 0.7772 `[0.7691, 0.7843]` | 0.5873 `[0.5750, 0.6004]` | 0.8298 | Caída severa por domain shift en `ID_ASEGURAMIENTO`, `ID_TITULACION_*`, `CALLE` y `HOSPITAL` |
| E04–E06 | _(pendiente)_ | XLM-RoBERTa | Baseline / 50% / 100% | 5 | — | — | — | — | **TODO (3/6 corridas pendientes):** requiere decisión humana (ver §10) |

## 8. Problemas y soluciones

| Fecha | Problema | Solución | Estado |
|---|---|---|---|
| 2026-10-29 | 01 pierde 1.336 menciones (5,9 %) por tokens con puntuación; `SEXO_SUJETO_ASISTENCIA` pierde 925 de 1.841 | Alineación por solapamiento de caracteres + subdivisión de tokens con 2+ entidades; 0 pérdidas | Resuelto (2026-11-04) |
| 2026-11-03 | El borrador de 02 esperaba `.jsonl` con `id`/`tags`; 01 genera `.json` con `id_documento`/`etiquetas_ner` | Adaptador JSON/JSONL y normalización de claves en 02 | Resuelto |
| 2026-11-03 | 02 encontraba `I-` huérfanos por fronteras mal alineadas de 01 | Reparación de `I-` huérfanas en 01 y en la carga de 02 (0 reparaciones necesarias) | Resuelto |
| 2026-11-03 | La versión Faker aumentó `dev` y `test` | Solo se aumentan los splits pedidos; dev/test se copian intactos (verificado) | Resuelto (2026-11-04) |
| 2026-11-03 | Salida JSON (`id_documento`, `etiquetas_ner`) incompatible con `03_train_ner.py` (JSONL `id`, `tags`) | 01 y 02 emiten JSONL `id`/`tokens`/`tags` | Resuelto (2026-11-04) |
| 2026-11-03 | Sin semilla; `random` y Faker no son reproducibles | `Faker.seed_instance` + `random.Random(seed)` por split/copia | Resuelto (2026-11-04) |
| 2026-11-03 | 351 tokens con espacios en train por nombres/ciudades multi-palabra | Tokenización de cada reemplazo + reconstrucción BIO; invariante verificado en 02 | Resuelto (2026-11-04) |
| 2026-11-03 | `B-HOSPITAL` colapsa a `Hospital`/`Clínica`/`Instituto`/`Centro` | Reemplazo del span completo con ~80 hospitales/clínicas | Resuelto (2026-11-04) |
| 2026-11-03 | Reemplazos multi-token incoherentes (teléfonos, IDs, CALLE, TERRITORIO, PAIS) | Un valor coherente por span, con longitud libre | Resuelto (2026-11-04) |
| 2026-11-03 | Solo 9 de 22 etiquetas se aumentaron | 18 etiquetas con generador (17 presentes en train; `ID_EMPLEO_PERSONAL_SANITARIO` no aparece en train); FECHAS/EDAD/SEXO/OTROS se excluyen por diseño (ver ADR) | Resuelto (2026-11-04) |
| 2026-11-03 | `Faker` no figura en `requirements.txt` | Añadido `faker>=24.0` + instrucción para `es_core_news_sm` | Resuelto (2026-11-04) |
| 2026-11-04 | 2 tokens con espacio (`"EE. UU."`) en `processed/train`; `_emitir_token` era código muerto | `_emitir_token` integrado en 01 | Resuelto |
| 2026-11-04 | 02 recortaba/rellenaba entidades (`'Calle Las Begonias Dpto. 6 315 Dpto.'`, `"Villa El"`) | Generación libre de longitud | Resuelto |
| 2026-11-04 | Nombres sin concordancia de género (`Ernesto` → `Silvia`) | Léxico de género + mención SEXO del documento (concordancia interna 100 %; ver corrección del mapeo `H.`/`M.`) | Resuelto |
| 2026-11-04 | Faker `es_ES` lista `José` como femenino y `María` como masculino | Excepciones forzadas en el léxico | Resuelto |
| 2026-11-04 | `ID_CONTACTO_ASISTENCIAL` generaba celulares; prefijos `ESSALUD`/`CMP` dentro de los IDs | Episodio de 7 dígitos; IDs sin prefijo | Resuelto |
| 2026-11-04 | Gazetteers de baja entropía (riesgo de memorización) | Listas ampliadas (ver sesión del 2026-11-04) | Resuelto |
| 2026-11-04 | Ruido en los límites: un token parcialmente cubierto hereda la etiqueta completa | Se documenta como limitación (1.952 tokens, 0,38 %; train 973, 0,39 %); se mantiene por el 100 % de recuperación | Aceptado |
| 2026-11-04 | `03_train_ner.py` apunta por defecto a `data/augmented/train.jsonl` (ya no se genera) | Valores por defecto actualizados a `data/augmented_baseline` en la etapa 03 | Resuelto (2026-11-11) |
| 2026-11-04 | **Bug:** mapeo de sexo invertido (`m` → masculino; `h` no reconocido); en MEDDOCAN `H.` = hombre y `M.` = mujer | Mapeo corregido + `anciana/o`, `recién nacido/a`, `hembra`; prioridad del SEXO del documento para el paciente. Concordancia externa del 93,1 % al 100 % | Resuelto |
| 2026-11-04 | 42 spans de `ID_SUJETO` sin dígitos (`lactante`, `casada`, …) se convertían en DNI | No se aumentan (Opción A); se documentan como ruido de anotación | Resuelto |
| 2026-11-04 | Apellidos que también son nombres de pila (`Luna`, `Santiago`, `Tito`, `de Jesús`) generaban spans de solo apellidos legibles como nombres del género opuesto | Filtro en los slots de apellido | Resuelto |
| 2026-11-04 | La bitácora decía «Perú pesa 12 de 27» sin aclarar el número de países | Corregido: 27 entradas, 16 países, P(Perú) = 44,4 %; distribución reportada | Resuelto |
| 2026-11-12 | `04_eval_summary.json` de `mixed` y `full` registraron `data_dir="data/augmented_baseline"` aunque `test_file` apuntó al escenario correcto | `notebooks/01_resultados.ipynb` infiere el escenario exclusivamente desde `scenario` y `Path(test_file).parent.name`; `04_evaluate_ner.py` ahora emite `WARNING` y guarda `test_file_sha256` | Resuelto |
| 2026-11-12 | `03_train_ner.py` mapeaba etiquetas desconocidas silenciosamente a `"O"` (`label2id.get(tag, O)`) y tenía comentario erróneo de 45 clases | Validación estricta con `ValueError` si una etiqueta no existe en `label2id` y corrección del comentario a 44 clases | Resuelto |

## 9. Próximos pasos y criterios de aceptación

- [x] Completar la descarga del corpus.
- [x] Ejecutar `01_brat_to_bio.py` y validar los tres archivos BIO.
- [x] Crear/adaptar `02_augmentation.py` a la salida JSON de 01.
- [x] Reemplazar `02_augmentation.py` por una versión basada en Faker + `ProviderPeruano`.
- [x] Ejecutar `02_augmentation.py` y generar `data/augmented/`.
- [x] Corregir los defectos de la versión Faker (train-only, JSONL, semilla, re-tokenización, semántica por etiqueta).
- [x] Re-ejecutar la aumentación solo sobre `train` y revisar una muestra.
- [x] Corregir la pérdida de entidades por puntuación en 01 y regenerar `data/processed/`.
- [x] Generar los escenarios `augmented_baseline`, `augmented_100` y `augmented_50`.
- [x] Fase 3: actualizar las rutas por defecto de `03_train_ner.py`, fijar la ventana 512/128 y aislar `test.jsonl`.
- [x] Entrenar y evaluar las 3 combinaciones de BETO (Baseline / 50% / 100%) en Google Colab (GPU Tesla T4). _(2026-11-11)_
- [x] Definir la métrica principal (F1 por entidad con `seqeval`, micro/macro) y usar el test como conjunto fijo.
- [x] Fase 4: implementar `04_evaluate_ner.py` (agregación por documento, bootstrap y análisis de errores).
- [x] Ejecutar las evaluaciones con `--bootstrap 1000` y documentar los resultados reales para BETO. _(2026-11-11)_
- [x] Fase 5: generalizar y ejecutar `notebooks/01_resultados.ipynb` con salidas guardadas y artefactos `results/05_*`. _(2026-11-12)_
- [x] Documentar la taxonomía de entidades usada (`docs/taxonomia_entidades.md`) y el protocolo experimental (`docs/protocolo_experimental.md`). _(2026-11-12)_
- [ ] **TODO (decisión humana — XLM-RoBERTa):** Definir el cierre de las 3/6 corridas restantes para `xlm-roberta-base` (ver opciones en §10).
  - *Criterio de aceptación:* O bien existen `models/xlmr_{baseline,mixed,full}/03_train_summary.json` + `results/xlmr_*__augmented_*/04_eval_summary.json` y el notebook re-ejecutado incluye los 6 modelos, o bien §2 se actualiza formalmente acotando el estudio experimental a BETO.
- [ ] **TODO (decisión humana — Validación en dominio peruano):** Decidir si se anota un conjunto de prueba clínico peruano (`gold test set`) o se declara como trabajo futuro (ver opciones en §10).
  - *Criterio de aceptación:* O bien se evalúan los 3 modelos sobre un archivo `test_peru.jsonl` con `--allow-non-test`, o bien la tesis reporta explícitamente la limitación de evaluar únicamente sobre el `test.jsonl` español de MEDDOCAN.
- [ ] EDA adicional en `notebooks/` (distribución por etiqueta, longitud de spans antes/después de aumentar).
  - *Criterio de aceptación:* Notebook ejecutado con estadísticas descriptivas de longitud de documentos en tokens/sub-tokens y proporción de documentos que superan 512 sub-tokens.

## 10. Estado de artefactos y decisiones pendientes (TODOs)

| Componente / Artefacto | Estado Actual en el Repositorio | Opciones de Decisión Humana (`TODO`) |
| :--- | :--- | :--- |
| **1. Pesos finales de BETO (`model.safetensors`)** | **No presentes en el repo.** Solo se conservan `config.json`, `labels.json` y `03_train_summary.json` de las corridas de Colab (5 épocas, `dev-F1`: `0.9401` / `0.9363` / `0.7827`). | **Opción A:** Subir los pesos finales desde Google Drive/Colab a Hugging Face Hub o GitHub Releases y registrar los enlaces/hashes en `models/README.md`.<br>**Opción B:** Mantener en Git solo los metadatos y resúmenes de evaluación (política actual en `.gitignore`). |
| **2. Checkpoint local obsoleto (`models/beto_baseline/checkpoints/checkpoint-64`)** | **Obsoleto / Parcial.** Corresponde a una prueba antigua de 1 época (`best_metric = 0.8429`, `eval_runtime = 139.2 s`) y **NO** al modelo final de Colab (5 épocas, `dev-F1 = 0.9401`, `eval_runtime = 5.75 s`). Ignorado por `.gitignore`. | **Acción recomendada:** No usar nunca `checkpoint-64` para inferencia ni evaluación. Puede eliminarse localmente cuando se desee liberar ~420 MB de disco. |
| **3. Corridas de `xlm-roberta-base` (3/6 experimentos)** | **Pendientes.** Las Fases 3 y 4 están completas para BETO (3/6 corridas), pero no se han ejecutado para `xlm-roberta-base`. | **Opción A:** Ejecutar las 3 corridas de `xlm-roberta-base` en Colab T4 y re-ejecutar `notebooks/01_resultados.ipynb` (que ya soporta múltiples modelos automáticamente).<br>**Opción B:** Acotar el alcance del documento de tesis en §2 a BETO y dejar `XLM-RoBERTa` como línea futura. |
| **4. Conjunto de prueba clínico peruano (`Gold Test Set`)** | **Ausente.** Toda la evaluación de Fase 4 se realizó sobre `test.jsonl` original de MEDDOCAN (España). | **Opción A:** Anotar un mini-corpus clínico peruano (30–50 notas desidentificadas) y evaluarlo con `04_evaluate_ner.py --allow-non-test`.<br>**Opción B:** Mantener la evaluación sobre MEDDOCAN declarando explícitamente que mide retención del dominio fuente y vulnerabilidad al reemplazo sintético, sin medir el recall real en hospitales peruanos. |
| **5. Versionado de datos `.jsonl` (`data/processed/`, `data/augmented_*/`)** | **Excluidos de Git en `.gitignore`.** Se versionan únicamente `01_summary.json` y `02_summary.json`. | **Opción A (actual):** No versionar los `.jsonl` (~35 MB por carpeta) dado que son 100% reproducibles ejecutando `01_brat_to_bio.py` y `02_augmentation.py` (`seed=42`).<br>**Opción B:** Usar DVC o un archivo comprimido en Releases si el jurado requiere descarga directa sin ejecutar los scripts 01 y 02. |
| **6. Notebook de Fase 5 (`notebooks/01_resultados.ipynb`)** | **Ejecutado y verificado.** Contiene las 6 celdas de código con salidas embebidas (`execution_count` 1..6) y exporta todos los artefactos `results/05_*`. | Listo para incluir tablas (`.tex`/`.csv`) y figuras (`.png`) en el manuscrito de tesis. |

## 11. Referencias

- SPACCC_MEDDOCAN — anotación de datos personales en casos clínicos en español.
- Hugging Face Transformers — `AutoModelForTokenClassification`.
- spaCy (`es_core_news_sm`) — tokenización en español.
- Faker 40.44.0 — generación de datos sintéticos con locale `es_MX` (nombres, profesiones, correos); el léxico de género usa los providers `es_ES`, `es_MX`, `es_CO`, `es_AR` y `es_CL`.

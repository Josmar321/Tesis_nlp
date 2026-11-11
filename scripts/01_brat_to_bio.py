#!/usr/bin/env python3
"""01_brat_to_bio.py - Convierte SPACCC_MEDDOCAN (BRAT) a BIO en JSONL.

Lee:
    data/raw/corpus/{split}/brat/*.txt
    data/raw/corpus/{split}/brat/*.ann

Escribe:
    data/processed/{split}.jsonl
    {"id": "...", "tokens": [...], "tags": [...]}

Alineacion:
    Las entidades BRAT se marcan a nivel de caracter y luego se alinean con
    los tokens de spaCy por solapamiento. Si un token contiene dos entidades
    (por ejemplo, "08017.Espana"), se divide en subtokens para no perder
    ninguna mencion.

    Ademas, todo token que spaCy haya dejado con espacios internos (por
    ejemplo, la excepcion de tokenizacion "EE. UU.") se divide por espacios
    en subtokens BIO (B- + I-), garantizando que ningun token del JSONL
    contenga espacios y que la relacion token/tag sea 1:1.

Limitacion conocida (ruido en los limites):
    Cuando una entidad cubre solo PARTE de un token de spaCy y no hay otra
    entidad dentro del mismo token (p. ej. el token "Madrid," cuando la
    entidad BRAT es solo "Madrid", o "NHC:2569870" cuando la entidad es solo
    el numero), el token COMPLETO recibe la etiqueta de la entidad. Esto
    introduce un ligero ruido en los bordes del span (caracteres de
    puntuacion o prefijos etiquetados como entidad). Se mantiene esta
    decision deliberadamente porque garantiza la recuperacion del 100% de
    las menciones BRAT (B- generadas == entidades BRAT) sin descartar
    ninguna. Solo cuando un token contiene dos o mas entidades distintas se
    subdivide a nivel de caracter.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import spacy

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_ENTRADA = BASE_DIR / "data" / "raw" / "corpus"
RUTA_SALIDA = BASE_DIR / "data" / "processed"

print("Cargando modelo spaCy (es_core_news_sm)...")
nlp = spacy.load("es_core_news_sm")


def parsear_anotaciones(ruta_ann: Path) -> list[dict]:
    """Extrae las entidades de un archivo .ann de BRAT."""
    entidades = []
    with ruta_ann.open("r", encoding="utf-8") as f:
        for linea in f:
            if not linea.startswith("T"):
                continue
            partes = linea.rstrip("\n").split("\t")
            if len(partes) != 3:
                continue
            info = partes[1].split()
            if len(info) < 3:
                continue
            etiqueta = info[0]
            inicio = int(info[1])
            fin = int(info[-1])
            entidades.append(
                {
                    "etiqueta": etiqueta,
                    "inicio": inicio,
                    "fin": fin,
                    "texto": partes[2],
                }
            )
    return sorted(entidades, key=lambda x: (x["inicio"], -(x["fin"] - x["inicio"])))


def _emitir_token(texto: str, tag: str, salida: list[dict]) -> None:
    """Anade un token; si contiene espacios, lo divide en subtokens BIO.

    Ej.: ("EE. UU.", "B-PAIS") -> [("EE.", "B-PAIS"), ("UU.", "I-PAIS")].
    Si el tag de entrada es I-X, todas las piezas quedan como I-X.
    """
    piezas = texto.split()
    if not piezas:
        return
    if tag == "O":
        salida.extend({"token": pieza, "etiqueta": "O"} for pieza in piezas)
        return
    etiqueta = tag[2:]
    for i, pieza in enumerate(piezas):
        prefijo = tag[:2] if i == 0 else "I-"
        salida.append({"token": pieza, "etiqueta": prefijo + etiqueta})


def convertir_a_bio(texto: str, entidades: list[dict]) -> tuple[list[dict], dict]:
    """Tokeniza y alinea por solapamiento, dividiendo tokens con 2+ entidades."""
    doc = nlp(texto)
    marcas: list[int | None] = [None] * len(texto)
    skipped = 0

    for idx, ent in enumerate(entidades):
        start = max(0, ent["inicio"])
        end = min(len(texto), ent["fin"])
        if start >= end:
            skipped += 1
            continue
        if any(marcas[i] is not None for i in range(start, end)):
            skipped += 1
            continue
        for i in range(start, end):
            marcas[i] = idx

    tokens_bio: list[dict] = []
    split_tokens = 0

    for token in doc:
        if token.is_space:
            continue
        start = token.idx
        end = start + len(token.text)

        # Agrupar los caracteres del token por entidad (None = O).
        runs: list[list] = []
        for i in range(start, end):
            m = marcas[i] if i < len(marcas) else None
            if not runs or runs[-1][0] != m:
                runs.append([m, i, i + 1])
            else:
                runs[-1][2] = i + 1

        entidades_en_token = {m for m, _, _ in runs if m is not None}

        if len(entidades_en_token) <= 1:
            # Un solo entity (o ninguno): conservar el token completo.
            # NOTA (ruido en limites): si la entidad cubre solo parte del
            # token, el token entero hereda la etiqueta. Ver docstring.
            tag = "O"
            for i in range(start, end):
                m = marcas[i] if i < len(marcas) else None
                if m is not None:
                    ent = entidades[m]
                    prefijo = "B-" if i == ent["inicio"] else "I-"
                    tag = prefijo + ent["etiqueta"]
                    break
            _emitir_token(token.text, tag, tokens_bio)
        else:
            # Varias entidades dentro del mismo token: dividir en subtokens.
            split_tokens += 1
            for m, rs, re_ in runs:
                segmento = texto[rs:re_]
                if not segmento or segmento.isspace():
                    continue
                if m is None:
                    tag = "O"
                else:
                    ent = entidades[m]
                    prefijo = "B-" if rs == ent["inicio"] else "I-"
                    tag = prefijo + ent["etiqueta"]
                _emitir_token(segmento, tag, tokens_bio)

    stats = {"skipped": skipped, "split_tokens": split_tokens}
    return tokens_bio, stats


def limpiar_etiquetas_bio(
    tokens_con_etiquetas: list[dict],
) -> tuple[list[str], list[str], int]:
    """Repara I- huerfanas. Devuelve tokens, etiquetas y numero de reparaciones."""
    tokens_limpios = []
    etiquetas_limpias = []
    prev = "O"
    reparaciones = 0
    for item in tokens_con_etiquetas:
        tag = item["etiqueta"]
        if tag.startswith("I-"):
            tipo = tag[2:]
            if prev == "O" or prev[2:] != tipo:
                tag = "B-" + tipo
                reparaciones += 1
        tokens_limpios.append(item["token"])
        etiquetas_limpias.append(tag)
        prev = tag
    return tokens_limpios, etiquetas_limpias, reparaciones


def procesar_corpus(split: str) -> dict:
    ruta_brat = RUTA_ENTRADA / split / "brat"
    if not ruta_brat.exists():
        return {}
    archivos = sorted(ruta_brat.glob("*.txt"))
    print(f"Procesando {len(archivos)} documentos de '{split}'...")

    docs = []
    total_brat = 0
    total_b = 0
    total_orphans = 0
    total_skipped = 0
    total_split = 0
    space_tokens = 0
    label_counts = Counter()

    for ruta_txt in archivos:
        ruta_ann = ruta_txt.with_suffix(".ann")
        texto = ruta_txt.read_text(encoding="utf-8")
        entidades = parsear_anotaciones(ruta_ann)
        tokens_etiquetados, stats = convertir_a_bio(texto, entidades)
        tokens, tags, repairs = limpiar_etiquetas_bio(tokens_etiquetados)

        if len(tokens) != len(tags):
            raise ValueError(f"{ruta_txt.name}: tokens={len(tokens)} tags={len(tags)}")
        if any(not t for t in tokens):
            raise ValueError(f"{ruta_txt.name}: token vacio")

        total_brat += len(entidades)
        total_b += sum(1 for t in tags if t.startswith("B-"))
        total_orphans += repairs
        total_skipped += stats["skipped"]
        total_split += stats["split_tokens"]
        space_tokens += sum(1 for t in tokens if any(c.isspace() for c in t))
        label_counts.update(t[2:] for t in tags if t.startswith("B-"))

        docs.append({"id": ruta_txt.stem, "tokens": tokens, "tags": tags})

    perdidas = total_brat - total_b
    print(
        f"  BRAT={total_brat} B={total_b} perdidas={perdidas} "
        f"skipped={total_skipped} tokens_divididos={total_split} "
        f"reparaciones={total_orphans} tokens_con_espacio={space_tokens}"
    )
    return {
        "docs": docs,
        "stats": {
            "docs": len(docs),
            "brat": total_brat,
            "b": total_b,
            "perdidas": perdidas,
            "skipped": total_skipped,
            "split_tokens": total_split,
            "orphans": total_orphans,
            "space_tokens": space_tokens,
            "labels": dict(label_counts),
        },
    }


def main() -> None:
    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)
    resumen = {}
    for split in ("train", "dev", "test"):
        resultado = procesar_corpus(split)
        if not resultado:
            print(f"[aviso] No se encontro '{split}' en {RUTA_ENTRADA}")
            continue
        salida = RUTA_SALIDA / f"{split}.jsonl"
        with salida.open("w", encoding="utf-8", newline="\n") as fh:
            for doc in resultado["docs"]:
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
        resumen[split] = resultado["stats"]
        print(f"[ok] {salida} ({len(resultado['docs'])} documentos)\n")

    (RUTA_SALIDA / "01_summary.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[ok] Resumen: {RUTA_SALIDA / '01_summary.json'}")


if __name__ == "__main__":
    main()

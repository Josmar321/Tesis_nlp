#!/usr/bin/env python3
"""02_augmentation.py - Peruaniza el corpus BIO de SPACCC_MEDDOCAN.

Lee:
    data/processed/{split}.jsonl            (salida de 01_brat_to_bio.py)

Escribe:
    data/augmented_<escenario>/{split}.jsonl
    data/augmented_<escenario>/02_summary.json

Escenarios de la tesis (presets con --scenario):
    1. Baseline (copia exacta):  python scripts/02_augmentation.py --scenario baseline
       -> data/augmented_baseline/   (train/dev/test sin cambios, rate=0)
    2. Aumentado (100%):         python scripts/02_augmentation.py --scenario full
       -> data/augmented_100/        (train aumentado rate=1.0; dev/test originales)
    3. Mixto (50%):              python scripts/02_augmentation.py --scenario mixed
       -> data/augmented_50/         (train aumentado rate=0.5; dev/test originales)

    Tambien se puede usar en modo manual: --splits, --rate, --out-dir.

Principios de diseno:
    * Generacion libre de longitud: el valor sintetico se inserta con su
      longitud natural (1..N tokens) y las etiquetas B-/I- del span se
      recalculan dinamicamente. No hay recorte ni relleno.
    * Concordancia de genero: para el paciente manda el sexo real del
      documento (mencion SEXO; en MEDDOCAN H. = hombre, M. = mujer) y, si no
      existe, el genero del nombre original. Para el personal sanitario se
      usa el genero del nombre original.
    * ID_SUJETO_ASISTENCIA sin digitos (descriptores como "lactante",
      "casada", "raza caucasica") no se aumenta.
    * Identificadores sin prefijos de texto quemados (sin "ESSALUD"/"CMP").
    * Las categorias FECHAS, EDAD, SEXO y OTROS no se modifican, para
      preservar su vocabulario y distribucion original.
    * dev/test nunca se aumentan salvo que se pidan explicitamente; esto es
      lo que evita la fuga de datos (data leakage) hacia la evaluacion.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from faker import Faker

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_ENTRADA = BASE_DIR / "data" / "processed"
RUTA_DATA = BASE_DIR / "data"

ESCENARIOS: dict[str, dict[str, Any]] = {
    "baseline": {"splits": "train,dev,test", "rate": 0.0, "out_dir": "augmented_baseline"},
    "full": {"splits": "train", "rate": 1.0, "out_dir": "augmented_100"},
    "mixed": {"splits": "train", "rate": 0.5, "out_dir": "augmented_50"},
}

EXCLUIDAS = {"FECHAS", "EDAD_SUJETO_ASISTENCIA", "SEXO_SUJETO_ASISTENCIA", "OTROS_SUJETO_ASISTENCIA"}

# Tokenizador simple para los valores sinteticos: separa puntuacion
# (, ; : ( ) ") como tokens propios, igual que hace spaCy en el corpus.
TOKEN_RE = re.compile(r"[^\s,;:()\"]+|[,;:()\"]")


def tokenizar(texto: str) -> list[str]:
    return TOKEN_RE.findall(texto)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s.lower().strip(".,;:")


# ---------------------------------------------------------------------------
# Gazetteers peruanos
# ---------------------------------------------------------------------------
APELLIDOS_PE = [
    # Andinos / amazonicos
    "Quispe", "Mamani", "Huamán", "Condori", "Choque", "Apaza", "Huanca", "Ticona",
    "Cusi", "Yupanqui", "Pari", "Chambi", "Ccahuana", "Layme", "Atao", "Huillca",
    "Puma", "Inca", "Taipe", "Paucar", "Ccorimanya", "Huaman", "Cóndor", "Yauri",
    "Chuquimia", "Ancco", "Ayma", "Calcina", "Coaquira", "Llanos", "Tito", "Huarcaya",
    "Sullca", "Ccopa", "Chávarry", "Asto", "Poma", "Rimachi", "Tuanama", "Shupingahua",
    # Hispanos frecuentes en Peru
    "Flores", "Vargas", "Rojas", "Castillo", "Chávez", "Sánchez", "Ramírez", "Torres",
    "Gutiérrez", "Paredes", "Cárdenas", "Espinoza", "Ríos", "Salazar", "Medina",
    "Palacios", "Rodríguez", "García", "López", "Díaz", "Mendoza", "Cruz", "Romero",
    "Ruiz", "Reyes", "Gonzales", "Alvarado", "Valverde", "Zevallos", "Benites",
    "Villanueva", "Aguilar", "Cáceres", "Delgado", "Herrera", "Vásquez", "Morales",
    "Silva", "Ramos", "Campos", "Pérez", "Tello", "Arias", "Bustamante", "Alarcón",
]
NOMBRES_PE_F = [
    "Milagros", "Rocío", "Yesenia", "Lizbeth", "Yovana", "Flor", "Maritza", "Deysi",
    "Karina", "Roxana", "Sonia", "Gladys", "Nelly", "Luz", "Marleny", "Yaneth",
    "Katherine", "Fiorella", "Kiara", "Brenda", "Melissa", "Nataly", "Zulema", "Rosa",
]
NOMBRES_PE_M = [
    "Jhon", "Jhonatan", "Wilmer", "Edwin", "Percy", "Elmer", "Nilton", "Walter",
    "Wilfredo", "Marco", "Richard", "Julio", "César", "Hugo", "Fredy", "Rolando",
    "Renzo", "Brayan", "Kevin", "Alcides", "Teodoro", "Néstor", "Gustavo", "Raúl",
]
LUGARES_PE = [
    # Departamentos / regiones
    "Lima", "Arequipa", "Cusco", "La Libertad", "Piura", "Loreto", "Junín", "Puno",
    "Ayacucho", "Cajamarca", "Callao", "Tacna", "Ica", "Lambayeque", "Áncash",
    "Huánuco", "San Martín", "Ucayali", "Apurímac", "Huancavelica", "Pasco",
    "Madre de Dios", "Amazonas", "Moquegua", "Tumbes",
    # Ciudades / provincias
    "Trujillo", "Iquitos", "Huancayo", "Chiclayo", "Juliaca", "Tarapoto", "Pucallpa",
    "Chimbote", "Huaraz", "Sullana", "Abancay", "Cerro de Pasco", "Puerto Maldonado",
    "Chachapoyas", "Moyobamba", "Jaén", "Talara", "Paita", "Huacho", "Barranca",
    "Cañete", "Chincha Alta", "Pisco", "Nazca", "Ilo", "Quillabamba", "Sicuani",
    "Andahuaylas", "Tingo María", "La Oroya", "Tarma", "Yurimaguas", "Ferreñafe",
    "Sechura", "Huamanga", "Urubamba", "Ollantaytambo", "Chulucanas", "Satipo",
    # Distritos de Lima Metropolitana / Callao
    "Miraflores", "San Isidro", "Santiago de Surco", "San Juan de Lurigancho",
    "San Martín de Porres", "Villa El Salvador", "Villa María del Triunfo", "Comas",
    "Ate", "La Molina", "Chorrillos", "Breña", "Jesús María", "Lince", "Pueblo Libre",
    "Magdalena del Mar", "San Borja", "Los Olivos", "Independencia", "Carabayllo",
    "Puente Piedra", "El Agustino", "Rímac", "Barranco", "Surquillo", "La Victoria",
    "Bellavista", "Ventanilla", "San Juan de Miraflores", "Santa Anita", "Lurín",
    "Pachacámac", "Chaclacayo", "San Miguel", "La Perla", "Cercado de Lima",
]
# Codigos de departamento para codigos postales peruanos (5 digitos: DD + 3)
CODIGOS_POSTALES_PE = list(range(1, 26))
PAISES = (
    ["Perú"] * 12
    + ["Venezuela", "Colombia", "Bolivia", "Ecuador", "Chile", "Argentina", "Brasil",
       "España", "Estados Unidos", "China", "Japón", "Italia", "México", "Haití", "Cuba"]
)
HOSPITALES = [
    # MINSA - Lima
    "Hospital Nacional Dos de Mayo", "Hospital Nacional Arzobispo Loayza",
    "Hospital Nacional Cayetano Heredia", "Hospital Nacional Hipólito Unanue",
    "Hospital Nacional Daniel Alcides Carrión", "Hospital Nacional Sergio E. Bernales",
    "Hospital María Auxiliadora", "Hospital de Emergencias José Casimiro Ulloa",
    "Hospital Santa Rosa", "Hospital Nacional Docente Madre Niño San Bartolomé",
    "Hospital Víctor Larco Herrera", "Hospital Hermilio Valdizán", "Hospital de Vitarte",
    "Hospital Carlos Lanfranco La Hoz", "Hospital de Huaycán",
    "Hospital de Emergencias Villa El Salvador", "Hospital San José del Callao",
    "Hospital de Emergencias Pediátricas",
    # Institutos especializados
    "Instituto Nacional de Salud del Niño", "Instituto Nacional de Salud del Niño San Borja",
    "Instituto Nacional Materno Perinatal", "Instituto Nacional de Enfermedades Neoplásicas",
    "Instituto Nacional de Ciencias Neurológicas", "Instituto Nacional Cardiovascular",
    "Instituto Nacional de Oftalmología", "Instituto Nacional de Rehabilitación",
    "Instituto Nacional de Salud Mental Honorio Delgado - Hideyo Noguchi",
    # EsSalud
    "Hospital Nacional Edgardo Rebagliati Martins", "Hospital Nacional Guillermo Almenara Irigoyen",
    "Hospital Nacional Alberto Sabogal Sologuren", "Hospital Nacional Carlos Alberto Seguín Escobedo",
    "Hospital Nacional Almanzor Aguinaga Asenjo", "Hospital Nacional Ramiro Prialé Prialé",
    "Hospital Víctor Lazarte Echegaray", "Hospital Nacional Adolfo Guevara Velasco",
    "Hospital III José Cayetano Heredia", "Hospital Alberto Leopoldo Barton Thompson",
    "Hospital Uldarico Rocca Fernández", "Hospital Grau",
    # FF. AA. y PNP
    "Hospital Militar Central", "Centro Médico Naval Cirujano Mayor Santiago Távara",
    "Hospital Central de la Fuerza Aérea del Perú", "Hospital Nacional PNP Luis N. Sáenz",
    # Regionales
    "Hospital Regional Honorio Delgado", "Hospital Antonio Lorena", "Hospital Regional del Cusco",
    "Hospital Belén de Trujillo", "Hospital Regional Docente de Trujillo",
    "Hospital Regional Lambayeque", "Hospital Regional Docente de Cajamarca",
    "Hospital Santa Rosa de Piura", "Hospital Regional de Loreto",
    "Hospital Regional Docente Clínico Quirúrgico Daniel Alcides Carrión",
    "Hospital Regional Manuel Núñez Butrón", "Hospital Hipólito Unanue de Tacna",
    "Hospital Regional de Ica", "Hospital Regional de Ayacucho", "Hospital Regional de Huacho",
    "Hospital Regional Hermilio Valdizán Medrano", "Hospital Regional de Pucallpa",
    "Hospital Eleazar Guzmán Barrón", "Hospital Víctor Ramos Guardia",
    # Clinicas privadas
    "Clínica Ricardo Palma", "Clínica Anglo Americana", "Clínica Internacional",
    "Clínica San Felipe", "Clínica Delgado", "Clínica Javier Prado", "Clínica San Pablo",
    "Clínica Maison de Santé", "Clínica Stella Maris", "Clínica Good Hope",
    "Clínica Montefiori", "Clínica Vesalio", "Clínica Arequipa", "Clínica Sanna El Golf",
    "Clínica San Gabriel", "Clínica Limatambo", "Clínica Peruano Americana",
    "Clínica Pardo", "Clínica San Juan de Dios",
]
CENTROS_SALUD = [
    "Centro de Salud San Juan de Miraflores", "Centro Materno Infantil Tahuantinsuyo Bajo",
    "Centro de Salud Villa María del Triunfo", "Centro Materno Infantil Juan Pablo II",
    "Centro de Salud Santa Luzmila", "Centro de Salud Lince", "Centro de Salud Surquillo",
    "Centro de Salud Chorrillos", "Centro de Salud Comas", "Centro de Salud Collique",
    "Centro de Salud Ermitaño Alto", "Centro de Salud Manuel Bonilla",
    "Centro de Salud Miraflores", "Centro de Salud Mirones", "Centro de Salud Santa Anita",
    "Centro de Salud Zárate", "Centro de Salud Max Arias Schreiber", "Centro de Salud Magdalena",
    "Centro de Salud Breña", "Centro de Salud Pedro Sánchez Meza", "Centro de Salud Belenpampa",
    "Centro de Salud Siete Cuartones", "Centro de Salud La Tulpuna", "Centro de Salud Wichanzao",
    "Centro de Salud Bellavista", "Centro de Salud Ciudad Nueva", "Centro de Salud Los Olivos",
    "Centro de Salud Laura Rodríguez Dulanto", "Centro de Salud Señor de los Milagros",
    "Centro Materno Infantil Manuel Barreto", "Centro Materno Infantil César López Silva",
    "Puesto de Salud Nueva Esperanza", "Puesto de Salud Huaycán", "Puesto de Salud Pamplona Alta",
    "Posta Médica Ate", "Policlínico Juan José Rodríguez Lazo", "Policlínico Pablo Bermúdez",
    "Policlínico Chincha", "Policlínico Francisco Pizarro", "Centro de Atención Primaria III Huaycán",
    "CAP III Metropolitano", "CAP II Lurín", "Centro de Salud Mental Comunitario Zárate",
]
INSTITUCIONES = [
    "Ministerio de Salud", "Ministerio de Salud del Perú", "MINSA", "Seguro Social de Salud",
    "EsSalud", "Seguro Integral de Salud", "SIS", "Instituto Nacional de Salud",
    "Dirección Regional de Salud de Cusco", "Gerencia Regional de Salud de Arequipa",
    "DIRESA Junín", "DIRIS Lima Centro", "DIRIS Lima Norte", "DIRIS Lima Sur",
    "Superintendencia Nacional de Salud", "SUSALUD", "Colegio Médico del Perú",
    "Colegio de Enfermeros del Perú", "Colegio Químico Farmacéutico del Perú",
    "Universidad Nacional Mayor de San Marcos", "Universidad Peruana Cayetano Heredia",
    "Pontificia Universidad Católica del Perú", "Universidad de San Martín de Porres",
    "Universidad Nacional de San Agustín", "Universidad Nacional de Trujillo",
    "Universidad Nacional de San Antonio Abad del Cusco", "Universidad Científica del Sur",
    "Universidad Ricardo Palma", "Universidad Peruana de Ciencias Aplicadas",
    "Laboratorio Clínico Roe", "Suiza Lab", "Laboratorio Referencial de Salud Pública",
    "Centro Nacional de Abastecimiento de Recursos Estratégicos en Salud",
    "Instituto Nacional de Estadística e Informática", "Fondo Intangible Solidario de Salud",
    "Ministerio de Trabajo y Promoción del Empleo", "Policía Nacional del Perú",
    "Sanidad de la Policía Nacional del Perú", "Cruz Roja Peruana",
    "Liga Peruana de Lucha contra el Cáncer", "Organización Panamericana de la Salud",
    "Municipalidad Metropolitana de Lima", "Municipalidad Provincial de Arequipa",
    "Instituto Peruano de Energía Nuclear", "Dirección General de Medicamentos, Insumos y Drogas",
]
TIPOS_VIA = ["Av.", "Av.", "Av.", "Jr.", "Jr.", "Calle", "Calle", "Psje.", "Prolongación", "Malecón"]
NOMBRES_VIA = [
    "Arequipa", "Javier Prado Este", "Abancay", "Tacna", "Garcilaso de la Vega", "Brasil",
    "Salaverry", "La Marina", "Universitaria", "Túpac Amaru", "Angamos Oeste", "Benavides",
    "Larco", "Pardo", "José Gálvez Barrenechea", "Petit Thouars", "Aviación", "Grau",
    "Bolognesi", "Ica", "Huallaga", "Ucayali", "Lampa", "Camaná", "Cusco", "Puno",
    "Los Próceres", "Las Flores", "Los Pinos", "Las Begonias", "Los Álamos", "Los Cedros",
    "San Martín", "Sucre", "Ayacucho", "Callao", "Zarumilla", "Colonial", "Venezuela",
    "Nicolás de Piérola", "Alfonso Ugarte", "Canevaro", "Mariscal Castilla", "Ricardo Palma",
    "Manco Cápac", "Atahualpa", "Pachacútec", "Mariano Melgar", "Ejército", "Ramón Castilla",
    "28 de Julio", "Dos de Mayo", "Miguel Grau", "Francisco Bolognesi", "Las Palmeras",
    "Santa Rosa", "Los Incas", "El Sol", "Independencia", "Industrial",
]
PREFIJOS_URB = ["Urb.", "AA.HH.", "Asoc.", "Coop.", "Res."]
NOMBRES_URB = [
    "Santa Patricia", "Los Jardines", "San Felipe", "El Pinar", "Mariscal Cáceres",
    "Villa Los Reyes", "Pachacútec", "Santa Rosa", "Los Ficus", "Huaycán", "Las Lomas",
    "San Antonio", "Primavera", "Monterrico", "El Trébol", "Señor de los Milagros",
]
PROFESIONES_PE = [
    "comerciante", "agricultor", "agricultora", "ama de casa", "docente", "profesora",
    "mototaxista", "chofer", "albañil", "estudiante universitario", "estudiante",
    "enfermera", "técnico en enfermería", "técnica en enfermería", "obrero de construcción",
    "minero", "pescador artesanal", "vendedora ambulante", "contador", "contadora",
    "ingeniero civil", "abogado", "abogada", "cocinera", "costurera", "policía", "militar",
    "ganadero", "artesano", "artesana", "taxista", "secretaria", "vigilante",
    "trabajadora del hogar", "mecánico", "electricista", "carpintero", "panadero",
    "agente de seguridad", "cajera", "operario de fábrica", "obstetra", "odontólogo",
]
DOMINIOS_CORREO = [
    "gmail.com", "gmail.com", "gmail.com", "hotmail.com", "hotmail.com", "outlook.com",
    "yahoo.es", "yahoo.com", "minsa.gob.pe", "essalud.gob.pe", "unmsm.edu.pe",
    "upch.pe", "pucp.edu.pe", "usmp.pe", "unsa.edu.pe",
]
# (masculino, femenino, modificadores permitidos)
PARENTESCOS = [
    ("padre", "madre", ()),
    ("hijo", "hija", ("mayor", "menor")),
    ("hermano", "hermana", ("mayor", "menor")),
    ("abuelo", "abuela", ("materno", "paterno")),
    ("tío", "tía", ("materno", "paterno")),
    ("primo", "prima", ("materno", "paterno")),
    ("sobrino", "sobrina", ()),
    ("nieto", "nieta", ()),
    ("esposo", "esposa", ()),
    ("cuñado", "cuñada", ()),
    ("suegro", "suegra", ()),
    ("yerno", "nuera", ()),
    ("padrastro", "madrastra", ()),
]
PARENTESCOS_PLURAL = ["padres", "hermanos", "hijos", "abuelos", "tíos", "primos", "familiares"]
PARENTESCOS_NEUTROS = ["pareja", "cónyuge", "familiar"]
PARTICULAS = {"de", "del", "la", "las", "los", "y", "-"}


def _feminizar_modificador(mod: str) -> str:
    return mod[:-1] + "a" if mod.endswith("o") else mod


# ---------------------------------------------------------------------------
# Lexicon de genero para nombres de pila (Faker es_* + nombres peruanos)
# ---------------------------------------------------------------------------
def _construir_lexicon() -> tuple[set[str], set[str], set[str]]:
    fem: set[str] = set()
    masc: set[str] = set()
    for loc in ("es_ES", "es_MX", "es_CO", "es_AR", "es_CL"):
        try:
            prov = importlib.import_module(f"faker.providers.person.{loc}").Provider
        except (ImportError, AttributeError):
            continue
        for nombre in getattr(prov, "first_names_female", ()):
            fem.add(_norm(nombre.split()[0]))
        for nombre in getattr(prov, "first_names_male", ()):
            masc.add(_norm(nombre.split()[0]))
    fem |= {_norm(n) for n in NOMBRES_PE_F}
    masc |= {_norm(n) for n in NOMBRES_PE_M}
    ambiguos = fem & masc
    # Excepciones: Faker es_ES lista "José" como femenino (por "María José")
    # y "María" como masculino (por "José María"). En posicion inicial el
    # genero lo marca el primer nombre, asi que se fuerzan explicitamente.
    forzados = {"jose": "M", "jesus": "M", "maria": "F", "rosario": "F", "guadalupe": "F"}
    for nombre, g in forzados.items():
        ambiguos.discard(nombre)
        (fem if g == "F" else masc).add(nombre)
        (masc if g == "F" else fem).discard(nombre)
    return fem - ambiguos, masc - ambiguos, ambiguos


LEX_FEM, LEX_MASC, LEX_AMBIGUOS = _construir_lexicon()
LEX_NOMBRES = LEX_FEM | LEX_MASC | LEX_AMBIGUOS
APELLIDOS_FAKER_EXCLUIDOS = {"Garica"}  # errata conocida en Faker es_MX

# Mapeo de menciones SEXO_SUJETO_ASISTENCIA (normalizadas con _norm, que quita
# acentos y puntos). En MEDDOCAN las abreviaturas son H. = hombre y M. = mujer;
# "m" NO significa masculino en este corpus.
SEXO_FEM = {
    "m", "f", "mujer", "femenino", "femenina", "hembra", "nina", "senora", "anciana",
    "nacida", "gestante", "puerpera", "embarazada", "esposa",
}
SEXO_MASC = {
    "h", "v", "varon", "hombre", "masculino", "nino", "senor", "anciano", "nacido", "esposo",
}


def _es_inicial(tok: str) -> bool:
    return bool(re.fullmatch(r"[A-ZÁÉÍÓÚÑ]\.", tok))


def genero_de_nombre(tokens: list[str]) -> str | None:
    """Infere 'F'/'M' a partir del primer nombre de pila reconocible."""
    for tok in tokens:
        n = _norm(tok)
        if n in LEX_FEM:
            return "F"
        if n in LEX_MASC:
            return "M"
        if n in LEX_AMBIGUOS:
            return None
    return None


def genero_de_documento(tokens: list[str], tags: list[str]) -> str | None:
    """Infere el sexo del paciente a partir de las menciones SEXO_SUJETO_ASISTENCIA."""
    votos = Counter()
    for start, end, label in entity_spans(tags):
        if label != "SEXO_SUJETO_ASISTENCIA":
            continue
        for tok in tokens[start:end]:
            n = _norm(tok)
            if n in SEXO_FEM:
                votos["F"] += 1
            elif n in SEXO_MASC:
                votos["M"] += 1
    if not votos:
        return None
    return votos.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Generadores (longitud natural, sin recorte ni relleno)
# ---------------------------------------------------------------------------
class Generador:
    def __init__(self, rng: random.Random, fake: Faker):
        self.rng = rng
        self.fake = fake

    # -- Personas ----------------------------------------------------------
    def nombre_pila(self, genero: str | None) -> str:
        g = genero or self.rng.choice("FM")
        if self.rng.random() < 0.3:
            return self.rng.choice(NOMBRES_PE_F if g == "F" else NOMBRES_PE_M)
        nombre = self.fake.first_name_female() if g == "F" else self.fake.first_name_male()
        return nombre.split()[0]

    def apellido(self) -> list[str]:
        # Se descartan apellidos que tambien son nombres de pila ("Luna",
        # "Santiago", "Tito", "Alba", "Rosario"...), para que un span de solo
        # apellidos no se lea como un nombre de genero opuesto al paciente.
        while True:
            if self.rng.random() < 0.6:
                ap = self.rng.choice(APELLIDOS_PE)
            else:
                ap = self.fake.last_name()
                if ap in APELLIDOS_FAKER_EXCLUIDOS:
                    continue
            partes = ap.split()
            if not any(_norm(p) in LEX_NOMBRES for p in partes if _norm(p) not in PARTICULAS):
                return partes

    def nombre_persona(self, original: list[str], genero_doc: str | None) -> list[str]:
        """Replica la estructura (nombres/iniciales + apellidos) y el genero.

        Prioridad de genero: `genero_doc` (sexo real del paciente segun la
        mencion SEXO del documento) > genero del nombre original. Los nombres
        sustitutos de MEDDOCAN no siempre concuerdan con el sexo del paciente,
        mientras que el resto del texto (pronombres, "la paciente") si.
        Para el personal sanitario se pasa genero_doc=None.
        """
        genero = genero_doc or genero_de_nombre(original)

        # Cuantos tokens iniciales son nombres de pila o iniciales (max 2)
        n_nombres = 0
        for tok in original:
            if n_nombres < 2 and (_norm(tok) in LEX_NOMBRES or _es_inicial(tok)):
                n_nombres += 1
            else:
                break
        resto = [t for t in original[n_nombres:] if _norm(t) not in PARTICULAS]
        n_apellidos = min(len(resto), 2)
        if n_nombres == 0 and n_apellidos == 0:
            n_nombres, n_apellidos = 1, 2

        tokens: list[str] = []
        for tok in original[:n_nombres]:
            if _es_inicial(tok):
                tokens.append(self.rng.choice("ABCDEFGHIJLMNOPRSTVY") + ".")
            else:
                tokens.append(self.nombre_pila(genero))
        for _ in range(n_apellidos):
            tokens.extend(self.apellido())
        return tokens

    def profesion(self) -> list[str]:
        if self.rng.random() < 0.6:
            return tokenizar(self.rng.choice(PROFESIONES_PE))
        for _ in range(10):
            job = self.fake.job()
            if len(job.split()) <= 4:
                return tokenizar(job[0].lower() + job[1:])
        return tokenizar(self.rng.choice(PROFESIONES_PE))

    def parentesco(self, original: list[str]) -> list[str]:
        primero = _norm(original[0]) if original else ""
        plurales = {_norm(p) for p in PARENTESCOS_PLURAL}
        if primero in plurales:
            return [self.rng.choice(PARENTESCOS_PLURAL)]
        masc = {_norm(m) for m, _, _ in PARENTESCOS}
        fem = {_norm(f) for _, f, _ in PARENTESCOS}
        if primero in fem:
            genero = "F"
        elif primero in masc:
            genero = "M"
        else:
            if self.rng.random() < 0.15:
                return [self.rng.choice(PARENTESCOS_NEUTROS)]
            genero = self.rng.choice("FM")
        m, f, mods = self.rng.choice(PARENTESCOS)
        tokens = [f if genero == "F" else m]
        if mods and self.rng.random() < 0.4:
            mod = self.rng.choice(mods)
            tokens.append(_feminizar_modificador(mod) if genero == "F" else mod)
        return tokens

    def correo(self) -> list[str]:
        usuario = _norm(self.fake.user_name()).replace(" ", "")
        if self.rng.random() < 0.3:
            usuario += str(self.rng.randint(1, 99))
        return [f"{usuario}@{self.rng.choice(DOMINIOS_CORREO)}"]

    # -- Lugares -------------------------------------------------------------
    def pais(self) -> list[str]:
        return tokenizar(self.rng.choice(PAISES))

    def territorio(self, original: list[str]) -> list[str]:
        if len(original) == 1 and re.fullmatch(r"\d{5}", original[0]):
            dep = self.rng.choice(CODIGOS_POSTALES_PE)
            return [f"{dep:02d}{self.rng.randint(1, 999):03d}"]
        return tokenizar(self.rng.choice(LUGARES_PE))

    def calle(self) -> list[str]:
        r = self.rng.random()
        via = f"{self.rng.choice(TIPOS_VIA)} {self.rng.choice(NOMBRES_VIA)}"
        numero = str(self.rng.randint(100, 4999)) if self.rng.random() > 0.05 else "s/n"
        if r < 0.40:
            texto = f"{via} {numero}"
        elif r < 0.60:
            texto = f"{via} {numero}, Dpto. {self.rng.randint(1, 15)}{self.rng.randint(1, 4):02d}"
        elif r < 0.70:
            texto = f"{via} {numero} Int. {self.rng.randint(1, 30)}"
        elif r < 0.80:
            texto = f"{via} cdra. {self.rng.randint(1, 40)}"
        else:
            mz = self.rng.choice("ABCDEFGHIJKLMN")
            urb = f"{self.rng.choice(PREFIJOS_URB)} {self.rng.choice(NOMBRES_URB)}"
            texto = f"Mz. {mz} Lt. {self.rng.randint(1, 40)}, {urb}"
        return tokenizar(texto)

    def hospital(self) -> list[str]:
        return tokenizar(self.rng.choice(HOSPITALES))

    def centro_salud(self) -> list[str]:
        return tokenizar(self.rng.choice(CENTROS_SALUD))

    def institucion(self) -> list[str]:
        return tokenizar(self.rng.choice(INSTITUCIONES))

    # -- Numeros e identificadores (sin prefijos de texto) ---------------------
    def _digitos(self, n: int, primero_no_cero: bool = True) -> str:
        cabeza = str(self.rng.randint(1, 9)) if primero_no_cero else str(self.rng.randint(0, 9))
        return cabeza + "".join(str(self.rng.randint(0, 9)) for _ in range(n - 1))

    def celular(self) -> list[str]:
        d = "9" + self._digitos(8, primero_no_cero=False)
        formato = self.rng.random()
        if formato < 0.35:
            return [d]
        if formato < 0.70:
            return [d[:3], d[3:6], d[6:]]
        if formato < 0.85:
            return ["+51", d[:3], d[3:6], d[6:]]
        return [f"{d[:3]}-{d[3:6]}-{d[6:]}"]

    def fijo(self) -> list[str]:
        if self.rng.random() < 0.6:  # Lima: 01 + 7 digitos
            codigo, numero = "01", str(self.rng.randint(2, 7)) + self._digitos(6, False)
        else:  # Provincias: 0XX + 6 digitos
            codigo = self.rng.choice(["044", "054", "084", "074", "073", "064", "065", "051", "052", "056"])
            numero = str(self.rng.randint(2, 6)) + self._digitos(5, False)
        formato = self.rng.random()
        if formato < 0.4:
            return [codigo, numero]
        if formato < 0.7:
            return [codigo, f"{numero[:-4]}-{numero[-4:]}"]
        return [f"{codigo}-{numero}"]

    def telefono(self) -> list[str]:
        return self.celular() if self.rng.random() < 0.7 else self.fijo()

    def id_sujeto(self) -> list[str]:
        # DNI (8 digitos) o numero de historia clinica (6-7 digitos)
        if self.rng.random() < 0.5:
            return [self._digitos(8)]
        return [self._digitos(self.rng.choice([6, 7]))]

    def id_contacto(self) -> list[str]:
        # Numero de episodio / atencion: 7 digitos
        return [self._digitos(7)]

    def id_aseguramiento(self) -> list[str]:
        # DNI (afiliacion SIS/EsSalud) o codigo autogenerado tipo EsSalud
        if self.rng.random() < 0.5:
            return [self._digitos(8)]
        fecha = f"{self.rng.randint(40, 99):02d}{self.rng.randint(1, 12):02d}{self.rng.randint(1, 28):02d}"
        letras = "".join(self.rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5))
        return [f"{fecha}{self.rng.randint(1, 9)}{letras}{self.rng.randint(1, 20):03d}"]

    def id_titulacion(self) -> list[str]:
        # Numero de colegiatura (CMP/CEP): 5-6 digitos
        return [self._digitos(self.rng.choice([5, 5, 6]))]

    def id_empleo(self) -> list[str]:
        # Codigo de planilla / empleado: 6-8 digitos
        return [self._digitos(self.rng.choice([6, 7, 8]))]

    # -- Despacho por etiqueta -------------------------------------------------
    def reemplazo(self, label: str, original: list[str], genero_doc: str | None) -> list[str] | None:
        name = label.upper()
        if name in EXCLUIDAS:
            return None
        if name == "NOMBRE_SUJETO_ASISTENCIA":
            return self.nombre_persona(original, genero_doc)
        if name == "NOMBRE_PERSONAL_SANITARIO":
            return self.nombre_persona(original, None)
        despacho = {
            "PROFESION": self.profesion,
            "CORREO_ELECTRONICO": self.correo,
            "PAIS": self.pais,
            "CALLE": self.calle,
            "HOSPITAL": self.hospital,
            "CENTRO_SALUD": self.centro_salud,
            "INSTITUCION": self.institucion,
            "NUMERO_TELEFONO": self.telefono,
            "NUMERO_FAX": self.fijo,
            "ID_CONTACTO_ASISTENCIAL": self.id_contacto,
            "ID_SUJETO_ASISTENCIA": self.id_sujeto,
            "ID_ASEGURAMIENTO": self.id_aseguramiento,
            "ID_TITULACION_PERSONAL_SANITARIO": self.id_titulacion,
            "ID_EMPLEO_PERSONAL_SANITARIO": self.id_empleo,
        }
        if name == "TERRITORIO":
            return self.territorio(original)
        if name == "FAMILIARES_SUJETO_ASISTENCIA":
            return self.parentesco(original)
        if name == "ID_SUJETO_ASISTENCIA" and not any(c.isdigit() for c in "".join(original)):
            # En MEDDOCAN algunos spans ID_SUJETO son descriptores sin digitos
            # ("lactante", "casada", "raza caucasica"). Convertirlos en un DNI
            # cambiaria la semantica del span: se conservan sin aumentar.
            return None
        fn = despacho.get(name)
        return fn() if fn else None


# ---------------------------------------------------------------------------
# BIO: spans, limpieza, lectura y aumentacion
# ---------------------------------------------------------------------------
def clean_bio_tags(tags: list[str]) -> list[str]:
    """Repara etiquetas I- huerfanas heredadas de la tokenizacion."""
    clean = []
    prev = "O"
    for tag in tags:
        if tag.startswith("I-") and (prev == "O" or prev[2:] != tag[2:]):
            tag = "B-" + tag[2:]
        clean.append(tag)
        prev = tag
    return clean


def entity_spans(tags: list[str]) -> list[tuple[int, int, str]]:
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
            current = tag[2:]
            if start is None or current != label:
                if start is not None:
                    spans.append((start, i, label or ""))
                start, label = None, None
        else:
            raise ValueError(f"Etiqueta BIO invalida: {tag!r}")

    if start is not None:
        spans.append((start, len(tags), label or ""))
    return spans


def load_documents(path: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            item = json.loads(line)
            tokens = item.get("tokens")
            tags = item.get("tags", item.get("etiquetas_ner", item.get("labels")))
            if tokens is None or tags is None or len(tokens) != len(tags):
                raise ValueError(f"{path.name}:{i + 1}: tokens/tags ausentes o desalineados")
            docs.append({
                "id": str(item.get("id", item.get("id_documento", f"doc{i:05d}"))),
                "tokens": [str(t) for t in tokens],
                "tags": clean_bio_tags([str(t) for t in tags]),
            })
    return docs


def augment_document(
    tokens: list[str], tags: list[str], gen: Generador, rate: float, stats: dict[str, Counter]
) -> tuple[list[str], list[str]]:
    """Reemplaza spans con valores de longitud natural y recalcula B-/I-."""
    new_tokens: list[str] = []
    new_tags: list[str] = []
    cursor = 0
    genero_doc = genero_de_documento(tokens, tags)

    for start, end, label in entity_spans(tags):
        new_tokens.extend(tokens[cursor:start])
        new_tags.extend(tags[cursor:start])
        original = tokens[start:end]
        stats["seen"][label] += 1

        replacement = None
        if rate > 0 and gen.rng.random() < rate:
            replacement = gen.reemplazo(label, original, genero_doc)

        if replacement:
            # Recalculo dinamico del span: B- + I- * (len - 1)
            new_tokens.extend(replacement)
            new_tags.append(f"B-{label}")
            new_tags.extend([f"I-{label}"] * (len(replacement) - 1))
            stats["replaced"][label] += 1
            stats["tok_in"][label] += len(original)
            stats["tok_out"][label] += len(replacement)
        else:
            new_tokens.extend(original)
            new_tags.extend(tags[start:end])
            stats["kept"][label] += 1

        cursor = end

    new_tokens.extend(tokens[cursor:])
    new_tags.extend(tags[cursor:])

    # Invariantes: 1:1 token/tag, sin tokens vacios ni con espacios
    if len(new_tokens) != len(new_tags):
        raise AssertionError("Desalineacion token/tag tras la aumentacion")
    if any(not t or any(c.isspace() for c in t) for t in new_tokens):
        raise AssertionError("Token vacio o con espacios tras la aumentacion")
    return new_tokens, new_tags


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_splits(value: str) -> list[str]:
    splits = [s.strip().lower() for s in value.split(",") if s.strip()]
    if set(splits) - {"train", "dev", "test"} or not splits:
        raise argparse.ArgumentTypeError("Usa combinaciones de: train, dev, test.")
    return splits


def resolver_ruta(p: Path) -> Path:
    """Rutas relativas se resuelven contra la raiz del proyecto (BASE_DIR)."""
    return p if p.is_absolute() else (BASE_DIR / p)


def resolve_input(in_dir: Path, split: str) -> Path:
    for name in (f"{split}.jsonl", f"corpus_bio_{split}.jsonl"):
        path = in_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Entrada no encontrada para '{split}' en {in_dir}.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=sorted(ESCENARIOS), help="Preset de escenario de tesis.")
    ap.add_argument("--in-dir", type=Path, default=RUTA_ENTRADA)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--splits", default=None, help="Splits a aumentar (default: train).")
    ap.add_argument("--rate", type=float, default=None, help="Tasa de reemplazo por entidad (0-1).")
    ap.add_argument("--copies", type=int, default=1, help="Versiones aumentadas por split.")
    ap.add_argument("--seed", type=int, default=42, help="Semilla de reproducibilidad.")
    ap.add_argument("--limit", type=int, default=0, help="Documentos a procesar (0 = todos).")
    args = ap.parse_args()

    preset = ESCENARIOS.get(args.scenario, {})
    splits = parse_splits(args.splits or preset.get("splits", "train"))
    rate = args.rate if args.rate is not None else preset.get("rate", 1.0)
    if not 0.0 <= rate <= 1.0:
        ap.error("--rate debe estar entre 0 y 1")
    out_dir = resolver_ruta(args.out_dir) if args.out_dir else RUTA_DATA / preset.get("out_dir", "augmented")
    in_dir = resolver_ruta(args.in_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Escenario={args.scenario or 'manual'} | rate={rate} | splits aumentados={splits} | seed={args.seed}")
    print(f"Entrada: {in_dir}\nSalida:  {out_dir}\n")

    resumen: dict[str, Any] = {
        "scenario": args.scenario or "manual", "rate": rate, "seed": args.seed,
        "augmented_splits": splits, "splits": {},
    }

    for split in ("train", "dev", "test"):
        try:
            input_path = resolve_input(in_dir, split)
        except FileNotFoundError:
            print(f"[aviso] No se encontro '{split}' en {in_dir}")
            continue

        docs = load_documents(input_path)
        if args.limit > 0:
            docs = docs[:args.limit]

        aumentar = split in splits
        current_rate = rate if aumentar else 0.0
        n_copias = args.copies if aumentar else 1

        for copy_idx in range(n_copias):
            suffix = "" if n_copias == 1 else f"_aug{copy_idx + 1}"
            output_path = out_dir / f"{split}{suffix}.jsonl"
            rng = random.Random(f"{args.seed}:{split}:{copy_idx}")
            fake = Faker("es_MX")
            fake.seed_instance(f"{args.seed}:{split}:{copy_idx}")
            gen = Generador(rng, fake)

            stats: dict[str, Counter] = {k: Counter() for k in ("seen", "replaced", "kept", "tok_in", "tok_out")}
            tokens_in = tokens_out = 0
            rows = []
            for doc in docs:
                tokens, tags = augment_document(doc["tokens"], doc["tags"], gen, current_rate, stats)
                tokens_in += len(doc["tokens"])
                tokens_out += len(tokens)
                rows.append({"id": doc["id"], "tokens": tokens, "tags": tags})

            with output_path.open("w", encoding="utf-8", newline="\n") as fh:
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")

            vistas = sum(stats["seen"].values())
            reemplazadas = sum(stats["replaced"].values())
            print(f"[ok] {output_path.name:18s} docs={len(rows):4d} | tokens {tokens_in} -> {tokens_out} "
                  f"({tokens_out - tokens_in:+d}) | entidades={vistas} reemplazadas={reemplazadas} "
                  f"({(reemplazadas / vistas * 100) if vistas else 0:.1f}%)")
            if not aumentar:
                print("     (copia original intacta para evaluacion)")

            resumen["splits"][f"{split}{suffix}"] = {
                "docs": len(rows), "tokens_in": tokens_in, "tokens_out": tokens_out,
                "entities": vistas, "replaced": reemplazadas,
                "per_label": {
                    lab: {
                        "seen": stats["seen"][lab], "replaced": stats["replaced"][lab],
                        "tokens_in": stats["tok_in"][lab], "tokens_out": stats["tok_out"][lab],
                    }
                    for lab in sorted(stats["seen"])
                },
            }

    (out_dir / "02_summary.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[ok] Resumen: {out_dir / '02_summary.json'}")


if __name__ == "__main__":
    main()

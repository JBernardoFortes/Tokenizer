"""Configuracao central de caminhos.

Todos os caminhos sao relativos ao diretorio raiz do projeto e podem ser
alterados via variaveis em .env (veja .env.example):

  DATASET_DIR: diretorio com os txts dos documentos (fonte).
  OUTPUT_DIR : diretorio de saida (corpus.txt e demais artefatos).
"""

from __future__ import annotations

import os
import pathlib

from dotenv import load_dotenv

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def _resolve(name: str, default: str) -> pathlib.Path:
    value = os.getenv(name) or default
    path = pathlib.Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


DATASET_DIR = _resolve("DATASET_DIR", "data/documents")
OUTPUT_DIR = _resolve("OUTPUT_DIR", "data")

CORPUS_PATH = OUTPUT_DIR / "corpus.txt"


def describe() -> str:
    return (
        f"PROJECT_ROOT = {PROJECT_ROOT}\n"
        f"DATASET_DIR  = {DATASET_DIR}\n"
        f"OUTPUT_DIR   = {OUTPUT_DIR}\n"
        f"CORPUS_PATH  = {CORPUS_PATH}"
    )
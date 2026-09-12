"""Step 1 - Carregar amostra do dataset BDTD e gerar o corpus em txt.

Mostra informacoes basicas: numero de documentos, numero medio de
paragrafos por documento e numero medio de palavras por documento.
Gera data/corpus.txt com todos os documentos separados pelo delimitador
<|endoftext|> (o token EOS do GPT-2).
"""

from __future__ import annotations

import pathlib
import statistics

import config

DELIMITER = "<|endoftext|>"


def load_documents(text_dir: pathlib.Path) -> list[str]:
    documents = []
    for txt_path in sorted(text_dir.glob("*.txt")):
        text = txt_path.read_text(encoding="utf-8").strip()
        if text:
            documents.append(text)
    return documents


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n")]
    return [p for p in paragraphs if p]


def count_words(text: str) -> int:
    return len(text.split())


def build_corpus(documents: list[str], delimiter: str) -> str:
    return f"\n{delimiter}\n".join(documents) + f"\n{delimiter}\n"


def main() -> None:
    text_dir = config.DATASET_DIR
    documents = load_documents(text_dir)
    n_docs = len(documents)
    paragraphs_per_doc = [len(split_paragraphs(d)) for d in documents]
    words_per_doc = [count_words(d) for d in documents]

    print("=" * 60)
    print("STEP 1 - ANALISE DO DATASET BDTD (amostra)")
    print("=" * 60)
    print(config.describe())
    print(f"Numero de documentos: {n_docs}")
    print(
        f"Numero medio de paragrafos por documento: "
        f"{statistics.mean(paragraphs_per_doc):.1f}"
    )
    print(
        f"Numero medio de palavras por documento: "
        f"{statistics.mean(words_per_doc):.1f}"
    )
    print(f"Numero total de paragrafos: {sum(paragraphs_per_doc)}")
    print(f"Numero total de palavras: {sum(words_per_doc)}")

    print()
    print("Exemplo (primeiro documento):")
    print("-" * 60)
    first = documents[0]
    first_paras = split_paragraphs(first)
    print(f"  Titulo/arquivo: {first[:120]!r}")
    print(f"  Paragrafos: {len(first_paras)} | Palavras: {words_per_doc[0]}")

    OUTPUT_PATH = config.CORPUS_PATH
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    corpus = build_corpus(documents, DELIMITER)
    OUTPUT_PATH.write_text(corpus, encoding="utf-8")
    size_mb = OUTPUT_PATH.stat().st_size / 1_048_576
    n_sep = corpus.count(DELIMITER)
    print()
    print(f"Arquivo txt gerado: {OUTPUT_PATH} ({size_mb:.2f} MB)")
    print(f"Separadores {DELIMITER!r} usados: {n_sep}")
    print(f"Ultimos 50 caracteres: {corpus[-50:]!r}")


if __name__ == "__main__":
    main()
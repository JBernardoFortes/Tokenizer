"""Step 3 - Gerar pares input-target (janelas deslizantes) com PyTorch.

Tokeniza o arquivo corpus.txt, constroi BdtdDataset e BdtdDataLoader
customizados e testa os parametros max_length, stride e batch_size.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader as TorchDataLoader

import tiktoken

import config
from dataset import BdtdDataLoader, BdtdDataset, describe_batch

CORPUS_PATH = config.CORPUS_PATH
ENCODING_NAME = "gpt2"


def tokenize_sample(
    enc: tiktoken.Encoding, sample_tokens: int | None
) -> list[int]:
    text = CORPUS_PATH.read_text(encoding="utf-8")
    allowed = {enc.decode([enc.eot_token])}
    tokens = enc.encode(text, allowed_special=allowed)
    if sample_tokens is not None:
        tokens = tokens[:sample_tokens]
    return tokens


def print_windows(
    enc: tiktoken.Encoding,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    max_rows: int = 4,
) -> None:
    print("Exemplos de janelas (input -> target):")
    for i in range(min(max_rows, inputs.shape[0])):
        inp = inputs[i].tolist()
        tgt = targets[i].tolist()
        print(f"  ids        {inp} -> {tgt}")
        print(f"  texto      {enc.decode(inp)!r} -> {enc.decode(tgt)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Janelas deslizantes input-target (PyTorch)")
    parser.add_argument("--max-length", type=int, default=6)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sample-tokens", type=int, default=20_000)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 3 - PARES INPUT-TARGET (JANELAS DESLIZANTES)")
    print("=" * 60)
    print(f"corpus: {CORPUS_PATH} | tokens amostrados: {args.sample_tokens}")
    print(f"max_length={args.max_length}, stride={args.stride}, batch_size={args.batch_size}")

    enc = tiktoken.get_encoding(ENCODING_NAME)
    tokens = tokenize_sample(enc, args.sample_tokens)
    print(f"Total de tokens do corpus (amostra): {len(tokens)}")

    dataset = BdtdDataset(tokens, max_length=args.max_length, stride=args.stride)
    print(f"BdtdDataset: {len(dataset)} janelas")

    loader = BdtdDataLoader(dataset, batch_size=args.batch_size, shuffle=True, seed=42)
    print(f"BdtdDataLoader: {len(loader)} batches (epoch)")
    print()

    for epoch in range(args.epochs):
        print(f"--- epoch {epoch + 1} ---")
        for batch_i, (inputs, targets) in enumerate(loader):
            print(f"batch {batch_i}: {describe_batch(inputs, targets)}")
            if batch_i == 0:
                print_windows(enc, inputs, targets)
            if batch_i == 1:
                break

    print()
    print("Comparacao: DataLoader do PyTorch sobre o mesmo dataset")
    torch_loader = TorchDataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    inputs, targets = next(iter(torch_loader))
    print(f"torch DataLoader -> {describe_batch(inputs, targets)}")

    print()
    print("Teste de parametros (max_length=8, stride=4, batch_size=8):")
    d2 = BdtdDataset(tokens, max_length=8, stride=4)
    l2 = BdtdDataLoader(d2, batch_size=8, shuffle=True, seed=7)
    print(f"  janelas={len(d2)}, batches={len(l2)}")
    inputs, targets = next(iter(l2))
    print(f"  {describe_batch(inputs, targets)}")
    print_windows(enc, inputs, targets)


if __name__ == "__main__":
    main()
"""Step 5 - Embeddings posicionais absolutos.

Valores sequenciais por janela/posicao: janela 1 -> 1.1, 1.2, ..., 1.6;
janela 2 -> 2.1, ..., 2.6; e assim por diante, considerando a janela de
contexto com max_length=6 tokens.

Soma os embeddings posicionais com os token embeddings para obter os
input embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import tiktoken

VOCAB_SIZE = 50_257
OUTPUT_DIM = 256
MAX_LENGTH = 6
ENCODING_NAME = "gpt2"
TEXT = "Raimundo Moura"


def absolute_positional_embeddings(
    window: int, max_length: int, output_dim: int, seq_len: int | None = None
) -> torch.Tensor:
    """Embeddings posicionais absolutos para uma janela.

    window: numero da janela (comeca em 1).
    max_length: largura da janela de contexto (em tokens).
    output_dim: dimensao do vetor de embedding.
    seq_len: comprimento da sequencia (usa max_length por padrao).
    """
    seq_len = seq_len or max_length
    positional = torch.zeros(seq_len, output_dim)
    for j in range(seq_len):
        value = window + (j + 1) / 10.0
        positional[j] = value
    return positional


def show_positional_values(window: int, max_length: int) -> None:
    values = [window + (j + 1) / 10.0 for j in range(max_length)]
    print(f"   janela {window}: {[f'{v:.1f}' for v in values]}")


def main() -> None:
    torch.manual_seed(0)

    print("=" * 60)
    print("STEP 5 - EMBEDDINGS POSICIONAIS (absolute positional)")
    print("=" * 60)
    print(f"max_length (janela de contexto) = {MAX_LENGTH}, output_dim = {OUTPUT_DIM}")
    print(f"Valores sequenciais por janela de {MAX_LENGTH} tokens:")
    show_positional_values(1, MAX_LENGTH)
    show_positional_values(2, MAX_LENGTH)
    show_positional_values(3, MAX_LENGTH)
    print("   ...")

    print()
    print(f"1) Vetores posicionais da janela 1 (shape 6 x {OUTPUT_DIM}):")
    pos_w1 = absolute_positional_embeddings(1, MAX_LENGTH, OUTPUT_DIM)
    print(f"   positional.shape = {tuple(pos_w1.shape)}")
    for j in range(MAX_LENGTH):
        print(f"   pos[{j}] = {pos_w1[j][:4].tolist()} ...  (valor {1 + (j + 1) / 10:.1f})")

    print()
    print(f"2) Vetores posicionais da janela 2:")
    pos_w2 = absolute_positional_embeddings(2, MAX_LENGTH, OUTPUT_DIM)
    for j in range(MAX_LENGTH):
        print(f"   pos[{j}] = {pos_w2[j][:4].tolist()} ...  (valor {2 + (j + 1) / 10:.1f})")

    print()
    print("3) Somar posicionais com token embeddings (input embeddings):")
    enc = tiktoken.get_encoding(ENCODING_NAME)
    token_ids = enc.encode(TEXT, allowed_special={enc.decode([enc.eot_token])})
    print(f"   tokens de {TEXT!r}: {token_ids}")
    tokens_emb = nn.Embedding(VOCAB_SIZE, OUTPUT_DIM)
    token_embeddings = tokens_emb(torch.tensor(token_ids, dtype=torch.long))
    positional = absolute_positional_embeddings(
        1, MAX_LENGTH, OUTPUT_DIM, seq_len=len(token_ids)
    )
    input_embeddings = token_embeddings + positional
    print(f"   token_embeddings.shape = {tuple(token_embeddings.shape)}")
    print(f"   positional.shape       = {tuple(positional.shape)}")
    print(f"   input_embeddings.shape = {tuple(input_embeddings.shape)}")
    print(
        f"   posicionais usados (janela 1, seq_len={len(token_ids)}): "
        + ", ".join(f"{1 + (j + 1) / 10:.1f}" for j in range(len(token_ids)))
    )
    print()
    for i, tid in enumerate(token_ids):
        value = 1 + (i + 1) / 10.0
        print(
            f"   id={tid:6d} | pos {value:.1f} | input_emb[{i}][:4] = "
            f"{input_embeddings[i][:4].tolist()}"
        )

    print()
    print("   Validacao (2.dims): input = token_emb + positional scalar")
    test = input_embeddings[0][0].item()
    exp = token_embeddings[0][0].item() + 1.1
    print(f"   input[0][0]={test:.4f} == token[0][0]+1.1={exp:.4f}: "
          f"{abs(test-exp)<1e-6}")


if __name__ == "__main__":
    main()
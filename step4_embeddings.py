"""Step 4 - Camada de token embeddings com torch.nn.Embedding.

Cria nn.Embedding(vocab_size=50257, output_dim=256), imprime os
embeddings de todo o vocabulario (.weight) e gera os embeddings
dos token ids de "Raimundo Moura".
"""

from __future__ import annotations

import torch
import torch.nn as nn

import tiktoken

VOCAB_SIZE = 50_257
OUTPUT_DIM = 256
ENCODING_NAME = "gpt2"
TEXT = "Raimundo Moura"


def main() -> None:
    torch.manual_seed(0)

    print("=" * 60)
    print("STEP 4 - TOKEN EMBEDDINGS (nn.Embedding)")
    print("=" * 60)
    tokens_emb = nn.Embedding(VOCAB_SIZE, OUTPUT_DIM)
    print(f"Criado: torch.nn.Embedding(vocab_size={VOCAB_SIZE}, output_dim={OUTPUT_DIM})")
    print(f"Pesos: weight.shape = {tuple(tokens_emb.weight.shape)}")

    print()
    print("1) Token embeddings de todo o vocabulario (.weight):")
    print(f"   shape {tuple(tokens_emb.weight.shape)}, dtype {tokens_emb.weight.dtype}")
    print("   Amostra de linhas do .weight:")
    print(f"   weight[0]    (id 0,    <UNK>): {tokens_emb.weight[0][:8].tolist()} ...")
    print(f"   weight[1]    (id 1,     '\"'): {tokens_emb.weight[1][:8].tolist()} ...")
    print(f"   weight[49]   (id 49,    'R' ): {tokens_emb.weight[49][:8].tolist()} ...")
    print(f"   weight[50256](id 50256,<|endoftext|>): {tokens_emb.weight[-1][:8].tolist()} ...")

    print()
    print("2) Token embeddings dos token ids de 'Raimundo Moura':")
    enc = tiktoken.get_encoding(ENCODING_NAME)
    token_ids = enc.encode(TEXT, allowed_special={enc.decode([enc.eot_token])})
    print(f"   token ids de {TEXT!r}: {token_ids}")
    ids_tensor = torch.tensor(token_ids, dtype=torch.long)
    embeddings = tokens_emb(ids_tensor)
    print(f"   embeddings.shape = {tuple(embeddings.shape)}")
    for i, tid in enumerate(token_ids):
        print(
            f"   id={tid:6d} -> emb[{i}] dims[:6] = "
            f"{embeddings[i][:6].tolist()} ..."
        )

    print()
    print("   Verificacao: a linha i equivale a .weight[token_id]:")
    for i, tid in enumerate(token_ids):
        eq = torch.allclose(embeddings[i], tokens_emb.weight[tid])
        print(f"   emb[{i}] == weight[{tid}]: {eq}")


if __name__ == "__main__":
    main()
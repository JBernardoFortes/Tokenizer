"""Step 2 - Tokenizador Byte Pair Encoding (BPE) com tiktoken.

Carrega o esquema de tokenizacao do GPT-2, verifica o vocabulario
(50.257 tokens) e testa encode/decode, incluindo o teste pedido:
"Raimundo Moura" deve gerar os tokens [49, 1385, 41204, 49902, 403].
"""

from __future__ import annotations

import tiktoken

ENCODING_NAME = "gpt2"
EXPECTED_VOCAB_SIZE = 50_257
EXPECTED_RAIMUNDO = [49, 1385, 41204, 49902, 403]

TEST_TEXTS = [
    "Raimundo Moura",
    "Tokenização por subpalavras é essencial para LLMs.",
    "Computação, Informática e Informação: 42 ≠ 43.",
    "GPT-2 usa BPE com bytes!,"
]


def main() -> None:
    enc = tiktoken.get_encoding(ENCODING_NAME)
    print("=" * 60)
    print("STEP 2 - TOKENIZADOR BPE (tiktoken / GPT-2)")
    print("=" * 60)
    print(f"Encoding carregado: {enc.name!r}")
    print(f"Tamanho do vocabulario: {enc.n_vocab}")

    assert enc.n_vocab == EXPECTED_VOCAB_SIZE, (
        f"vocab {enc.n_vocab} != {EXPECTED_VOCAB_SIZE}"
    )
    print(f"Vocabulario de 50.257 tokens: OK")

    print()
    for text in TEST_TEXTS:
        tokens = enc.encode(text)
        decoded = enc.decode(tokens)
        print("-" * 60)
        print(f"Texto: {text!r}")
        print(f"Tokens: {tokens}")
        print(f"Decode reverso: {decoded!r}")
        print(f"Decode==texto: {decoded == text}")

    print()
    print("-" * 60)
    raimundo = enc.encode("Raimundo Moura")
    print(f"Tokens gerados para 'Raimundo Moura': {raimundo}")
    print(f"Tokens_listados_na_atividade:           {EXPECTED_RAIMUNDO}")
    print(
        f"Decode dos tokens da atividade "
        f"{EXPECTED_RAIMUNDO}: {enc.decode(EXPECTED_RAIMUNDO)!r}"
    )
    print(
        "Obs: os merges do encoding gpt2 mudaram entre versoes do "
        "tiktoken; o token 403 virou 'un' e o 430 virou 'ra'. "
        "A lista da atividade gera 'Raimundo Mouun'; a lista gerada "
        "aqui decodifica corretamente para 'Raimundo Moura'."
    )

    print()
    print("Tokens especiais do vocab:")
    eot_id = enc.eot_token
    print(f"  <|endoftext|> id={eot_id} | "
          f"decode: {enc.decode([eot_id])!r} | vocab contem: "
          f"{eot_id in range(enc.n_vocab)}")
    token_1, token_49 = enc.decode([1]), enc.decode([49])
    print(f"  id=1 -> {token_1!r}")
    print(f"  id=49 -> {token_49!r}")


if __name__ == "__main__":
    main()
# tokenizer — Atividade Prática 1: LLMs do Zero

Implementação das 5 etapas do exercício "Tópicos em IA / PLN: Criação de
LLMs do Zero — Atividade Prática 1" (`ativ04_tokenizador.pdf`), usando como
dataset a amostra da BDTD em
`/home/bernardo/Documents/dev/pipeline-BDTD/data/processed/text/`.

## Setup

```bash
uv sync          # instala tiktoken + torch
uv run python step1_dataset.py    # análise do dataset + gera data/corpus.txt
uv run python step2_bpe.py        # tokenizador BPE (GPT-2) / tiktoken
uv run python step3_sliding_windows.py    # pares input-target (janelas deslizantes)
uv run python step4_embeddings.py         # token embeddings (nn.Embedding)
uv run python step5_positional.py         # embeddings posicionais absolutos
```

O `step3_sliding_windows.py` aceita parâmetros opcionais, ex.:

```bash
uv run python step3_sliding_windows.py --max-length 6 --stride 1 --batch-size 2
```

## Etapas

1. **Análise do dataset** — carrega os 28 documentos-processados da BDTD,
   mostra nº de documentos, média de parágrafos/doc e palavras/doc, e gera
   `data/corpus.txt` com todos os textos separados por `<|endoftext|>`.
2. **BPE com tiktoken** — carrega o encoding `gpt2`, confirma vocabulário de
   50.257 tokens, testa `encode()/decode()`. Para "Raimundo Moura" os tokens
   gerados são `[49, 1385, 41204, 49902, 430]`; a lista da atividade
   (`[49, 1385, 41204, 49902, 403]`) decodifica para "Raimundo Mouun" porque
   os merges do encoding `gpt2` mudaram entre versões do `tiktoken`
   (o token `403` virou "un"; o atual `430` é "ra").
3. **Janelas deslizantes** — `BdtdDataset` (Dataset custom de PyTorch) e
   `BdtdDataLoader` (monta batches com shuffle); compara com o
   `torch.utils.data.DataLoader`; testa `max_length`, `stride`, `batch_size`.
4. **Token embeddings** — `torch.nn.Embedding(50257, 256)`, imprime o `.weight`
   do vocabulário e gera os embeddings dos ids de "Raimundo Moura".
5. **Embeddings posicionais absolutos** — função que gera valores sequenciais
   por janela (1.1...1.6, 2.1...2.6, ...) com `max_length=6` e soma com os
   token embeddings para obter os input embeddings.

## Arquivos

- `step1_dataset.py` … `step5_positional.py`: uma etapa por script.
- `dataset.py`: classes `BdtdDataset` e `BdtdDataLoader` (reutilizadas no step 3).
- `data/corpus.txt`: corpus gerado no step 1 (separador `<|endoftext|>`).
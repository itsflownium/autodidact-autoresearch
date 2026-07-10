---
license: cdla-sharing-1.0
language:
  - en
pretty_name: Autodidact TinyStories Prepared Dataset
task_categories:
  - text-generation
size_categories:
  - 1M<n<10M
tags:
  - tinystories
  - tokenized
  - bpe
  - autoresearch
---

# Autodidact TinyStories Prepared Dataset

This private dataset repository stores the complete, immutable `tinystories-v1`
artifact used by
[autodidact-autoresearch](https://github.com/itsflownium/autodidact-autoresearch).
It is model-ready data, not a replacement for the original TinyStories source.

Keep this repository private. The archive contains evaluator-only `promotion` and
`sealed_final` splits. Training and research processes should receive only the
extracted `public/` directory.

## Artifact

| Field | Value |
| --- | --- |
| Path | `data/autodidact-tinystories-v1.tar.zst` |
| Compression | Zstandard level 15, 128 MiB window |
| Compressed size | 454,752,409 bytes (433.7 MiB) |
| Extracted file bytes | 1,183,154,998 bytes (1.10 GiB) |
| Regular files | 120 |
| Archive SHA-256 | `49fa417804c3e905cf986392d2397ec58e55317925e31021c7cb128417e153ac` |
| Hub revision | `1123f36219fdeb261212a73df750be6278a697bb` |
| Pipeline config SHA-256 | `b07b68ee77c3b2501dc3e2f22622a64c79a6c7a40d4f6e58540a6b5f4e581a10` |
| Tokenizer SHA-256 | `e186a50c80fee106f9903eb958e3d035689fd44f28b8add7183281c0a012ca10` |

The compressed archive is 38.4% of the extracted size. It has one
`tinystories-v1/` root and contains:

```text
tinystories-v1/
├── public/
│   ├── tokenizer.json
│   ├── data_policy.json
│   ├── train/
│   ├── dev/
│   └── manifest.json
└── protected/
    ├── promotion/
    ├── sealed_final/
    └── manifest.json
```

## Contents

| Split | Stories | Tokens | Access |
| --- | ---: | ---: | --- |
| Train | 2,119,489 | 534,707,558 | Public training plane |
| Development | 10,998 | 2,691,298 | Public development |
| Promotion | 5,438 | 1,322,731 | Evaluator only |
| Sealed final | 5,554 | 1,369,152 | Evaluator only |

Token shards are document-preserving little-endian `uint16` arrays. Each shard
has a NumPy index containing token offsets, token counts, original UTF-8 byte
counts, and story content hashes. The vocabulary is a fixed 1,792-token
byte-level BPE trained on training stories only.

## Download

Authenticate before downloading this private dataset:

```bash
hf auth login
hf download Flownium/autodidact-dataset \
  data/autodidact-tinystories-v1.tar.zst \
  --repo-type dataset \
  --revision 1123f36219fdeb261212a73df750be6278a697bb \
  --local-dir .
```

The project preparation CLI downloads to temporary storage, verifies the archive
hash, extracts without allowing links or path traversal, verifies every dataset
manifest and shard, seals the tree read-only, and removes the compressed copy:

```bash
uv run prepare.py fetch-prepared
```

## Source and license

The archive was deterministically prepared from
[`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories)
revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`.

TinyStories and this prepared redistribution use
CDLA-Sharing-1.0. The project source code remains separately licensed under MIT.

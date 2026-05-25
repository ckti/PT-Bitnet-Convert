# PT-BitNet-style GGUF ternary converter

This directory contains a standalone CUDA/NVCC converter for local GGUF files. It reads a GGUF model, preserves metadata and tensor order, and converts eligible dense weight tensors to llama.cpp-compatible `TQ1_0` ternary tensor blocks.

## What is implemented

- Reads GGUF v2/v3-style headers, metadata, tensor infos, and tensor data.
- Converts dense `F32`, `F16`, and `BF16` tensors with at least 2 dimensions and `ne[0] % 256 == 0`.
- Leaves token embeddings and output tensors unquantized by default, matching the paper's note that embeddings are not ternarized.
- Packs converted tensors as llama.cpp `TQ1_0` type id `34` with 256-weight ternary blocks.
- Uses a PT-BitNet-inspired weight-only threshold search:
  - base threshold `Delta = 0.75 * mean(abs(W))`
  - grid search over `50%..150%` of that base by default
  - per-block scale `alpha` is the least-squares scale for active ternary weights
- Patches `general.file_type` to `MOSTLY_TQ1_0` when that metadata key exists.

## Build

```bash
make
```

This builds:

- `ptq_ternary_gguf`: CUDA GGUF-to-ternary converter
- `hf_to_ternary_gguf`: NVCC-built Hugging Face safetensors-to-ternary GGUF pipeline driver

## GGUF To Ternary

```bash
./ptq_ternary_gguf input-f16.gguf output-tq1_0.gguf
```

Useful options:

```bash
./ptq_ternary_gguf --dry-run input.gguf output.gguf
./ptq_ternary_gguf --chunk-mib 256 input.gguf output.gguf
./ptq_ternary_gguf --include-regex 'blk\..*(attn|ffn).*weight' input.gguf output.gguf
./ptq_ternary_gguf --exclude-regex 'token_embd|output' input.gguf output.gguf
./ptq_ternary_gguf --quantize-output --quantize-token-embd input.gguf output.gguf
```

## Hugging Face Safetensors To Ternary GGUF

Use this for a downloaded Hugging Face model directory containing `config.json`, tokenizer files, and `.safetensors` shards.

```bash
./hf_to_ternary_gguf \
  --hf-dir /path/to/hf-model-dir \
  --out /path/to/model-tq1_0.gguf
```

Pipeline:

1. Runs llama.cpp `convert_hf_to_gguf.py` to create an intermediate F16 GGUF.
2. Runs `ptq_ternary_gguf` to CUDA-quantize eligible dense tensors to `TQ1_0`.
3. Deletes the intermediate GGUF unless `--keep-intermediate` is set.

Useful options:

```bash
./hf_to_ternary_gguf --dry-run --hf-dir /path/to/hf-model-dir --out model-tq1_0.gguf
./hf_to_ternary_gguf --outtype bf16 --hf-dir /path/to/hf-model-dir --out model-tq1_0.gguf
./hf_to_ternary_gguf --chunk-mib 256 --hf-dir /path/to/hf-model-dir --out model-tq1_0.gguf
./hf_to_ternary_gguf --keep-intermediate --hf-dir /path/to/hf-model-dir --out model-tq1_0.gguf
```

The HF conversion step needs llama.cpp's Python conversion dependencies. If `torch` or related packages are missing:

```bash
python3 -m pip install -r <llama.cpp.directory>/requirements/requirements-convert_hf_to_gguf.txt
```

## Verify output

Use llama.cpp tooling from a build that supports `TQ1_0`:

```bash
llama-gguf-hash output-tq1_0.gguf
llama-perplexity -m output-tq1_0.gguf -f wiki.test.raw --chunks 32
```

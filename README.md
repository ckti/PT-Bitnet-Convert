# PT-BitNet-style GGUF ternary converter

This reposotory contains a standalone CUDA/NVCC converter for local GGUF files. It reads a GGUF model, preserves metadata and tensor order, and converts eligible dense weight tensors to llama.cpp-compatible `TQ1_0` ternary tensor blocks.  THis is based on the Mathematics and Algorithims found at 

PT-BitNet: Scaling up the 1-Bit large language model with post-training quantization,
Neural Networks,
Volume 191,2025,107855,ISSN 0893-6080,
https://doi.org/10.1016/j.neunet.2025.107855

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

`ptq_calibrate_reconstruct_gguf` links against the local llama.cpp build at `llama-cpp` by default. Override paths if building elsewhere:

```bash
make LLAMA_DIR=/path/to/llama.cpp LLAMA_BUILD=/path/to/llama.cpp/build
```

This builds:

- `ptq_ternary_gguf`: CUDA GGUF-to-ternary converter
- `hf_to_ternary_gguf`: NVCC-built Hugging Face safetensors-to-ternary GGUF pipeline driver
- `ptq_stage2_refine_gguf`: CUDA second-pass TQ1_0 weight refinement
- `ptq_calibrate_reconstruct_gguf`: llama.cpp calibration execution + activation-aware TQ1_0 reconstruction
- `ptq_qat_ternary_gguf`: experimental local ternary-aware QAT/distillation pass for existing `TQ1_0` tensors
- `ptq_qat_trainer_gguf`: prototype post-training fake-ternary trainer that starts from the dense GGUF and writes into a `TQ1_0` layout GGUF

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

## Second-Pass TQ1_0 Refinement

This pass takes the original dense GGUF plus a `TQ1_0` GGUF created by `ptq_ternary_gguf`, then rewrites the ternary tensor data in a new output file.

```bash
./ptq_stage2_refine_gguf \
  --fp-model /path/to/original-bf16-or-f16.gguf \
  --tq-model /path/to/model-tq1_0.gguf \
  --out /path/to/model-tq1_0-refined.gguf
```

Useful options:

```bash
./ptq_stage2_refine_gguf --dry-run --fp-model original.gguf --tq-model model-tq1_0.gguf --out refined.gguf
./ptq_stage2_refine_gguf --iterations 12 --chunk-mib 256 --fp-model original.gguf --tq-model model-tq1_0.gguf --out refined.gguf
```

This is a weight-space refinement against the original weights. It is not the full PT-BitNet activation/block-reconstruction optimization from the paper, which also needs calibration text, model execution, collected block activations, and an optimizer over block output error.

## Calibration And Activation-Aware Reconstruction

This pass links against the local consolidated llama.cpp build, runs calibration text through the original dense GGUF, captures `GGML_OP_MUL_MAT` input activations with `llama_context_params.cb_eval`, and rewrites `TQ1_0` tensors in a copy of the ternary GGUF.

```bash
./ptq_calibrate_reconstruct_gguf \
  --fp-model /path/to/original-bf16-or-f16.gguf \
  --tq-model /path/to/model-tq1_0.gguf \
  --calib /path/to/calibration.txt \
  --out /path/to/model-tq1_0-calibrated.gguf
```

Useful options:

```bash
./ptq_calibrate_reconstruct_gguf --dry-run --fp-model original.gguf --tq-model model-tq1_0.gguf --calib calib.txt --out calibrated.gguf
./ptq_calibrate_reconstruct_gguf --ctx 4096 --batch 512 --max-tokens 4096 --fp-model original.gguf --tq-model model-tq1_0.gguf --calib calib.txt --out calibrated.gguf
./ptq_calibrate_reconstruct_gguf --n-gpu-layers -1 --chunk-mib 256 --fp-model original.gguf --tq-model model-tq1_0.gguf --calib calib.txt --out calibrated.gguf
./ptq_calibrate_reconstruct_gguf --include-regex 'blk\.(0|1)\..*weight' --fp-model original.gguf --tq-model model-tq1_0.gguf --calib calib.txt --out calibrated.gguf
```

The reconstruction objective uses the captured activations as a diagonal block-output loss approximation:

- Captures matmul input activations for each target weight tensor.
- Accumulates per-input-channel activation energy, `mean(x_j^2)`.
- Re-optimizes each 256-weight `TQ1_0` block with activation-weighted reconstruction error.
- Falls back to uniform weights for a target tensor if llama.cpp does not expose a matching activation tensor name.

This is more faithful than the weight-only passes, but it is still not a full end-to-end PT-BitNet optimizer with dense 256x256 activation covariance or Adam updates over full layer output error. Use real calibration text from the target domain and validate with perplexity or task evals.

## Experimental Local QAT / Distillation

`ptq_qat_ternary_gguf` is the first training-style pass. It runs calibration text through the original dense GGUF, captures matmul input activation covariance, then applies CUDA STE-style updates to the latent ternary assignment for each 256-weight `TQ1_0` block before writing a new GGUF.

```bash
./ptq_qat_ternary_gguf \
  --fp-model /path/to/original-bf16-or-f16.gguf \
  --tq-model /path/to/model-tq1_0.gguf \
  --calib /path/to/calibration.txt \
  --out /path/to/model-tq1_0-qat.gguf
```

Useful options:

```bash
./ptq_qat_ternary_gguf --dry-run --fp-model original.gguf --tq-model model-tq1_0.gguf --calib calib.txt --out qat.gguf
./ptq_qat_ternary_gguf --include-regex 'blk\.0\.attn_q\.weight' --ctx 512 --max-tokens 512 --fp-model original.gguf --tq-model model-tq1_0.gguf --calib calib.txt --out qat-one-tensor.gguf
./ptq_qat_ternary_gguf --qat-steps 32 --learning-rate 0.01 --grad-clip 0.5 --fp-model original.gguf --tq-model model-tq1_0.gguf --calib calib.txt --out qat.gguf
./ptq_qat_ternary_gguf --capture-stride 4 --max-capture-mib 512 --fp-model original.gguf --tq-model model-tq1_0.gguf --calib calib.txt --out qat.gguf
```

Implementation details:

- Teacher is the original dense GGUF; the target file supplies the tensor list and `TQ1_0` storage layout.
- Captures `GGML_OP_MUL_MAT` input activations with llama.cpp graph callbacks.
- Builds one 256x256 covariance matrix per 256-channel input block.
- CUDA kernel runs one thread block per `TQ1_0` block and performs STE-style latent updates against local block-output MSE.
- Falls back to identity covariance for tensors with no captured samples.

This is still local layer-wise QAT, not full transformer backpropagation through attention, MLP nonlinearities, KV cache, and logits. If direct `TQ1_0` remains gibberish, use this to experiment with limited tensor subsets first, then validate with perplexity before trying whole-model runs.

## Prototype Post-Training Fake-Ternary Trainer

`ptq_qat_trainer_gguf` is the safer training entry point when the first-pass `TQ1_0` model has already collapsed. The existing `TQ1_0` GGUF is used only as the output tensor layout; the trainable latent weights are initialized from the original dense GGUF.

Create the layout file first:

```bash
./ptq_ternary_gguf original.gguf initial-tq1_0.gguf
```

Then run the trainer:

```bash
./ptq_qat_trainer_gguf \
  --fp-model original.gguf \
  --layout-tq-model initial-tq1_0.gguf \
  --calib calibration.txt \
  --out trained-tq1_0.gguf
```

Useful options:

```bash
./ptq_qat_trainer_gguf --dry-run --fp-model original.gguf --layout-tq-model initial-tq1_0.gguf --calib calib.txt --out trained.gguf
./ptq_qat_trainer_gguf --include-regex 'blk\.0\.attn_q\.weight' --ctx 512 --max-tokens 512 --fp-model original.gguf --layout-tq-model initial-tq1_0.gguf --calib calib.txt --out trained-one-tensor.gguf
./ptq_qat_trainer_gguf --qat-steps 64 --learning-rate 0.005 --grad-clip 0.25 --fp-model original.gguf --layout-tq-model initial-tq1_0.gguf --calib calib.txt --out trained.gguf
./ptq_qat_trainer_gguf --capture-stride 4 --max-capture-mib 512 --fp-model original.gguf --layout-tq-model initial-tq1_0.gguf --calib calib.txt --out trained.gguf
```

This is still a layer-local CUDA training approximation: it captures dense teacher activations, builds 256x256 activation covariance blocks, and optimizes local block-output MSE with STE-style ternary updates. It does not yet run a full fake-ternary student forward pass with sequence/logit loss and transformer backpropagation.

For calibration, do not use a tiny prompt file except for smoke tests. Use at least several thousand tokens of representative text from the target domain. A few short Q/A lines can prove that callbacks work, but they cannot fit a whole-model ternary layout.

If a fully ternarized model repeats common words or produces gibberish, treat that as model collapse rather than a sampling issue. Re-run with `--include-regex` on one tensor or one layer first, measure perplexity/generation, then expand the ternary coverage gradually. A coherent mixed BF16/`TQ1_0` model is a useful intermediate result; a fully `TQ1_0` model generally needs real end-to-end ternary-aware training, not just local post-hoc reconstruction.

## Verify output

Use llama.cpp tooling from a build that supports `TQ1_0`:

```bash
llama-gguf-hash output-tq1_0.gguf
llama-perplexity -m output-tq1_0.gguf -f wiki.test.raw --chunks 32
```

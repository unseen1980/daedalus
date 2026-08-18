# Daedalus-150M

A 150M-parameter language model built for **CPU inference**. It replaces two
thirds of a transformer's attention layers with short convolutions that keep a
fixed-size state, so decoding does not slow down as the context grows.

Trained from scratch on 59.9B tokens. Apache 2.0.

```bash
brew install llama.cpp   # or build from ggml-org/llama.cpp
hf download Unseen1980/daedalus-checkpoints instruct/model-q4_0.gguf --local-dir ~/daedalus

llama-cli -m ~/daedalus/instruct/model-q4_0.gguf -cnv \
  --temp 0.8 --top-p 0.9 --repeat-penalty 1.15
```

## Results

Scored on HellaSwag, ARC-Easy, PIQA, OpenBookQA and WinoGrande, with every peer
re-scored on the same harness rather than quoted from its paper.

| Model | Training tokens | 5-task mean |
|---|---|---|
| **Daedalus-150M** | **59.9B** | **47.31** |
| MobileLLM-125M | 1T | 46.3 *(published)* |
| GPT-2 124M | — | 42.2 |
| OPT-125M | 180B | 42.1 |
| GPT-neo-125M | 300B | 41.9 |
| Pythia-160M | 300B | 41.0 |
| SmolLM2-135M | 2T | 51.2 |

It beats every model in its size class trained on 3–17× more data. SmolLM2-135M,
trained on 2T tokens, remains ahead on quality — that was conceded in advance;
the trade this project makes is speed.

Validation bits-per-byte: **0.8685** over a 645M-token held-out set.

## The speed claim

CPU decode, 4-bit weights, 8 threads, measured against a parameter-matched
all-attention twin trained on identical data:

| Context depth | Daedalus | Dense twin | Ratio |
|---|---|---|---|
| 0 (empty) | 1112 tok/s | 923 tok/s | 1.20× |
| 512 | 960 tok/s | 664 tok/s | 1.45× |
| **2048** | **739 tok/s** | **420 tok/s** | **1.76×** |

**The shape of that table is the result, not the bottom row.** At an empty
context the hybrid has nothing to gain — its advantage *is* the key–value cache
it does not keep. The advantage grows with context, which is the signature the
mechanism predicts. Against an external 135M peer the same pattern reaches
**2.08× at 2048 tokens**.

A first-order memory-bandwidth model predicts only 1.17× at that depth, so
byte-counting alone does not explain the gap; latency-bound cache traversal and
layer count account for the rest. That analysis is in the paper.

## How it works

```
18 blocks, d_model 768, vocab 49,152, context 2048

block:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18
type:   C  C  C  C  A  C  C  A  C  A  C  A  C  A  C  C  A  C

A = full attention (6)     GQA, 12 query heads / 4 KV heads
C = short convolution (12) depthwise, kernel 3, fixed 2-step state
```

Each attention layer must re-read its cache for every token it generates, and
that cache grows with the conversation. A convolution layer's state is two
timesteps wide regardless of context, so it costs the same at token 2000 as at
token 2.

Per token of context, this model reads **6,144 bytes** of cache against a
24-layer all-attention model's **12,288** — exactly half. At 2048 tokens that is
12.6 MB re-read per generated token instead of 25.2 MB.

Also: tied embeddings (23% of parameters), a narrow 2048 FFN, and `Q4_0`
quantisation chosen for its ARM kernel speed rather than its error curve.

## Training

| | |
|---|---|
| Tokens | 59.9B over a 16.9B-token corpus (~3.5 epochs, capped at 4/source) |
| Optimisers | Muon (matrices) + AdamW (embeddings, norms) |
| Schedule | WSD — warmup, stable at peak, **linear decay to zero** over the final 45% |
| Hardware | 1× RTX 5090, ~$46 of GPU time |
| Post-training | SFT on smol-smoltalk + one DPO round on UltraFeedback |

The corpus is ten public English sources weighted toward educational text —
FineWeb-Edu 37.5%, DCLM-baseline 22.5%, Stack-Edu (Python) 9%, FinePDFs-Edu 8%,
FinePhrase 7%, Cosmopedia-v2 5%, FineMath and InfiWebMath 6%, FineWiki-en 3%,
everyday-conversations 2%.

**A note on the loss curve, since it looks alarming:** under WSD the learning
rate is held at its peak for the first 55% of training, and loss goes flat for
tens of thousands of steps. That is the schedule working. Nearly all remaining
quality arrives during the decay phase — validation bits-per-byte moved
0.9924 → 0.8685 once decay began.

## Was the hybrid actually worth it?

Two models, parameter-matched to within 0.5% (160.49M hybrid vs 161.25M dense),
trained on identical data and schedule for 5B tokens each. **The win condition
was written down before either was scored:** beat the twin by more than 0.5% on
validation bits-per-byte.

| | val_bpb ↓ | 5-task mean |
|---|---|---|
| **Hybrid** | **0.910398** | 44.68 |
| Dense twin | 0.917774 | **44.82** |

The hybrid won the pre-registered metric by 0.81%. On downstream tasks the twin
is nominally ahead by 0.14 points — about 0.24σ, well inside noise, with the two
trading wins task by task. At 5B tokens that suite is near its floor; WinoGrande
scores 50.0 against a 50.0 chance baseline.

**So: matched on quality, decisively faster.** Not a quality win, and not
presented as one.

## Models

All on [`Unseen1980/daedalus-checkpoints`](https://huggingface.co/Unseen1980/daedalus-checkpoints):

| File | What |
|---|---|
| `instruct/model-q4_0.gguf` | chat model, 4-bit — **start here** |
| `gguf/hero-base-q4_0.gguf` | base model, 4-bit, text completion |
| `gguf/instruct-f16.gguf` | instruct, f16 — for re-quantising |
| `gguf/hero-base-f16.gguf` | base, f16 |
| `hf/instruct/`, `hf/base/` | HF-format safetensors |
| `final/hero/checkpoint.pt` | base weights + optimizer state, for resuming |

The base model is text-completion only. It has no chat template on purpose —
giving one to a base model makes llama.cpp wrap prompts in markup it never saw.
Use `llama-completion`, or `llama-cli` with plain prompts.

## Reproducing

```bash
pip install -r requirements.txt

# 1. Build the corpus (public sources, ~17B tokens)
python -m daedalus.dataprep --out data/shards

# 2. Train
python train.py --config daedalus-150m --data-dir data/shards-hero-split/train \
  --val-dir data/shards-hero-split/holdout --total-tokens 59900000000 --muon-lr 0.02

# 3. Instruction-tune
python post.py --init-from runs/hero/checkpoint.pt --config daedalus-150m --dpo

# 4. Export to GGUF (--ppl-text-file is required; without it no GGUF is written)
python export.py --checkpoint runs/post-sft/final.pt --config daedalus-150m \
  --out-dir runs/export/post --ppl-text-file data/eval/ppl-finewiki-150k.txt
```

`python -m pytest` runs the suite.

## Layout

```
daedalus/       model, data pipeline, optimizers, quantisation
train.py        pretraining
post.py         SFT + DPO
export.py       HF + GGUF export, Q4_0 quantisation
eval.py         5-task suite + bits-per-byte
paper/          the write-up (IEEE format PDF)
runs/eval/      measured results behind every number above
docs/USAGE.md   running the model locally
```

## Known limitations

- **English only**, 2048-token context, single seed.
- **The 4-bit model costs ~6% perplexity**, not the ~2.5% intended.
  Quantisation-aware training was built and validated against llama.cpp's exact
  Q4_0 grid, then crashed on activation and never ran. The f16 files above make
  this fixable without retraining.
- **~48% of convolution channels are dead** — 13.6M inert parameters. They
  cannot be pruned at export: llama.cpp shape-checks those tensors against the
  model width, tested and confirmed. Fix belongs in the next model's
  initialisation.
- **The vocabulary is oversized.** 49,152 entries were inherited from a
  tokenizer chosen for a distillation plan that was later cancelled; scaling
  laws suggest 24–32k at this size. It costs 23% of parameters to a lookup table.
- **Mixture skew 10.42** against a 10.0 pre-registered limit, from training
  59.9B tokens on a 16.9B corpus.

## Paper

`paper/daedalus.pdf` — the architecture, why it is fast on a CPU, the
head-to-head experiment against a conventional model, the measured results, and
every limitation listed above.

## License

Apache 2.0. The corpus is assembled from public datasets under their own terms.

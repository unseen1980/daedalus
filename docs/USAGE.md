# Running Daedalus locally

How to get the model onto your machine and use it. Everything here is CPU-only;
no GPU is required.

---

## Setup

```bash
brew install llama.cpp        # macOS
# or build from https://github.com/ggml-org/llama.cpp
```

You also need the Hugging Face CLI to fetch the weights:

```bash
pip install huggingface_hub
```

---

## The chat model

This is the one most people want. It answers questions.

```bash
hf download Unseen1980/daedalus-checkpoints instruct/model-q4_0.gguf \
  --local-dir ~/daedalus

llama-cli -m ~/daedalus/instruct/model-q4_0.gguf -cnv \
  --temp 0.8 --top-p 0.9 --repeat-penalty 1.15
```

`-cnv` starts a chat session. Type at the `>` prompt.

Example answers:

```
> What is the capital of France?
The capital of France is Paris. Paris is a city known for its historical
landmarks and cultural institutions, such as the Eiffel Tower and the
Louvre Museum.

> Explain photosynthesis in one sentence.
Photosynthesis is a vital process that occurs in plants, algae, and some
bacteria, where they convert light energy into chemical energy.
```

### Sampling flags are not optional

llama.cpp sets `--repeat-penalty` to 1.0 by default, which means **off**. Without
it the model can lock onto a word and repeat it until it runs out of tokens.

| flag | why |
|---|---|
| `--repeat-penalty 1.15` | the one that stops loops |
| `--repeat-last-n 128` | how far back the penalty looks |
| `--temp 0.8` | 0 is greedy and loops; much higher is incoherent |
| `--top-p 0.9` | trims the nonsense tail |

Keep generations short. Quality drops as the output grows, so `-n 80` or less.

---

## The base model

The base model **continues text** rather than answering questions. Ask it a
question and it will often write more questions, because that is what follows a
question in web text.

```bash
hf download Unseen1980/daedalus-checkpoints gguf/hero-base-q4_0.gguf \
  --local-dir ~/daedalus

llama-completion -m ~/daedalus/gguf/hero-base-q4_0.gguf \
  -p "The capital of France is" -n 40 --temp 0.7 --top-p 0.9
```

Use `llama-completion`, not `llama-cli`. `llama-cli` is built around chat and
will wrap your prompt in conversation markup the base model has never seen.

Give it the start of a sentence, not an instruction:

```
"Photosynthesis is the process by which"
  → plants convert sunlight into chemical energy to drive their growth.
    The process is carried out by chlorophyll-a, a pigment found in the
    chloroplasts of plants.

"The Second World War began in"
  → 1939 and ended in 1945.
```

The base model has no chat template on purpose. Adding one makes llama.cpp treat
it as a chat model and inject markup it was never trained on.

---

## Forcing JSON output

Do not ask the model politely for JSON — constrain what it is allowed to emit.
At each step the sampler discards any token that would break the structure, so
invalid JSON becomes impossible rather than merely unlikely.

```bash
llama-cli -m ~/daedalus/instruct/model-q4_0.gguf --chat-template chatml \
  --grammar-file answer.gbnf -st \
  -p "What is the capital of France?"
```

`answer.gbnf` ships in the repository root. It avoids the counted-repetition
syntax (`{1,2}`, `{0,20}`) that some llama.cpp builds reject when generating a
grammar from `--json-schema`.

Training on JSON examples would only make the shape *likely*. The grammar
guarantees the shape; training would improve the content inside it.

---

## Running on CPU rather than GPU

On a Mac, llama.cpp offloads to Metal by default. The speed claims for this
model are about **CPU** decoding, so force it:

```bash
llama-cli -m ~/daedalus/instruct/model-q4_0.gguf -ngl 0 ...
```

Expect roughly 400–450 tokens/second on an M3 Pro.

---

## Available files

All under [`Unseen1980/daedalus-checkpoints`](https://huggingface.co/Unseen1980/daedalus-checkpoints):

| File | Size | What |
|---|---|---|
| `instruct/model-q4_0.gguf` | 102 MB | chat model — start here |
| `gguf/hero-base-q4_0.gguf` | 102 MB | base model, text completion |
| `gguf/instruct-f16.gguf` | 323 MB | instruct at half precision |
| `gguf/hero-base-f16.gguf` | 323 MB | base at half precision |
| `hf/instruct/`, `hf/base/` | 321 MB | safetensors, for transformers |
| `final/hero/checkpoint.pt` | 1.4 GB | base weights + optimiser state |
| `final/post-sft/final.pt` | 642 MB | instruct weights, full precision |

The half-precision files are there so you can re-quantise to a different format
without retraining anything.

---

## What to expect

| | |
|---|---|
| 5-task benchmark | 47.31 |
| Validation bits-per-byte | 0.8685 |
| CPU decode | ~440 tokens/second |
| Size on disk | 102 MB |

It is a 150M-parameter model. It writes fluent, plausible text and gets plenty
of facts wrong. The fair comparison is GPT-2 124M, which it beats — not a
modern chat assistant.

Short factual answers work well. Explaining a concept works. Open-ended creative
writing is the weakest case: with no factual anchor, it drifts after a few lines.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Repeats one phrase forever | no repeat penalty | add `--repeat-penalty 1.15` |
| Answers a different question | base model given an instruction | use the instruct model, or give the base model a sentence opening |
| Forum or blog boilerplate | chat markup wrapped a base-model prompt | use `llama-completion` |
| `Failed to initialize samplers` | build rejects the generated grammar | use `--grammar-file answer.gbnf` |
| Very slow | running on one thread | add `-t 8` |

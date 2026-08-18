"""Daedalus-150M on a free CPU Space.

The model is 102 MB quantised to 4 bits, which is the point: it was designed to
run on an ordinary CPU rather than a GPU, so a basic Space is the honest place
to demonstrate it.
"""

import os
import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

REPO = "Unseen1980/daedalus-checkpoints"
N_THREADS = max(2, (os.cpu_count() or 2))

MODELS = {
    "Instruct (chat)": "instruct/model-q4_0.gguf",
    "Base (text completion)": "gguf/hero-base-q4_0.gguf",
}

_loaded: dict[str, Llama] = {}


def get_model(name: str) -> Llama:
    """Load on first use and keep it. Both models together are ~200 MB."""
    if name not in _loaded:
        path = hf_hub_download(repo_id=REPO, filename=MODELS[name])
        _loaded[name] = Llama(
            model_path=path,
            n_ctx=2048,
            n_threads=N_THREADS,
            verbose=False,
        )
    return _loaded[name]


def chat(message, history, temperature, top_p, repeat_penalty, max_tokens):
    """Instruction-tuned model, ChatML format."""
    llm = get_model("Instruct (chat)")

    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": m}
                for turn in (history or []) for i, m in enumerate(turn) if m]
    messages.append({"role": "user", "content": message})

    prompt = ""
    for m in messages:
        prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"

    out = llm(
        prompt,
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        repeat_penalty=float(repeat_penalty),
        stop=["<|im_end|>", "<|im_start|>"],
    )
    return out["choices"][0]["text"].strip()


def complete(prompt, temperature, top_p, repeat_penalty, max_tokens):
    """Base model: continues text, does not answer questions."""
    llm = get_model("Base (text completion)")
    out = llm(
        prompt,
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        repeat_penalty=float(repeat_penalty),
    )
    return prompt + out["choices"][0]["text"]


INTRO = """
# Daedalus-150M

A **150M-parameter** language model built for CPU inference. It keeps full
attention in only 6 of its 18 layers; the other 12 use short convolutions whose
memory is two timesteps wide however long the conversation gets. That is why it
does not slow down as context grows.

Running here on a **free CPU Space** — the hardware it was designed for.

| | |
|---|---|
| Benchmark (5-task mean) | **47.31** — beats GPT-2 124M, Pythia-160M, OPT-125M, GPT-neo-125M |
| Training data | 59.9B tokens |
| Size on disk | 102 MB (4-bit) |
| Speed | ~440 tokens/sec on a laptop CPU |

**Set expectations:** at 150M parameters this writes fluent, plausible text and
gets plenty of facts wrong. The fair comparison is GPT-2, not a modern
assistant. Short questions work best; long creative writing drifts.
"""

with gr.Blocks(title="Daedalus-150M", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)

    with gr.Accordion("Sampling settings", open=False):
        gr.Markdown(
            "`repeat penalty` matters most — at 1.0 the model can lock onto a "
            "word and repeat it until it runs out of tokens."
        )
        temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="temperature")
        top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top_p")
        repeat_penalty = gr.Slider(1.0, 1.5, value=1.15, step=0.05, label="repeat penalty")
        max_tokens = gr.Slider(16, 256, value=96, step=16, label="max new tokens")

    with gr.Tab("Chat"):
        gr.ChatInterface(
            fn=chat,
            additional_inputs=[temperature, top_p, repeat_penalty, max_tokens],
            examples=[
                "What is the capital of France?",
                "Explain photosynthesis in one sentence.",
                "What is the difference between a CPU and a GPU?",
                "Give me three tips for learning to cook.",
            ],
            cache_examples=False,
        )

    with gr.Tab("Text completion (base model)"):
        gr.Markdown(
            "The base model **continues text** rather than answering. Give it the "
            "beginning of a sentence, not an instruction."
        )
        inp = gr.Textbox(
            label="Start of a sentence",
            value="Photosynthesis is the process by which",
            lines=3,
        )
        out = gr.Textbox(label="Continuation", lines=8)
        gr.Button("Continue", variant="primary").click(
            complete, [inp, temperature, top_p, repeat_penalty, max_tokens], out
        )
        gr.Examples(
            [["The capital of France is"],
             ["The Second World War began in"],
             ["def fibonacci(n):"]],
            inputs=inp,
        )

    gr.Markdown(
        "Model: [Unseen1980/daedalus-checkpoints]"
        "(https://huggingface.co/Unseen1980/daedalus-checkpoints) · "
        "Code and paper: [unseen1980/daedalus](https://github.com/unseen1980/daedalus) · "
        "Apache 2.0"
    )

demo.queue(max_size=8).launch()

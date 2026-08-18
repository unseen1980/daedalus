---
title: Daedalus-150M
emoji: 🪶
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: A 150M model built for CPU inference - try it here
---

# Daedalus-150M

A 150M-parameter language model designed for CPU inference. Two thirds of its
layers use short convolutions with a fixed-size state instead of attention, so
decoding does not slow down as the conversation grows.

This Space runs it on a free CPU instance — the hardware it was built for.

- Model: [Unseen1980/daedalus-checkpoints](https://huggingface.co/Unseen1980/daedalus-checkpoints)
- Code and paper: [unseen1980/daedalus](https://github.com/unseen1980/daedalus)

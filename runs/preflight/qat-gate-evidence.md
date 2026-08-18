# QAT gate evidence — the CUDA-gated tests, run on an idle box

Written by `scripts/qat_tests_when_quiet.sh` at 2026-08-11T07:39:08Z.
This is blocking precondition 2 of the `hero` gate.

## Verdict: **PASS**

| | |
|---|---|
| passed | 41 |
| **skipped** | **0** (pass condition is **0**) |
| failed | 0 |
| pytest rc | 0 |

Zero skips is the pass condition, not "no failures": these tests
self-skip while a trainer owns the GPU, and pytest exits 0 when they do.

```
.........................................                                [100%]
=============================== warnings summary ===============================
tests/test_qat.py::test_q4_0_matches_real_llama_cpp_bit_for_bit
  /workspace/daedalus/tests/test_qat.py:105: UserWarning: The given buffer is not writable, and PyTorch does not support non-writable tensors. This means you can write to the underlying (supposedly non-writable) buffer using the tensor. You may want to copy the buffer to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_new.cpp:1576.)
    d = float(torch.frombuffer(raw[off:off + 2], dtype=torch.float16)[0])

tests/test_qat.py: 14 warnings
  /venv/main/lib/python3.12/site-packages/torch/jit/_script.py:365: DeprecationWarning: `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
41 passed, 15 warnings in 14.61s
```

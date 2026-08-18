# When the `hero` gate can actually open — the post-arm chain, timed

**2026-08-11 00:35Z.** `STATUS.md` has said "the gate opens **11th ~07:05Z**,
launch ~07:15Z" since the gate was drafted. Timing the chain that has to run
first puts it at **~07:47Z and ~07:52Z** — **42 minutes later**.

Nothing is broken and no job is missing. The estimate simply never priced the
longest step in the chain: `finish_dense_arm.py` computes `val_bpb` over the
**full 1.3 GB holdout**, which for arm 1 took most of 41 minutes.

## The chain, and why it is serial

Four unattended jobs run between arm 2's last training step and a launchable
`hero`. Each waits for the previous one because each wants the box to itself:

| # | job | needs | why it waits |
|---|---|---|---|
| 1 | `finish_dense_arm.py --wait` | GPU | arm 2's trainer must be gone |
| 2 | `score_dense_arm.sh` → `eval.py` | GPU | waits for `results.json`, i.e. for (1) |
| 3 | `rebench_when_quiet.sh` → `rebench_arms.py` | **CPU** | waits for no `train.py`/`eval.py`; a GPU job still burns CPU, and this measures *CPU decode* |
| 4 | `qat_tests_when_quiet.sh` → `pytest tests/test_qat.py` | GPU | waits for no `rebench_arms.py`/`llama-bench`, so its GPU work cannot perturb (3) |

The ordering is right and I am not changing it. (3) produces the paired decode
ratio — the headline Pareto number the ablation exists for — and it is the one
measurement a co-tenant would corrupt rather than merely slow.

## Measured timings

| stage | minutes | where the number comes from |
|---|---:|---|
| arm 2 training remaining | — | 2.29B left at the steady 108,000 tok/s → **ends ~06:25Z** |
| (1) val_bpb + export + decode bench | **45** | arm 1: trainer gone 16:36:39Z → export files written 17:18:03Z = **41.4 min**, of which 9.9 shared the GPU with the 5-task eval. Dense forward is ~11% slower (18 attention blocks vs 6), so alone-but-slower ≈ 45 |
| (2) poll + 5-task eval | 11.9 | arm 1's eval measured **9.9 min** (16:37:36Z → 16:47:31Z), plus one ≤120 s poll |
| (3) quiet wait + paired decode | 25.5 | 2 polls (≤240 s) + 90 s settle, then ~20 min for the alternating pass |
| (4) quiet wait + QAT tests | 5.1 | 2 polls + 60 s settle, then **2.6 s** — measured tonight, `32 passed, 7 skipped` |

```
06:25Z  arm 2 training ends
07:10Z  results.json           <- (1)
07:21Z  dense arm's 5-task eval done   <- (2)
07:47Z  decode-paired.json     <- (3)   GATE CAN POST
07:52Z  qat-gate-evidence.md   <- (4)   HERO CAN LAUNCH
```

Uncertainty is **±15 min**, essentially all of it in (1): the arm-1 basis is one
observation, and it was contended for a quarter of its length.

## Why the 42 minutes matter, beyond being late

The QAT stage is the launch precondition — `hero._cli` exits **rc 2** without
`runs/preflight/qat-gate-evidence.md`. So had the gate gone up at 07:05Z as
promised and been answered promptly, **`hero` would have refused**, for the
second night running and for a different reason than last night's mixture skew.

It would have been a recoverable refusal — wait, relaunch — not a lost run. But
it is avoidable by stating the real time, and "gate opens 07:05Z" sitting in
`STATUS.md` while nothing appears until 07:47Z reads from a phone exactly like a
stall. `HEARTBEAT.md` computes its stall verdict from time-since-last-real-commit
plus GPU utilisation, and the box is busy throughout — so the file would say
healthy while the document said late.

## What I considered changing, and did not

**Move the QAT tests ahead of the re-bench.** They take 2.6 s and want the GPU,
which `rebench_arms.py` does not; `rebench_when_quiet.sh` already excludes
`pytest` from its busy check, so the two would not fight. That would satisfy the
launch precondition ~25 min earlier.

It buys nothing. The **gate** cannot post without `decode-paired.json` either
way, so the critical path is (3) regardless of where (4) sits. Re-arming a live
unattended script hours before a $59.85 launch, to save time on a stage that is
not the constraint, is a bad trade.

**Shorten the full-holdout `val_bpb`.** No: it is the ablation's decision
criterion, and arm 1 was measured the same way. Changing it for arm 2 alone
would make the two arms incomparable, which is the one thing this experiment
cannot afford.

# `hero._cli`'s prefix, executed rather than reasoned about — 2026-08-11 06:38Z

## Why

Fixed 11 (`STATUS.md`) found that every pasteable launch command began with an
interpreter that does not exist on this box, and that under the mandated
`setsid nohup … &` the failure lands in a log while the shell reports rc 0. The
question that found it — **has anyone ever run this?** — has one more layer
underneath it: the interpreter now resolves, so `hero.py` *starts*. Does it get
as far as spawning `train.py`?

At 02:42Z the three gate functions were called directly, in-process. That covers
`check_mixture` / `check_qat_evidence` / `check_gpu_free`. It does **not** cover
`_cli`'s glue between them, and one line there is unconditional:

    hero.py:393    table = format_mixture_note(summary, args.total_tokens)

It runs on the passing path as well as the failing one, immediately after the
preflight, before anything is launched. It had never been rendered against the
real 58B summary — only against fixtures. A raise there (a `None` epochs, a
divide by a zero median) exits `hero` with a traceback into
`/tmp/hero-launch.log` seconds after $59.85 is approved, with the shell showing
success. Same shape as fixed 11, one line further along.

## What was run

Isolated cwd, so nothing could touch the repo's `STATUS.md` or `runs/`
(`data` symlinked in read-only usage; `runs/` a fresh empty dir):

```
/tmp/hero-probe/
  data -> /workspace/daedalus/data
  runs/                                   (empty, real dir, not a symlink)
```

```
cd /tmp/hero-probe
/venv/main/bin/python /workspace/daedalus/hero.py \
    --run-name hero-probe \
    --data-dir data/shards-hero-split/train \
    --val-dir data/shards-hero-split/holdout \
    --total-tokens 58000000000 --muon-lr 0.02 --micro-batch 16 \
    --qat-evidence /tmp/hero-probe/definitely-absent.md
```

The QAT evidence path is pointed at a file that cannot exist, so the run stops
at gate 2 by construction. That is deliberate: it executes the whole prefix at
the real ask and **cannot** reach `run_with_resume`.

`--data-dir`/`--val-dir` are pinned so the carve is skipped — that path is
already covered by
`test_hero_relaunching_the_carve_reproduces_the_split_it_was_approved_on`, and
re-carving beside a live GPU job is not worth the I/O.

## Result — rc 2, both `note_in_status` branches executed

```
[hero] mixture preflight: 16,932,674,383 tokens on disk, l1_skew 4.94 pts,
       worst repetition 4.0x (limit 4.0)
[hero] REFUSING TO LAUNCH: `…/definitely-absent.md` does not exist — the
       CUDA-gated QAT tests have not been re-run on an idle box
rc=2
```

`format_mixture_note` rendered against the real summary, and the table it
produced reproduces the one in `STATUS.md` source for source — which is a
genuine cross-check, because that table was computed by a different path
(`scripts/`-side, at 02:15Z) and this one comes off `check_mixture`'s own
`summary["per_source"]` inside the launcher:

| source | target → effective | epochs |
|---|---|---|
| dclm-baseline | 23.0% → 22.8% | 4.00× (capped) |
| finepdfs-edu | 8.2% → 7.8% | 4.00× (capped) |
| fineweb-edu | 38.3% → 38.0% | 4.00× (capped) |
| finewiki-en | 3.1% → 3.0% | 4.00× (capped) |
| stack-edu-python | 9.2% → 7.6% | 4.00× (capped) |
| cosmopedia-v2 | 5.1% → 5.8% | 3.73× |
| finephrase | 7.1% → 8.1% | 2.38× |
| finemath-3plus | 3.1% → 3.5% | 1.57× |
| infiwebmath-3plus | 3.1% → 3.5% | 1.56× |

The outlier rule reported *"Epochs are even: max 4.00× vs median 4.00×, no
source far above the rest"* — i.e. it did not fire, which is the correct verdict
here and is the row the operator asked for before launch.

Peak RSS 0.5 GB, no GPU, nothing written outside `/tmp/hero-probe`. Arm 2's
finisher was untouched.

## What this does and does not prove

**Proven executed at the real 58B ask:** `_cli` lines 361-437 — run-dir
creation, the mixture preflight, `format_mixture_note`, `note_in_status` on
*both* the passing and the refusing branch, and the QAT gate's refusal text.

**Still not executed as a whole:** lines 470-503 — `build_train_cmd` →
`start_watchdog` → `run_with_resume`. `run_with_resume` was rehearsed for real
on CPU with a real SIGKILL and a real `--resume`
(`runs/preflight/supervisor-rehearsal.md`), and `build_train_cmd` is covered by
`tests/test_hero.py`, but the three have never run in sequence from `_cli`.
They cannot be, without launching a real trainer — which is what the gate is
for.

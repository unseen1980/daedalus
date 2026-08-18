# Draft: `[ASK HUMAN] ready for hero`

**Not yet posted.** Written 2026-08-09 ~23:55Z, while `dataprep` finishes, so
that the gate goes up within minutes of `abl-arch` ending rather than after an
hour of assembling numbers — the box idles at $10.78/day from that moment until
the operator answers.

**Opens ~07:47Z on the 11th** — *re-timed 2026-08-11 00:35Z, and it moved 42
minutes later; see the note under the table.* The chain is not just two training
arms: each is followed by a full-holdout val pass and an export, and two more
jobs run after arm 2's.

| | ends |
|---|---|
| arm 1 train (11.33 h @ 122,612 tok/s) | 10th 16:32Z |
| arm 1 full-holdout val (650.8M tokens, **measured 31 min**) | 10th 17:03Z |
| arm 1 export + GGUF + decode bench (**measured 10 min**) | 10th 17:13Z |
| arm 2 train (2.29B left at the steady 108,000 tok/s) | 11th ~06:25Z |
| arm 2 val + export (`finish-dense-arm.py --wait`, armed) — 45 min: arm 1's 41 min, and dense forward is ~11% slower | 11th ~07:10Z |
| arm 2 5-task eval (`score-dense-arm.sh`, armed — fires on `results.json`) — ≤120 s poll + arm 1's measured 9.9 min | 11th ~07:21Z |
| **paired decode re-bench** (`rebench_when_quiet.sh`, armed — waits for a quiet box) — 2 polls + 90 s settle + ~20 min | 11th ~**07:47Z** |
| **QAT evidence** (`qat_tests_when_quiet.sh`, armed — blocking precondition 2) — 2 polls + 60 s settle + 2.6 s | 11th ~**07:52Z** |

> **This table said 06:45Z / 07:05Z / 07:15Z until 00:35Z on the 11th.** Two
> errors, in opposite directions and both since fixed above. Arm 2's train-end
> was carried at 06:04Z from a projection made before the OOM restart; it is
> ~06:25Z on the metrics the trainer is writing now. And the re-bench was given
> **10 minutes** in this table while the prose four paragraphs below priced the
> same job at *"~20 min, ~$0.15"* — the document contradicted itself, and the
> shorter number was the one that reached the summary. Timings and their
> provenance: `runs/preflight/post-arm-chain-timing.md`. Uncertainty ±15 min,
> essentially all of it in the val_bpb row.

**The last row is why posting the gate and launching `hero` are different
times.** `hero._cli` refuses with rc 2 until `runs/preflight/qat-gate-evidence.md`
exists, and `qat_tests_when_quiet.sh` waits for the re-bench to release the GPU
before it can produce it — its `busy()` covers `train.py`, `eval.py`, `rebench_arms.py` and `llama-bench`,
because the seven tests self-skip while anything holds the GPU and a suite
reporting "7 skipped" exits 0. From a quiet box it needs two 120 s polls, a 60 s
settle and the run itself (2.9 s with the CUDA tests skipping; allow ~2 min with
them). So **post at ~07:47Z, launch no earlier than ~07:52Z** — and the refusal
is the safety net, not the schedule: if the evidence is late, `hero` says so
instead of training the last 3B tokens on an unverified quantisation lattice.

The re-bench row is a required step, not an optional one: each arm benchmarks
itself inside its own export, ~12 h apart, and that method measured ~8% high
against a paired pass on the same files (`runs/eval/decode-hybrid-vs-dense.md`).
`abl_table.py` picks the sidecar up automatically and marks the ratio
"indicative" if it is missing, so a forgotten step is visible rather than silent.

**Nothing was running it.** `after_abl_arch.sh` ran it as step 1, and that chain
was disarmed when `abl_arch.py` was killed to free the VRAM that was OOM-ing the
dense arm; the two waiters armed in its place do the val/export and the 5-task
eval and neither re-benches. Armed 19:59Z as `scripts/rebench_when_quiet.sh`.

**So do not post this gate at ~07:10Z.** The re-bench now runs *after* the 5-task
eval rather than before it — it waits for a quiet box, because `score-dense-arm.sh`
starts `eval.py` within 120 s of `results.json` and alternating rounds cancel
drift, not a co-tenant. Post once `runs/abl-arch/decode-paired.json` exists,
~07:47Z — and launch once `runs/preflight/qat-gate-evidence.md` exists, ~07:52Z
(the row above). The ~20 min costs ~$0.15 and is the difference between a
measured headline ratio and one `abl_table.py` labels indicative.

**25.5 h total → $11.47**, against the $11.46 carried in `COSTS.md`. The two val
passes are 1.05 h of that and were already priced separately; recomputing them
independently from the holdout's real token count (650,817,766) was a check on
that line, and it holds.

Everything marked `<<TODO>>` needs `abl-arch`'s real output. Everything else is
measured and final. Fill from `runs/abl-arch/results.json`, then post.

**How to post it — do not pass this file to `gh` (2026-08-11 01:55Z).** Its body
was **88,308 characters** and **GitHub refuses an issue body over 65,536**:

    GraphQL: Body is too long (maximum is 65536 characters) (createIssue)

— measured against a size-matched filler body, which created nothing. So
`gh issue create --body-file runs/preflight/hero-gate-draft.md` would have failed
at ~07:47Z, posted nothing, and left the box renting at $0.449/h until
`gate_deadman.sh` noticed three hours later. Post the extracted body instead:

```
/venv/main/bin/python scripts/gate_body.py --check                       # refuses an unpostable body
/venv/main/bin/python scripts/gate_body.py --out /tmp/gate-post.md
gh issue create --title '[ASK HUMAN] ready for hero' --body-file /tmp/gate-post.md
echo "rc=$?"     # check gh directly; piping into `tail` reports the pipe's status
```

Sections whose heading is followed by `<!-- appendix -->` stay in this file and
are linked from the post rather than deleted; `tests/test_gate_body.py` asserts
the post fits, that every reply option keeps its launch command, and that
nothing trimmed is missing from here.

---

## Body

### Decide from this box; everything below is the evidence for it

**Asking for:** `hero` at **58B** — **~133.3 h, ~$59.85**, on the config below.
It does not start without your go-ahead.

> ### ⚠ Read this first: the hourly rate does not match what everything here is costed at
>
> **You are the only one who can settle this**, and it is a ten-second check on
> the vast console from a phone. I cannot do it: reading the balance means
> calling `vastai`, which your standing instructions forbid outright.
>
> Between your two stated balances — **$94.66 @ 2026-08-09 23:50Z** and
> **$99.00 @ 2026-08-11 06:45Z** (after you added $25, so $74.00 before it) —
> **30.92 h elapsed and $20.66 disappeared**. That is **$0.668/hr**, against the
> **$0.449/hr** every number in this document is built on. **49% higher.**
>
> | at | $0.449/hr (assumed) | $0.668/hr (measured) |
> |---|---|---|
> | `hero` at **58B** (133.3 h) | $59.85 | **$89.04** |
> | credit at completion | **≈$30.73** | **≈ −$2.56 — does not finish** |
> | `hero` at **51B** (117.2 h) | $52.62 | $78.29 |
> | credit at completion | ≈$37.96 | **≈$8.19 — closes either way** |
>
> **The most likely explanation is bandwidth, and it is largely spent.** That
> window is exactly when the 34 GB corpus and the first checkpoints went to the
> Hub; vast bills egress separately from the $0.449 GPU+storage line. `hero`
> ships ~24 GB over 5.9 days — a far lower rate — so the true figure is probably
> between the two. **Probably is not a budget**, and you have no more money
> behind this one.
>
> **So:** reply `go` for 58B **if the console shows ≈$0.449/hr**. Reply
> `go 51B` if it shows anything near $0.668, or if you would rather not check —
> 51B closes under either rate, costs one point of headline quality at most, and
> its WSD schedule is already verified. Tell me the rate either way and I will
> re-anchor `runs/credit-anchor.json` and re-price everything from it.
>
> **This is late and that was deliberate.** The chain finished at 07:39Z and this
> gate was ready to post at 07:47Z quoting $94.66 and "≈$12.62 left". The
> re-anchor landed in `runs/credit-anchor.json` at 06:51Z and **nothing
> propagated it** — not this draft, not `COSTS.md`. It was caught only because
> `test_the_anchor_shipped_in_the_repo_is_the_operators_stated_figure` failed in
> the pre-post suite run, still asserting $94.66. Posting on time with the wrong
> money would have been the worse trade.

> **This was 60B until 23:40Z, and 60B does not launch.** The top-up finished,
> so I carved the real split and ran the preflight on it instead of on a
> projection: `l1_skew` **10.21** against `hero`'s 10.0 limit. You would have
> replied `go` and got rc 2. The margin that said "LAUNCH by 0.64%" modelled
> the holdout as a flat 2% haircut; the real carve reserves whole shard files
> and takes **3.67%**, unevenly. Root cause, fix and the full table are under
> *The 60B mixture margin* below. **58B is a recommendation, not a
> constraint** — `go 59.9B` also launches and holds your decision to within
> 0.13%.

| | |
|---|---|
| **Config** | **58B tokens**, `daedalus-150m`, Muon lr 0.02, micro-batch 16 |
| **Schedule** | 120,528 steps, decay from **66,290** (55.000%), QAT over the last 2.90B — recomputed at 58B by replaying the ramp, final lr multiplier 0.00002 |
| **Money** | **~$59.85** (~5.6 days), against ~$61.89 at 60B |
| **Credit at the end** | **~$30.73** at $0.449/hr, from your stated **$99.00** of 06:45Z, after a 5.0 h response-time allowance ($2.25) and `post` + eval + GGUF ($5.50). **At the measured $0.668/hr it is −$2.56 and the run does not finish** — see the box above. Was "~$12.62" until 07:55Z, against the superseded $94.66 anchor |
| **Prerequisite** | the corpus top-up (issue #5) — **already done**, run beside `abl-arch` arm 2 rather than after it, so it costs **no extra rent** and floors nothing. `hero` starts the moment you reply |
| **Bar to beat** | **42.2** (5-task mean, our harness — clears every 300B-class peer). A **2σ** win over them needs **42.7–43.8** |
| **Where we are** | **44.68 at 5B tokens** — `abl-arch` arm 1, fully decayed, measured 2026-08-10 16:47Z. **Already above the bar**, and above all four 300B-class peers on every one of the five tasks. Prior point: 40.2 at 0.5B |
| **Margin at 58B** | **comfortable: skew 4.94 against a 10.0 limit.** Measured on the materialized split, not modelled. The ceiling is **59.92B** — 60B misses by 0.14%, and the curve is a cliff over the last 5B (2.37 → 4.94 → 10.21) because each extra billion pushes another source onto the 4-epoch cap. See *The 60B mixture margin* below |
| **Epochs** | **3.43** over the 16.93B train split (was ~4.3 when the corpus was 13.94B — the top-up bought repetition headroom as well as tokens). Worst single source sits on the 4.00 cap |
| **Honest verdict** | **materially de-risked, still not assured** — the bar is cleared by 2.5 points at **one twelfth** of `hero`'s budget, but 58B has to hold that and no amount of arithmetic on two points makes a third a measurement |

**The buffer is ~$30.73, and only at $0.449/hr.** Your $25 top-up is what moved
it — this line read "$12.62" against the old anchor.
It is not slack: historically ~27% of spend here went to failed attempts, and on
the $65 still to come that ratio would be about $18 — and at the measured
$0.668/hr the figure above is **negative** before any rework at all.
**The arithmetic only closes if `hero` runs once.** That is what the checkpoint
durability work buys, and why the uploader wedge found this morning mattered out
of proportion to its size: without the fix, one hung transfer would have made a
restart cost the whole run instead of the two hours since the last upload.

> **Correction, 2026-08-10 15:55Z.** This block and the summary row above read
> **~$2.41** until now, and the *Projected credit* section below has always said
> **$9.70** — two different end-of-run balances in the one document that
> authorises the largest expense in the project, which is the same defect found
> in this draft's ask this morning. **$9.70 is the right one**, confirmed two
> ways: forward from your stated $94.66 of 2026-08-09 23:50Z (30.92 h of rent to
> gate-open leaves $80.78, then −1.89 −1.80 −61.89 −5.50), and forward from the
> $87.80 balance at 15:07Z. The $2.41 subtracted this project's **cumulative**
> ~$27.87 of spend from a balance that already reflected most of it — a
> double-count, and in the pessimistic direction. The `go 40B` figure was wrong
> the same way and is **~$32.2**, not ~$22.
>
> **Superseded in one term by the 16:20Z correction below:** the `−1.89 −1.80`
> above billed the top-up and the gate wait as separate rent for the *same*
> hours — the top-up runs *during* the wait. They are now one **−$2.25** line and
> the end figure is **$11.14**, not $9.70. Also pessimistic, also corrected
> rather than left to be discovered. *(That $11.14 was itself refreshed to
> **$10.95** at 21:55Z on a newer balance basis — the current figure is the one in
> the summary row and in* Projected credit*, not this historical note.)*

**A 60B prerequisite — resolved before this gate opened, and worth knowing about
anyway (issue #5).** 4
epochs × the 14.218B corpus is 56.87B, so at 60B `cap_weights_by_epochs` cannot
satisfy every cap and returns the target shares untouched — the repetition guard
switches off. `everyday-conversations` would then take 2% of 60B (1.2B tokens)
from a dataset exhausted at 403,573, i.e. ~2,200 conversations repeated **2,973
times**; and `l1_skew_pts`, the graded signal, reads **0.00** there against
29.91 at 50B. Adding 3.50B tokens to five short sources (**~5.0 h, ~$2.25**)
restores it to skew 3.99 — identical to 40B today — with every source ≤4.00
epochs.

**This is done, and it cost neither money nor delay.** It ran *beside* `abl-arch`
arm 2 from 18:22Z on the 10th (`scripts/topup_beside_arm2.py`, `--max-workers 1`,
measured peak **11.0 GB** tree RSS against the 20 GB ceiling — worst case
1.0 GB parent + 2 × the 6.0 GB per-worker cap = 13.0 GB, because a soft-RSS
respawn briefly overlaps the outgoing and incoming worker) instead of serially
after it, so its ~6 h
sit inside hours already billed to `abl-arch` — **zero incremental rent** — and
it finished at **23:29:45Z, rc 0** (5.12 h, 18:22:30Z → 23:29:45Z — an hour
earlier than the ~00:35Z projected), hours before this gate opened.
`hero` therefore has no start floor: it begins when you reply, not 5 h later.
Arm 2's throughput was unaffected (108.5k tok/s against the 107.6k baseline the
guard was armed on).

> **Correction, 2026-08-10 16:20Z.** This read **4.2 h / $1.89** until now, and
> called the wall-clock free. Both were wrong. The 4.2 h came from "the measured
> 0.83B tok/h", which is an *aggregate over 3–6 concurrent workers*, applied to
> a script that runs fewer. Re-measured per source from the 2026-08-09 build's
> own shard mtimes: `fineweb-edu` 0.629 B/h, `dclm-baseline` 0.599, `finepdfs-edu`
> 0.255, `finewiki-en` 0.153. `fineweb-edu` and `dclm-baseline` share a
> near-dup group, so 3.00B of the 3.50B is **serial at any worker count** —
> a 4.86 h floor. The script now runs 2 workers (15.0 GB projected against the
> 20 GB ceiling), which hides the three small groups inside that floor: **~5.0 h**.
> → `runs/preflight/topup-duration.md`

**What you are actually deciding.** Not whether the pipeline works — that is
measured. Whether to spend $59.85 on a *bet* whose margin is thinner than the
noise between the peers — and as of this morning that is a measurement rather
than a turn of phrase: the 5-task mean carries **±0.59 points** of binomial
sampling error, so a difference must reach **1.65 points** to be two sigmas.
A 42.5 finish would beat this peer group as measured and be a statistical
**tie**, and it will be reported as one. (Worse, not better, than that sounds:
±0.59 is sampling error alone and excludes seed variance, which is unmeasured
here and reported at 2–3 points at this scale.)

### The 60B mixture margin — CORRECTED 23:40Z: **60B refuses. 58B is the ask.**
<!-- appendix -->

> **And the split you approve is the split that runs — checked 02:20Z.** The
> launch commands do not pin `--data-dir`, so `hero._cli` **re-carves** the
> holdout from `data/shards` before it gates on the mixture. Every number here
> is measured on the already-materialized split, so a carve that differed at all
> would mean approving one mixture and training on another, with only 5.06 pts
> of skew margin to absorb it. Re-carved into a scratch root and compared:
> **identical, source for source** — 9 sources, 16,932,674,383 train /
> 644,478,630 holdout. Deterministic by construction (`select_holdout_shards`
> takes whole shards from the end of the manifest), but it had never been run.
> Test: `test_hero_relaunching_the_carve_reproduces_the_split_it_was_approved_on`.

> **This section said "LAUNCH, by 0.64%" until 23:40Z on the 10th. It was
> wrong, and it was the number this whole gate turned on.**
>
> The top-up finished at 23:29Z, so I carved the real split and ran
> `hero.check_mixture` on it instead of on a projection. **`l1_skew` 10.21
> against a 10.0 limit — `hero` exits rc 2 at 60B.** You would have replied
> `go` and got a refusal.
>
> Both numbers came from `scripts/mixture_margin.py`. The difference is that
> the old one *modelled* the holdout as a flat `n × (1 − 2%)` haircut, under a
> docstring claiming it was "carved like the real split". The real carve
> reserves whole shard **files**, so a source whose last shard is small
> relative to the 2% target must also give up the ~100M shard before it:
>
> | source | shards | real carve |
> |---|---|---|
> | `stack-edu-python` | 13, tail 10.96M vs a 24.2M target | **9.16%** |
> | `cosmopedia-v2` | 26 | 5.32% |
> | `finepdfs-edu` | 33 | 5.20% |
> | `fineweb-edu` | 113, tail large enough | 2.06% |
> | `everyday-conversations` | **1 — cannot be split, dropped entirely** | 100% |
>
> 3.67% carved overall against a 2.0% target, and *unevenly* — which moves the
> mixture, the only quantity this gate turns on. `stack-edu-python` falls from
> a 9.2% share to 7.3%; `finephrase` rises to 9.4%.
>
> Fixed at the root: `select_holdout_shards` is extracted from
> `make_holdout_split` and `mixture_margin` now calls that function rather than
> imitating it, so the mirror **is** the selector. It reproduces the
> materialized split source by source
> (`test_the_model_agrees_with_the_split_hero_will_actually_read`).

Measured on the real `data/shards-hero-split/train` (16,932,674,383 tokens):

| budget | l1_skew | verdict |
|---|---|---|
| 55B | 2.37 | LAUNCH |
| **58B — the ask** | **4.94** | **LAUNCH** |
| 59.92B | ~10.0 | LAUNCH — the exact ceiling |
| **60B** | **10.21** | **REFUSE, by 0.14%** |

**58B is recommended over 59.9B, and not as a fallback.** 59.9B would honour
your 60B decision to within 0.13% and it launches — but the skew curve is a
cliff at the top (2.37 → 4.94 → 10.21 over the last 5B), because each extra
billion pushes another source onto the 4-epoch cap and redistributes its mass
to the three with headroom. 58B buys **less than half the mixture skew for 3.3%
fewer tokens**, and your own instruction is that *"corpus mixture balance
matters more than corpus size"*. `go 60B` remains available with
`--allow-skewed-mixture`, which records the override.

`stack-edu-python` **exhausted its stream** 139M short of budget, so a source
running out is not hypothetical on this corpus — it has already happened once,
and that shortfall is baked into the row above.

> **RESOLVED 2026-08-10 ~22:0xZ.** `finepdfs-edu` had to reach ≥ 1,124,340,092
> tokens for 60B to launch. It finished its **full** budget —
> `done finepdfs-edu: tokens=1,200,001,535` — so the threshold is met with
> 75.7M to spare and the REFUSE row above cannot happen.

It was never close to drying up, which is now measured rather than assumed:
`finepdfs-edu` reads **100 files / 298.67 GB** of `eng_Latn` and consumed part
of two. `scripts/source_headroom.py` (26 tests) checked all ten sources against
the real Hub metadata: **`stack-edu-python` is the only short one**, and the only
unfinished source, `fineweb-edu`, has **764x** the headroom it needs (4 of 2,410
files read). The exhausted source is the only one in the mixture reading a small
file set — 10 files, 1.79 GB — which at its achieved density was never going to
reach 1.35B, and was knowable from metadata before it streamed a document.
Detail: `runs/preflight/source-headroom.md`.

I will re-run this against the **final** corpus when the gate goes up and quote
the real ceiling, rather than asking you to act on a projection. Full analysis
and the rejected mitigations — halving the holdout buys only +0.61B and does not
rescue the bad case — in `runs/preflight/hero-60b-mixture-margin.md`.

### Per-source epochs, and the one thing the budget curve says

You asked for the epoch counts before launch. **No source lands far above the
rest**: 1.71 to 4.00 epochs, and the maximum *is* the 4-epoch cap rather than an
outlier past it. Seven of ten sit at 4.00; `everyday-conversations` is pinned to
**0.00%** effective share, which is issue #5's fix working.

Newly measured: how the skew moves with budget.

| budget | l1_skew | capped | |
|---|---|---|---|
| 30–51B | **3.99** | 1 | free — identical to 40B |
| 55B | 4.73 | 2 | |
| 58B | 5.63 | 6 | |
| **60B** | **9.01** | **7** | 0.99 pts under the refusal limit |

**The corpus is effectively sized for ~51B at 4 epochs**, so 60B's thin margin is
the price of one notch past the knee, not an error. **I still recommend 60B** —
repetition is bounded at exactly the cited guideline, and the skew's composition
(less code, more synthetic/maths) does not point at the five tasks we are judged
on, none of which is a code task. But `go 51B` is a real option and you should
see it here rather than afterwards.

**Reply with one of:** `go` (**58B**, $59.85, **~$30.73** credit left, skew 4.94 — recommended **only if the console shows ≈$0.449/hr**) ·
`go 59.9B` (your 60B to within 0.13%, $61.80, skew ~10.0 — the ceiling) ·
`go 60B anyway` (needs `--allow-skewed-mixture`; skew 10.21, and the override is recorded) ·
`go 51B` (**~$9.3 cheaper**, skew 1.11, 7B fewer tokens — **closes at either hourly rate; the safe reply if you would rather not check**) ·
`go 40B` ($41.26, **~$52.0** credit left, skew 0.00) · `hold`.

`go 58B` has been **withdrawn**: it existed only in case `finepdfs-edu` fell
short, and that source is now complete, so it would cost 2B tokens to insure
against a risk measured at zero.

Both start immediately — the corpus top-up that used to make 40B the faster
option is already built, so **40B now buys only money, not time**. It is the
same corpus either way; 40B simply reads 2.9 epochs of it instead of 4.3.

### On approval, exactly this runs — nothing improvised

```bash
/venv/main/bin/python hero.py --muon-lr 0.02 --micro-batch 16 --total-tokens 58000000000   # 58B — the ask
/venv/main/bin/python hero.py --muon-lr 0.02 --micro-batch 16 --total-tokens 59900000000   # 59.9B — the ceiling
/venv/main/bin/python hero.py --muon-lr 0.02 --micro-batch 16 --allow-skewed-mixture       # 60B, override
/venv/main/bin/python hero.py --muon-lr 0.02 --micro-batch 16 --total-tokens 51000000000   # 51B — the knee
/venv/main/bin/python hero.py --muon-lr 0.02 --micro-batch 16 --total-tokens 40000000000   # 40B
# only if the ablation escalates — see "If the dense twin wins quality"
/venv/main/bin/python hero.py --config dense-150m --muon-lr 0.02 --micro-batch 16 --total-tokens 58000000000   # dense 58B
/venv/main/bin/python hero.py --config dense-150m --muon-lr 0.02 --micro-batch 16 --total-tokens 51000000000   # dense 51B
```

**Each line is the job, and it gets launched detached** — `hero.py` is not a
one-shot: it supervises `train.py` and `watchdog.py` for ~5.6 days, so it has to
outlive the shell that starts it, exactly as every long job on this box has been
started (`abl-arch` arm 2 is running under `setsid nohup` right now):

```bash
setsid nohup <the line above> > /tmp/hero-launch.log 2>&1 &
/venv/main/bin/python scripts/verify_hero_launched.py     # <- do not skip this
```

Run in the foreground it would die with the session — the agent's own shell caps
a command at 600 s, so a literal copy-paste of the line alone loses the run in
ten minutes. This is written down because it never was: the block promised
"nothing improvised" while leaving the one detail that makes the command survive
to the person typing it.

**The second line is not optional, and it is new (2026-08-11 07:05Z).** Making
the interpreter absolute fixed *one* way the detached launch starts nothing, not
the blindness underneath it: `setsid nohup … &` reports **rc 0 whatever happens
to the job**, and `hero._cli` has four exits that train nothing and all look
like success from the shell — the three refusals above (`return 2`) and any
traceback before `run_with_resume` (`format_mixture_note` runs unconditionally
at `hero.py:393`, after the preflight, before anything starts).

**Two of the four are expected to be live when you reply.** This post already
calls the GPU guard "the one most likely to fire", because the gate goes up
while the `abl-arch` chain is still finishing — and the QAT evidence file is
written by the last stage of that same chain. So the likeliest outcome of
pasting the launch line promptly is a clean refusal the shell calls success.

Each refusal does reach `STATUS.md` via `hero.note_in_status`, so it is
recoverable — but passively, and until now this procedure ended at the launch
line and never asked whether a trainer exists. `verify_hero_launched.py` asks:
it polls `supervise.trainer_is_live` (cmdline-checked, not `kill(pid, 0)` —
`train.pid` goes stale and Linux reuses low pids), fails fast on a log that has
already given a verdict, and **exits non-zero with the reason** rather than
letting the box bill $0.449/h having started nothing. 11 tests, one per exit.

Every one of these has had its WSD schedule replayed at that exact budget —
step count, decay-start step, and the final-step lr multiplier — because the
budget is what the schedule is computed *from*, and the one prior WSD bug on
this project was in exactly that arithmetic:

| reply | budget | steps | decay from | final lr mult |
|---|---:|---:|---:|---:|
| `go` | 58B | 120,528 | 66,290 (55.000%) | 0.0000184 |
| `go 59.9B` | 59.9B | 124,476 | 68,461 (54.999%) | 0.0000179 |
| `go 60B anyway` | 60B | 124,684 | 68,576 (55.000%) | 0.0000178 |
| `go 51B` | 51B | 105,981 | 58,289 (55.000%) | 0.0000210 |
| `go 40B` | 40B | 83,123 | 45,717 (54.999%) | 0.0000267 |

`hero.py`'s own default is still **60B** and is deliberately not being changed:
the budget is your decision, so it travels as an explicit flag rather than as a
default I moved while you were asleep.

One command, not two: `scripts/topup_for_60b.sh` already ran beside arm 2, so the
corpus step is gone from this list rather than being re-run. It is deliberately
*not* left in as a harmless retry — it refuses to start beside a trainer
(`topup_for_60b.sh:163-169` — the guard matches `hero\.py` and `train\.py`
alike), so once `hero` is training it would exit 1 and the
non-zero status would look like a failed precondition.

You do not have to take my word for any of the preconditions. `hero._cli` now
re-checks all three itself and refuses with **rc 2** before the split, the
trainer or the watchdog starts, writing the reason to `STATUS.md`:

| it refuses if | override |
|---|---|
| the corpus is short or the mixture skewed | `--allow-skewed-mixture` |
| the CUDA-gated QAT tests are unverified | `--allow-unverified-qat` |
| **another job is still holding the GPU** | `--allow-busy-gpu` |

The third was added at 21:30Z on the 10th and is the one most likely to fire.
Every other job on this box guards this way — `run_dense_arm.sh`,
`score_dense_arm.sh`, `rebench_when_quiet.sh`, `topup_for_60b.sh` — and `hero`,
the most expensive thing that runs here, did not. **The window is certain to
occur:** this gate goes up at ~07:47Z while `score_dense_arm.sh`'s eval and the
paired re-bench are still finishing, and you can reply within minutes.

Launching into that does not fail cleanly. `train.py` would OOM somewhere in the
batch ramp, `run_with_resume` would read it as a crash and retry, and ten
attempts would burn against a card busy for another twenty minutes — while
`STATUS.md` reported a `hero` that had started. This box has OOM'd three times
today; twice it killed arm 2 outright with 620 MiB free. Run against the live
box at 21:30Z the guard correctly refused, naming arm 2's trainer and reporting
**4.1 GB free of 32.6**.

None of this changes what you decide. It changes what happens if you reply while
the box is still busy: a clean refusal I can see and retry, instead of a silent
OOM loop at the top of a 5.9-day run.

Which expands to (rendered from `hero.build_train_cmd`, not retyped):

```
train.py --run-name hero --config daedalus-150m \
  --data-dir data/shards-hero-split/train --total-tokens 58000000000 \
  --muon-lr 0.02 --tags hero --micro-batch 16 \
  --val-dir data/shards-hero-split/holdout --qat-frac 0.05
```

(This block carried **40000000000** until 23:15Z on the 10th, while claiming to
be "rendered from `hero.build_train_cmd`, not retyped" — it was neither. It is
now rendered, and `test_the_rendered_train_command_matches_what_hero_would_really_run`
checks every flag against `build_train_cmd` and the budget against
the **Asking for** line above, so the two cannot drift apart again.)

`hero.py` also carves the 2% per-source holdout, clears any stale watchdog halt
marker, starts the watchdog with `--supervised`, and wraps `train.py` in
`run_with_resume` (10 attempts). Checkpoint durability needs no flag: `--hub-repo`
defaults to `ckpt_uploader.DEFAULT_MODEL_REPO` = **`Unseen1980/daedalus-checkpoints`**
(verified, not assumed — it is why `abl-arch` is uploading tonight without
`abl_arch.py` ever mentioning the Hub).

### The ask

`hero` is the single largest expense in the project: at **58B**, **~133.3 h and
~$59.85**. Per standing instruction it does not start without an explicit
go-ahead.

You set 60B on 2026-08-10 and I am not quietly walking it back: **60B does not
launch.** Measured on the split `hero` actually reads, once the top-up finished
and it could be carved for real, the mixture skew is 10.21 against `hero`'s own
10.0 refusal limit. The ceiling is 59.92B. 58B is where the skew becomes
comfortable (4.94) rather than marginal, and it returns $2.05 to the buffer;
`go 59.9B` keeps your number to within 0.13% and also launches.

(The `go 40B` fallback in the summary is **~91.9 h and ~$41.26** — the same
measurement, 40/60 of the tokens. It never needed the corpus top-up, but that no
longer distinguishes it: the top-up is built, so 40B saves **$20.63 and zero
hours of waiting**, where until today it also saved ~5 h. That figure is
itself down from ~96 h / ~$43.12 as of 06:35Z on 2026-08-10, not a re-estimate
but a replacement of a synthetic benchmark with a production measurement taken
on `abl-arch` at the exact shape `hero` runs at. See "Measured throughput"
below.)

### Winning config

| | |
|---|---|
| architecture | `daedalus-150m` (hybrid, 18 blocks, conv/attention interleave) |
| micro-batch | 16 (measured to fit both arms; hybrid peaks 25.29 GB of 32.6) |
| seq len | 2048 |
| total tokens | **58B** over the finished corpus (**3.43 epochs** of the 16.93B train split, every source ≤4.00) — the top-up it needed is built; see "40B or 30B?" below for the superseded comparison |
| muon_lr | **0.02** — see "the sweep did not discriminate" below |
| WSD decay_frac | 0.45 — 120,528 steps, decay from 66,290 |
| qat_frac | 0.05 (final **2.90B** tokens at 58B — fraction-derived, not an absolute count) |

### The sweep did not discriminate, and that is itself the result
<!-- appendix -->

Re-run 2026-08-10 under the fixed WSD schedule (the first sweep's probes never
annealed, so its winner was thrown out). Three 0.5B-token probes on fresh
`fineweb-edu`, all three scored, none failed, none diverged:

| Muon lr | val_bpb |
|---|---|
| 0.01 | 1.091783 |
| **0.02** | **1.087067** |
| 0.04 | 1.087570 |

The winner beats the runner-up by **0.05%**, against a 0.5% noise floor. So the
rule this project pre-registered *before* the numbers landed applied: when the
grid discriminates, follow the measurement; when it does not, follow the
blueprint. `abl-arch` is training at **0.02**, which is both. `hero` takes
`--muon-lr` as a required flag rather than reading `best.json`, so this is a
decision recorded here, not a default that slid through.

**What this buys, honestly:** the knowledge that `hero` is insensitive to Muon
lr within a 2x band at this scale — which is worth knowing before spending $43
— rather than a sharp optimum. It is not the answer the $1.64 was spent
expecting, and a third sweep would buy nothing.

### 40B or 30B? — SUPERSEDED by the operator's 60B decision, kept as the record
<!-- appendix -->

> **This section was written before 2026-08-10, when the operator raised the
> budget to 60B and funded it from the underspend. It is retained because it is
> the evidence for *why* the epoch cap matters — which is exactly what makes the
> 60B top-up necessary — and because `go 40B` is still an option in the summary.
> Its closing recommendation is historical; it is not the current ask.**

The draft used to say "~2.9 epochs, under the 4.0 cap". That is true of the
corpus *in aggregate* and false of the sources that matter. `MixtureBatchSource`
samples at blueprint shares clamped per source to `4 × on_disk / run_tokens`, so
the binding constraint is per-source, and at 40B the two largest sources —
**fineweb-edu and dclm-baseline, 60% of the mixture between them** — sit at
**exactly 4.00 epochs**, right on the limit rather than comfortably inside it.

That is why the corpus was topped up to 14.2B overnight (a DNS blip had left
dclm at 1.57B of 2.25B; see `mixture-cap-vs-hero-budget.md`).

**Re-verified 2026-08-10 02:10Z against the finished corpus** — 14.218B on
disk, dclm complete at 2,250,000,677 — by running the real
`train.cap_weights_by_epochs` over the real per-source manifests, not by
re-deriving the earlier estimate:

| T | L1 skew | pinned at the 4.0-epoch cap |
|---|---|---|
| 20B | 3.98 pts | `everyday-conversations` only |
| 30B | 3.99 pts | `everyday-conversations` only |
| 40B | 3.99 pts | + **`fineweb-edu`, `dclm-baseline`** (both at exactly 4.00 epochs) |

Worth reading precisely, because the shape is the opposite of the intuition:
at 40B the two big sources land *exactly* on their targets (37.50% / 22.50%)
**because** they are pinned, and the leftover mass inflates the small sources
instead (stack-edu-python 9.47% vs its 9.00% target). At 30B nothing binds but
`everyday-conversations`, and fineweb-edu floats slightly *above* target
(38.26%) by absorbing the mass that source cannot take.

After the top-up:

| hero tokens | mixture L1 skew | sources at the 4.0 limit | wall clock | cost |
|---|---|---|---|---|
| 30B | 3.99 pts | everyday-conversations only | 68.6 h | **$30.80** |
| 40B | 3.99 pts | + fineweb-edu, dclm-baseline | 91.9 h | **$41.26** |

(Both re-priced 06:35Z from the measured 121,994 tok/s. They move down together,
so the ~$10.5 that separates the two options is unchanged and the recommendation
below is unaffected.)

**Two consequences worth having explicitly, added 07:55Z:**

**The mixture does not argue for 40B over 30B.** Skew is 3.99 pts at both, and
essentially all of it is the recorded `everyday-conversations` deviation. So
this decision is purely quality-per-dollar — picking 30B does not also cost you
data balance. That makes the cheaper option cleaner than the draft implied.

**But there is no third option above 40B, and that is the one to watch.** 40B is
not an arbitrary ceiling: the corpus was sized as `blueprint_share × 40B / 4`,
so it is exactly the largest budget the blueprint mixture survives. Past it the
cap binds hard and the web backbone collapses:

| | 40B | 50B |
|---|---|---|
| fineweb-edu | 37.5% | **30.0%** |
| dclm-baseline | 22.5% | **18.0%** |
| finephrase | 7.4% | **13.1%** |
| **L1 skew** | **3.99 pts** | **29.91 pts** |

`finephrase` (Table/FAQ/Tutorial) nearly doubles while the two best web sources
give up 12 points between them. **This matters because "it's going well, let's
extend it" is the natural request after a good `hero`** — it costs money to act
on, and the lever is a token budget that does not look like a data change. Above
~55B the cap gives up and warns loudly, so 50B is the quiet case, not the
extreme one.

`train.py` already logged the skew and the capped sources; what it did not do
was **grade** them — 3.99 pts and 29.91 pts printed the same shape of line. It
now warns above 10.0 pts, a threshold quiet at every budget the corpus was built
for and loud at the first one that breaks it.
→ `runs/preflight/mixture-vs-token-budget.md`

Both now hit the same mixture — the residual 3.99 is entirely
`everyday-conversations`, whose whole dataset is 0.4M tokens, and no amount of
data fixes it. So the choice is a straight **quality-per-dollar** question, not
a data-balance one:

- **40B** is 33% more tokens for $10.78 more. At 150M params that is 267
  tokens/param vs 200 — both far past Chinchilla-optimal (~20), so this is deep
  in diminishing returns, but the returns are still positive and the mission is
  benchmark quality, not efficiency.
- **30B** leaves fineweb-edu and dclm at 3.0 epochs instead of 4.0, i.e. real
  margin under the Muennighoff et al. (arXiv 2305.16264) "up to 4 epochs is
  ~free" result rather than sitting exactly on it. It also frees $10.78 against
  a balance with no slack.

**Recommendation as written at the time: 40B** — the repetition result is quoted
*at* 4 epochs, not below it, and the top-up bought exactly the headroom that
makes 40B legitimate. **The operator chose 60B instead**, funded from the
underspend, accepting ~4.3 epochs and a thin buffer (quoted here as ~$2.41 at the
time; **corrected to ~$11.14**, since refreshed to **~$10.95** — see the two
correction notes in the summary). That
decision stands and the ask above reflects it; this paragraph is kept so the
reasoning that was available at the time is not quietly rewritten after the fact.

### Measured throughput and projection

**Updated 2026-08-10 06:35Z — this is now a production measurement, not a
benchmark.** The earlier 115,692 tok/s came from `smoke.py`, a synthetic 5-step
run taken while `dataprep` competed for CPU. `abl-arch` is training the hero
architecture at the hero micro-batch and has now passed its ramp, so the number
below is the real thing at the exact shape `hero` runs at
(`runs/preflight/steady-state-throughput.md`):

| | |
|---|---|
| throughput | **121,994 tok/s** (hybrid, batch 16, seq 2048, batch 512k) |
| | sd **130 (0.11%)** across uninterrupted windows, quiet box |
| rate | $0.449/h |
| **58B tokens — the ask** | **133.3 h**, **$59.85**, **120,528 steps** (decay from 66,290) |
| 60B tokens — refuses on mixture skew; `go 60B anyway` overrides | 137.9 h, $61.89, 124,684 steps (decay from 68,576) |
| 51B tokens — the `go 51B` knee | 117.2 h, $52.61, 105,981 steps (decay from 58,289) |
| 40B tokens — the `go 40B` fallback | 91.9 h, $41.26, 83,123 steps |

Both rows are the same measurement scaled by tokens; the overheads below were
priced at 40B and scale with it. Step counts are not scaled — each is simulated
against `hero`'s exact config (`test_hero_step_estimate_is_exact`, which replays
the loop rather than re-deriving the estimator's own arithmetic).

**$1.86 cheaper than this draft previously asked for.** Priced from measurement
rather than allowance: the 1.435 GB checkpoint save costs **7.4 s** every 30 min
(visible at 06:13:26Z), a val pass **3.4 s** every 500 steps, and my own
10-minute check-ins ~**2.9 s** each — together 0.97 h of the 91.9. QAT adds
nothing (measured **−0.9%**, free within noise).

**Read the precision correctly.** That 0.11% was the first seven minutes; over 25
minutes and 16 uninterrupted windows the figure is **122,612 tok/s, sd 0.81%**,
range 121,665–124,275. The variation is the **hardware**, not the measurement —
this 5090 sits exactly on its 500 W cap (`power.draw 500.05 W / limit 500.00 W`,
throttle reason `0x4` = SW Power Cap) at 2745 of 3105 MHz, so window times land in
discrete clock modes.

Swept across that entire band, `hero` at 60B costs **$60.87 to $62.16** — a $1.29
spread on a $61.89 ask, with the quoted figure in the upper half (at the `go 40B`
fallback the same sweep is $40.58–$41.44 on $41.26). **Hero's cost is not
sensitive to the throughput number at the precision this argument needs.** It is
sensitive to whether the run completes without rework, which is a different and
much larger risk — and one a 25-minute sample cannot speak to, since it cannot
bound four days of thermal behaviour.

Read the *direction* honestly: the pre-registered prediction said steady state
would be "roughly flat, plausibly slightly up" versus the ramp. It came in
**−2.3%** (124,271 → 121,994). The predicted seq cost (~2%) was right; the
predicted batch-amortisation gain did not appear. The band (120–126k) held, so
the projection stands, but the next one should not assume that offset.

**30B on the same measured basis: 68.6 h / $30.80** (the "40B or 30B?" decision
above; both figures move down together, so the $10.46 gap between them is
essentially unchanged).
**The schedule at each budget you can reply with** (recomputed from
`train.estimate_total_steps` directly, not scaled from the 40B figures this
table used to carry):

| | 58B — the ask | 60B — `go 60B anyway` | 40B — the `go 40B` fallback |
|---|---|---|---|
| total steps | **120,528** | 124,684 | 83,123 |
| decay / milestone step | **66,290 = 55.000%** | 68,576 = 55.000% | 45,717 = 55.0% |
| QAT starts (final 5%) | step **114,996**, ≈**2.90B tokens** | step 118,962, ≈3B tokens | step 79,308, ≈2B tokens |
| lr multiplier at the last step | **1.84e-5** | 1.78e-5 | 2.7e-5 |

(`go 59.9B` is 124,476 steps, decay from 68,461; `go 51B` is 105,981 steps,
decay from 58,289. Every one of the five is replayed by
`test_hero_step_estimate_is_exact`.)

The schedule is pinned by tests against hero's exact config, so a ramp change
cannot silently move the decay point. Note the QAT window is **2.90B tokens at
58B, not 2B** — it is derived as a fraction, so nothing had to be changed for
that, but it is the number to quote.

**The QAT step numbers in that row were wrong until 2026-08-11 00:40Z, at every
budget** (118,449 rather than 118,962 at 60B; 78,967 rather than 79,308 at 40B).
They were computed as `0.95 x total_steps`, but QAT is *token*-derived —
`qat_active_at` is fed by `Trainer._progress`, which is `tokens_seen /
total_tokens` — and
the batch ramp makes the first 10% of steps consume well under 10% of the
tokens, so the token curve trails the step curve for the whole run. 95% of the
tokens therefore lands ~500 steps *later* than 95% of the steps. Nothing about
the run changes: the code was always right and the document was quoting an
approximation of it. But it is the number I would have checked the live run
against, and a QAT window that looked 500 steps late is exactly the kind of
non-finding that burns an hour on day five. Replayed and pinned by
`test_the_gate_quotes_the_qat_start_step_the_trainer_will_really_use`.

### Budget

| job | cost |
|---|---|
| spent by the time this gate opens | **~$27.87** (62.1 h wall clock — recompute at post time, see below) |
| `abl-arch` | `<<TODO: actual>>` (~$11.47 projected, included above) |
| corpus top-up for 60B (issue #5) | **$0 incremental** — ran beside `abl-arch` arm 2, so its hours are already inside the row above. Do not add it again |
| **`hero` at the 58B being asked for** | **$59.85** (from the 121,994 tok/s measurement at 06:35Z; `go 60B anyway` is $61.89, the `go 40B` fallback $41.26) |
| `post` + final eval + GGUF | ~$5.5 |
| **left afterwards** | **≈$30.73** at 58B · ≈$28.7 at 60B · ~$52.0 at 40B — **at $0.449/hr**; see *Projected credit* below (re-anchored 2026-08-11 07:55Z to your stated $99.00), which is the authoritative version of this row and gives the $0.668/hr column too |

**"Spent so far" must be recomputed when this issue is posted, not copied from
here.** An earlier draft of this table said **~$13.66**, taken from the
attributed row-totals in `COSTS.md`. That is the wrong basis, and `COSTS.md`
says so itself: several rows are marked *"shared wall-clock with dataprep — not
additive"*, so they price the **activity** while the vast balance loses **every
rented hour, attributed or not**. The reconciliation note in that file is
explicit — *"treat ~$14.6–15.3 as the real spend and the vast balance as ground
truth"* — and that read was already 4 h stale when this line was written.

Use wall-clock from this box's first RTX 5090 commit (`cf76ae7`,
2026-08-08 16:40Z) at $0.449/h, and state the clock time it was computed at:

```
spent ≈ (now − 2026-08-08T16:40Z) × $0.449/h
```

At 2026-08-10 07:59Z that is **39.3 h ≈ $17.65**, and projected forward to the
gate opening at 11th 06:45Z, **62.1 h ≈ $27.87**. It keeps climbing while the
gate waits for an answer, which is the honest thing for the operator to see —
an unanswered gate costs $10.78/day.

**A correction to my own arithmetic, made while filling this in.** I first
computed it from the repo's first commit (`b24cc6c`, 15:02Z) rather than this
box's (`cf76ae7`, 16:40Z). The repo was imported before the 5090 existed, so
that basis charges the box for 1.63 h it was not rented — **$0.73 overstated**.
Small, but it runs against the operator, and this file is what funding decisions
are made from. The methodology note above already named the right commit; I
simply did not use it.

### Projected credit at the end of `hero` — approve with this number visible

Your standing instruction is that this figure appears in the gate, so it is
here rather than assembled afterwards. Tracked forward from **your own last
statement of the balance**, because checking the account means calling `vastai`,
which the standing instructions forbid.

**Re-anchored to the launch moment (04:55Z).** This table previously priced
`abl-arch`'s *remaining* hours from a 00:56Z snapshot — "arm 2 at 57%" — which
would be read at 07:47Z, hours after arm 2 finished. Everything before launch is
now one line of rent, which is both simpler and immune to going stale: the
balance loses every rented hour whether it is attributed to a job or not, which
is the same basis `COSTS.md` uses.

**Re-anchored again at 07:55Z, to your $99.00 of 06:45Z.** The row that used to
head this table — *"balance you stated (2026-08-09 ~23:50Z) — $94.66"* — is
superseded by your own later statement, and carrying it here while
`runs/credit-anchor.json` held $99.00 was the defect described at the top of
this post. Both rates are now shown, because I cannot tell you which is real.

| | $0.449/hr (assumed) | $0.668/hr (measured) |
|---|---|---|
| balance you stated (2026-08-11 ~06:45Z, after the $25 top-up) | **$99.00** | **$99.00** |
| rent to this gate going up (~08:15Z, 1.5 h) | −$0.67 | −$1.00 |
| gate wait — 5.0 h allowance, **pure response time** (see below) | −$2.25 | −$3.34 |
| **balance at launch** | ≈ $96.08 | ≈ $94.66 |
| **`hero` at 58B** (133.3 h) | **−$59.85** | **−$89.04** |
| `post` + final eval + GGUF (12.25 h) | −$5.50 | −$8.18 |
| **projected at completion** | **≈ $30.73** (≈$28.7 at 60B · ≈$52.0 at 40B) | **≈ −$2.56 — short** |

At 51B (117.2 h) the same table ends at **≈$37.96** and **≈$8.19** — positive
under both, which is why it is the recommended reply if you do not want to read
the rate off the console.

The bottom line moved **+$0.06** on that rewrite, which is the point: the old
table reached ≈$12.62 through four rows that each had to be kept current, and
the new one reaches ≈$12.68 through one that cannot go stale. `scripts/credit_watch.py`
computes the same balance independently and live — $81.83 at 04:24Z against this
table's $81.82 for the same instant.

**Refreshed 21:55Z, and it barely moved (−$0.19).** Six hours of rent came off the
balance (−$3.05) and were almost exactly offset by `abl-arch` getting cheaper as
arm 2 progressed (+$2.86) and the top-up's charge disappearing (+$2.25). I first
wrote this refresh as *~$2 better* by dropping the 5.0 h response-time allowance
and crediting the difference to the top-up — that deleted a real cost rather than
a double-count, and is corrected above.

**The two things that eat that ~$11, stated plainly.**

1. **Your response time is the largest uncontrolled term — and it now converts
   directly into money, which it did not before.** That line used to be floored
   by the top-up's own duration: answering instantly bought nothing, because
   `hero` could not start until the corpus was built. The top-up now finishes
   **~22:40–00:35Z**, roughly **8 h before this gate opens**, so at gate-open
   everything is ready and `hero` starts the moment you say go. The 5.0 h is a
   plain allowance, not a floor. Every extra 10 h of waiting is **$4.49**, so a
   two-day pause consumes the buffer on its own before `hero` starts — and
   conversely, a fast answer now genuinely saves what a fast answer previously
   could not.
2. **It assumes `hero` completes without rework.** ~27% of this project's spend
   has gone to failed attempts. A restarted `hero` does not fit in ~$11.

If credit looks like running out mid-run I will say so in `STATUS.md` and open
an issue rather than let it stop silently — the checkpoints are on the Hub, so a
top-up recovers the run, but only if you hear about it in time.

**A correction to how this was previously framed.** `COSTS.md` carried "whole
project ≈ $92.25 against $94.66 — about $2.40 of headroom". That total was built
from the *attributed* row-by-row spend, which the same file tells the reader not
to use for this purpose; on the wall-clock basis it recommends, the identical
arithmetic reads ~$2.5 **over** rather than $2.40 under. Neither is the right
question — the balance projected forward is, and that is the table above. Both
files now carry it.

### The four hard preconditions — all met, with evidence

Per the standing instruction, `hero` does not launch until all four hold.

**A fifth is now in place too** — restart-resume, in `STATUS.md`.

1. **Weights-only uploads to a Hub model repo on a ~2 h cadence, out-of-band.**
   `Unseen1980/daedalus-checkpoints` (private). Measured **321.0 MB** bf16.
   Transfers run in a separate torch-free process, so a hung link costs no
   training time. Proven against the real Hub, not a mock: 1.76 GB in 58.8 s
   *while dataprep streamed* — and then proven again **on `abl-arch` itself**
   at 05:18:32Z, which is the point of the precondition: the step-1 checkpoint
   staged, the uploader took it on its first poll, and 321.0 MB landed as
   `rolling/abl-arch-daedalus-150m/weights.pt` (verified by listing the repo,
   not by trusting local bookkeeping) while training held 124,493 tok/s.
   **The second upload then landed at 07:15Z on the 2 h cadence** (step 2499,
   862M tokens, 321.0 MB, again verified by listing the repo), with throughput
   across the transfer window at **124,312 tok/s**. One upload proves the path;
   two prove the *schedule*, which is what a 92 h run actually depends on.

   **And now on a second, independent run.** Arm 2 has been uploading all night
   on the same cadence — `rolling/abl-arch-dense-150m/weights.pt`, 322.6 MB, last
   landed at **00:59Z** (step 6,047), with `hub_pending` **0** and the trainer at
   109,004 tok/s. Re-verified at 02:05Z by listing the live repo rather than
   reading local bookkeeping: the `rolling` branch carries weights for **both**
   arms. Arm 1 proved the path on the code as it stood this morning; arm 2 is
   proving it on the code `hero` will actually run, including this evening's
   refuse-to-persist-a-NaN-model change.

   **And it is now watched rather than assumed** (04:40Z) — see the third hole
   below.
2. **Milestone with full optimizer state at the decay-start step, on its own
   revision the rolling copy cannot overwrite.** **Fired for real on `abl-arch`
   arm 1 at 11:01:37Z**, at step **5,715** — the step predicted in writing
   beforehand — with `lr_mult_at_branch` **1.000000** and both optimizer states
   present, on revision `abl-arch-daedalus-150m-stable-end-step5715`. Measured
   **1435.3 MB**. Verified by `scripts/check_milestone.py` **against the live
   Hub**, which recomputes the step from the record's own `total_steps`/
   `decay_frac` rather than trusting it. `decay_start_step()` is shared with
   `wsd_lr`, so schedule and artifact cannot drift.
   Tests: `test_milestone_step_is_exactly_the_wsd_decay_start`,
   `test_trainer_milestone_step_tracks_decay_frac`,
   `test_milestone_written_at_decay_start_with_optimizer_state`.

   **Fired a second time, on arm 2, at 00:09Z on the 11th** — same step 5,715
   (the two arms share a schedule), 1416.2 MB, on its own revision
   `abl-arch-dense-150m-stable-end-step5715`. Confirmed present on the live Hub
   at 02:05Z. So the branch-point mechanism has now produced two milestones on
   two configs without either overwriting the other or the rolling copy, which
   is the specific property `hero` needs it for.

   **It did not land on the first attempt, and that is the part worth reading.**
   The uploader wedged mid-transfer — socket CLOSE-WAIT, 0% CPU, no bytes on the
   wire for 20 minutes, SIGTERM ignored, reproduced on restart. `upload_once`
   catches every exception and `watch()` wraps the pass in a belt-and-braces
   `except`; **a hang is not an exception**, so nothing retried and nothing
   logged, and the rolling checkpoint queued behind it never went either. On a
   92 h `hero` that is checkpoint durability silently ending for the rest of the
   run. Each pass now runs in a child process under a 900 s deadline
   (`subprocess.run` SIGKILLs it, which is what returns the socket, fd and
   memory); the fixed uploader landed the milestone on its **first bounded
   attempt**. `train.py`'s end-of-run drain is bounded for the same reason — it
   is synchronous, so the same hang stalls the trainer itself. The trainer also
   now *grades* `hub_stale_h` instead of only recording it.
3. **Restore-from-Hub tested end to end.** Test name:
   **`test_restore_from_hub_end_to_end`** (`tests/test_train.py`), plus a
   live restore into a clean directory at the exact step, worst relative weight
   delta 0.0039 (bf16 precision), training continued, milestone came back with
   its Muon momentum buffers at lr multiplier 1.0.
4. **Repo name recorded in `STATUS.md` and the model card, with the branch
   command.** `STATUS.md` had it; the model card did not exist until
   2026-08-10 — `export.py` wrote weights and GGUFs and no `README.md` at all.
   `export.render_model_card` now writes one on every export, carrying the
   success bar, the architecture, the Q4_0 delta, the checkpoint repo and the
   runnable branch command built from `milestone.json`
   (`test_model_card_publishes_the_branch_command_from_the_milestone`).
   Once `hero` passes 55%:

```bash
/venv/main/bin/python train.py --run-name hero-ext --config daedalus-150m \
  --data-dir data/shards --total-tokens <new budget> \
  --resume 'hub://Unseen1980/daedalus-checkpoints/milestone/hero/checkpoint.pt?rev=hero-stable-end-step<N>'
```

**At the 58B being asked for, `<N>` is `66,290`** — so the revision to look for
is `hero-stable-end-step66290`, at 55.000% of 120,528 steps, and on a 07:52Z
launch it should appear around **2026-08-14 03:48Z** (67.9 h in).

**The checkpoint there will have seen 29.56B tokens — 50.97% of the budget, not
55%.** Worth stating explicitly because the natural reading of "the branch point
is at 55% of the run" gives `0.55 x 58B = 31.9B`, which is 2.3B too high; at the
60B this was decided against it is 30.58B rather than the ~33B the decision note
assumed. The gap is the batch ramp again — the first 10% of the budget is spent
climbing 128k → 512k tokens per step, so early steps are cheap and the token
curve trails the step curve by a constant ~4 points from then on (50.97% at
every budget: 58B, 60B and 40B alike). Nothing is wrong: WSD decays over the
final 45% of *optimizer steps*, which is what `decay_start_step` implements and
what the milestone correctly follows. But "how many tokens seen" is one of the
four things your precondition says to record about the branch point, and it is
not `0.55 x budget`.

`<N>` stays a placeholder in the
command because it follows the budget (`go 51B` would make it 58,289; `go 60B
anyway`, 68,576), but naming the expected value turns "did the milestone fire
correctly?" into something answerable on day four without recomputing anything.
Arm 1 is the precedent: it fired at step 5,715, exactly the step predicted in
writing, which is why precondition 2 is evidence rather than an assurance.
Pinned to the ask by `test_the_gate_names_the_milestone_step_for_the_budget_it_asks_for`.

### A fifth gate, added 19:25Z: `hero` now refuses a corpus that cannot deliver the mixture
<!-- appendix -->

Not one of your four — added because checking whether the running top-up could
be cut short turned up something worse: **`hero.py` gated on the mixture not at
all.** `train.py` warns and trains anyway, which is right for `sweep` and
`abl-arch` and wrong here, where one line at step 0 of 124,684 is
indistinguishable from silence on a phone.

Two regimes would have launched today, and each is invisible to the metric that
catches the other:

| corpus | what `hero` would have done | the graded metric said |
|---|---|---|
| **14.683B** (now) | repeated `everyday-conversations` **2,973×** as 2% of the run | `l1_skew_pts` **0.00 — its best possible value** |
| **15.0–17.3B** (mid-top-up) | trained `fineweb-edu` at **28.4%** against a 37.5% target | `max_epochs_seen` **4.00× — perfectly capped** |

`hero.check_mixture()` now checks both and `_cli` refuses with **rc 2** before
the watchdog or the trainer starts, with the reason in `STATUS.md`.
`--allow-skewed-mixture` overrides and records that choice, so a knowingly-bad
run stays traceable in the writeup. The preflight reads one `manifest.json` per
source — no GPU, no memmaps — and shares the loader's code path rather than
reimplementing it, because a preflight that computes the mixture differently
from the thing that samples it gates on one answer and trains on another.

Tests: `test_check_mixture_catches_unbounded_repetition`,
`test_check_mixture_catches_a_bounded_but_skewed_mixture`,
`test_check_mixture_passes_a_corpus_that_can_deliver_the_mixture`,
`test_hero_refuses_to_launch_and_starts_nothing`,
`test_allow_skewed_mixture_launches_and_records_the_choice`,
`test_the_preflight_agrees_with_the_loader_it_is_standing_in_for`,
`test_the_gate_reads_the_train_split_not_the_whole_corpus`,
`test_hero_gates_at_its_own_default_budget_not_a_test_sized_one`,
`test_the_top_up_target_clears_the_gate_with_the_holdout_carved_out`.

**One number to have in hand before you answer.** The gate reads the *train
split*, after the 2% holdout carve, because that is what `train.py` samples. At
the top-up's 17.717B target the split is 17.362B and skew is **7.19** against
the 10.0 limit — it passes, but bisecting shows the split must retain **≥96.2%**
of target while the carve takes 2.0%. So **the top-up may finish ~3.8% short and
no further.** It is on track to finish in full at ~00:30Z, well ahead of this
gate; if it is ever interrupted, `hero` will refuse rather than quietly train on
a skewed corpus. Full curve and reasoning: `runs/preflight/topup-early-stop.md`.

### The speed half of the bar now has a number: 2.08× SmolLM2-135M

Measured 2026-08-10, both models Q4_0 with llama.cpp defaults, alternating
rounds at 8 threads (`runs/eval/decode-vs-smollm2.md`):

| decode at depth | Daedalus-150M | SmolLM2-135M | ratio |
|---|---|---|---|
| 0 | 960.9 ± 3.4 | 908.1 ± 37.1 | 1.06× |
| 512 | 933.7 ± 28.5 | 625.7 ± 38.5 | 1.49× |
| 2048 | 648.6 ± 12.6 | 312.4 ± 7.2 | **2.08×** |

This matters for the gate because it is the half of the success definition that
does **not** depend on `hero` succeeding: decode speed is a function of shapes
and quantization, not of how well the weights were trained. So "beat
SmolLM2-135M decisively on CPU decode" is already true and already measured,
whatever `hero` does to the quality column — and it is 2× rather than the 1.06×
a default `llama-bench` run reports, because only 6 of our 18 blocks keep a KV
cache against all 30 of theirs.

**The same correction applies to the ablation's own headline, and it was found
in time.** `abl-arch`'s hybrid-vs-dense decode number was going to be measured
at depth 0 — `export.measure_decode_speed` ran a bare `llama-bench` with no
`-d`. Re-measured on the dry-run GGUFs at matched threads and alternating
rounds (`runs/eval/decode-hybrid-vs-dense.md`):

| depth | `daedalus-150m` | `dense-150m` | ratio |
|---|---|---|---|
| 0 | 951.9 ± 37.8 | 825.3 ± 5.0 | 1.15× |
| 512 | 869.5 ± 22.9 | 593.6 ± 29.6 | 1.46× |
| **2048** | **648.5 ± 7.0** | **354.1 ± 33.8** | **1.83×** |

So the architecture claim is **1.83×** against a param-matched dense twin, not
the 1.15× this project has been quoting — the mechanism is exactly 2× the KV
bytes per decoded token (6 of 18 blocks at 4 KV heads vs 24 of 24 at 2), and it
only shows up once there is a context to re-read. Fixed before arm 1 exports at
~17:13Z, so **both arms report all three depths on trained weights** and the
number in this gate is measured rather than corrected afterwards.

### The lr decision, decided before the number arrives
<!-- appendix -->

Probe 1 has scored: **val_bpb 1.0918 at muon_lr 0.01**. Probes 2 and 3 follow.
Writing the branches down now, before knowing which wins, so the choice is not
made to fit whichever number turns up:

Every branch below now means *wins by more than the 0.5% noise floor, measured
against the runner-up*. That qualifier is no longer prose: `resolve_muon_lr`
enforces it, so a winner inside the floor is ignored in favour of the
blueprint's 0.02 (see "the tie" branch below) — for `abl-arch` automatically,
and for `hero`, which takes `--muon-lr` explicitly, by this decision.

- **0.02 wins** → use it. Agrees with the blueprint; nothing to discuss.
- **0.01 wins** → use it. Interior to the grid, and lower is the safe direction
  when the probe horizon is 120× shorter than the run it is choosing for.
- **0.04 wins by more than the noise floor** → say so and do *not* silently
  adopt it. It sits at the grid edge, and a 0.5B-token probe systematically
  favours a higher lr than a 60B run would: the short horizon rewards fast
  early progress, which is exactly what a too-high lr buys and then pays back.
  `scripts/check_sweep.py` already warns here (and only when the grid spread
  exceeds its noise fraction, so an edge winner on a flat grid is not treated
  as evidence). The options are then: take 0.04 as measured; take the
  blueprint's 0.02 as the long-horizon-safe choice; or spend ~$0.55 on a fourth
  probe at 0.08 to find out whether the optimum really lies outside the grid.
  My recommendation in that case would be **0.02 for `hero` and the fourth
  probe only if the operator wants the question settled** — being wrong high
  costs a divergence and a rollback, being wrong low costs a little
  convergence, and `hero` is 96 h with no budget for the former.

- **Nothing wins — the probes tie.** This is now the *likeliest* outcome and it
  was not in the list above, so it is written in before the number lands.
  Comparing probes 1 and 2 at matched steps, 0.02 leads early (it is the higher
  lr, so it makes faster early progress), 0.01 crosses ahead at step ~500, and
  the two then **converge through the decay phase**:

  | step | lr 0.01 | lr 0.02 | delta |
  |---|---|---|---|
  | 20 | 11.1862 | 9.3001 | −1.8860 |
  | 200 | 5.1863 | 5.1648 | −0.0214 |
  | 500 | 3.9175 | 3.9182 | +0.0007 |
  | 620 | 3.6973 | 3.7067 | +0.0093 |
  | 800 | 3.5580 | 3.5609 | **+0.0028** |

  At step 800 they differ by **0.0028 nats, about 0.08%** — an order of
  magnitude inside `check_sweep.py`'s 0.5% noise fraction, and the gap is
  *narrowing*, not opening. If the final val_bpb spread lands under that floor,
  the gate will say so itself ("the lr choice is close to arbitrary at this
  budget"), and the honest report is **"the sweep could not tell 0.01 and 0.02
  apart", not "0.0x won"**.

  The rule I will follow, stated before the data rather than after: **when the
  grid discriminates, follow the measurement; when it does not, follow the
  blueprint** — i.e. take **0.02**, because a tie is not evidence for deviating
  from a locked design decision. What this must *not* become is a decisive
  headline: a 0.08% separation over a 0.5B-token probe says nothing about which
  lr is better over 60B, and the two branches above ("use it; nothing to
  discuss") would have over-read exactly that.

`abl-arch` is unaffected either way: both arms share whatever lr the sweep
picks, so its internal validity does not depend on that lr being hero's.

### Before this gate opens: a measured quality point, not just a speed one

As drafted, this gate would ask for **$61.89 — the 60B it asked for at the time,
now $59.85 at 58B — and nearly six days with no evidence that the quality half of
the bar is reachable.** The speed half is measured
(2.08× at 2048). The quality half — clearing **42.2**, the best 300B-class peer
on our own harness — rests on nothing but the plan.

`abl-arch` fixes that for free: it leaves a real 5B-token checkpoint of the hero
architecture on disk (`entry["checkpoint"]`, `abl_arch.py:350`, not deleted).
Scored on the same 5 tasks, at full splits, through the same harness that
produced the peer table, that is a **directly comparable** number. Two points
beat one, so `sweep` probe 1's 0.5B checkpoint gets scored too — 0.5B and 5B
give a slope to extrapolate toward 60B instead of a single dot.

Both commands already exist and **were exercised end to end at 03:55Z** (the
`--ours` path fed a peer JSON of identical shape, which appended a correct
5-task-mean row), so this is ~40 min of GPU and ~$0.30, not new code:

```
/venv/main/bin/python eval.py --config daedalus-150m --out runs/eval/ours-5B.json \
  --checkpoints runs/abl-arch-daedalus-150m/checkpoint.pt
/venv/main/bin/python scripts/peer_table.py --ours runs/eval/ours-5B.json
```

**Its main value is as a bug detector, not as a forecast.** A 150M model at 5B
tokens will sit low, and HellaSwag in particular may be near its 25% chance
floor, so the extrapolation to 60B is weak and will be reported as weak. But if
*every* task comes back at chance, something is wrong — tokenizer, packing, or
the eval itself — and that is worth finding for $0.30 rather than for $61.89.
Whatever it says, it goes in this issue **before** the operator is asked to
approve, including if it is discouraging.

#### RESULT — the 0.5B point landed at 06:02Z, and the pipeline is not broken
<!-- appendix -->

`sweep` probe 2 (Muon lr 0.02, 0.5B tokens, fully decayed, the architecture and
tokenizer `hero` will use), scored on full splits through the harness that
produced the peer table:

| | HellaSwag | ARC-Easy | PIQA | OpenBookQA | WinoGrande | **mean** |
|---|---|---|---|---|---|---|
| chance | 25.0 | 25.0 | 50.0 | 25.0 | 50.0 | **35.0** |
| **Daedalus @0.5B** | **27.3** | **38.7** | **56.0** | **28.4** | **50.7** | **40.2** |
| GPT-2 124M (budget undisclosed) | 31.2 | 40.1 | 62.1 | 27.6 | 49.8 | **42.2** |
| above chance | +2.3 | **+13.7** | **+6.0** | +3.4 | +0.7 | **+5.2** |

**The bug-detection question is answered: pass.** The failure mode this run
existed to catch was all five tasks at 35.0, which is what a tokenizer mismatch,
document-crossing packing or a broken eval formatter produces. Instead the three
tasks that discriminate at this scale all move — ARC-Easy most of all — and the
two that never discriminate sit flat, exactly where the 300B peers also sit.
That pattern is very hard to produce by accident.

**ARC-Easy at +13.7 is the one surprise, and the peer data says not to bank
it.** A 0.5B-token model is 1.4 points off GPT-2 there. Checking the raw
accuracies already on disk shows why, and it is not the flattering explanation:

| model | tokens | ARC-Easy acc | acc_norm |
|---|---|---|---|
| **ours** | **0.5B** | **43.6** | **38.7** |
| GPT-2 124M | ~30B | 44.2 | 40.1 |
| GPT-neo-125M | 300B | 43.7 | 39.4 |
| OPT-125M | 300B | 43.5 | 39.9 |
| Pythia-160M | 300B | 37.5 | 37.5 |
| SmolLM2-135M | 2T | 64.4 | 58.8 |

**Across 0.5B → 300B — a 600× increase in tokens — ARC-Easy moves by nothing
that correlates with tokens.** The 6.7-point spread in that range is ordered by
data and recipe (Pythia last, at 300B), not by budget. Only SmolLM2's 2T breaks
out of the cluster. Our educational mixture (fineweb-edu, finemath, infiwebmath,
cosmopedia) plausibly explains sitting at the *top* of that cluster rather than
at Pythia's end — but it is not evidence of a trajectory.

The consequence for this gate is the conservative one: **do not count ARC-Easy's
+1.4 as a gain more tokens will buy.** It shifts that burden onto HellaSwag and
PIQA, which is where the decomposition below puts it anyway. This corrects an
earlier read of the same number as "corpus fit, ahead of schedule" — the peer
column shows it is better explained as a task that saturates early at this
scale.

#### What the 0.5B point does and does not say about 42.2
<!-- appendix -->

It does **not** say "72% of the way to the bar on 1.25% of the budget, so the
rest is easy". Early gains on these benchmarks are the cheap ones. The useful
version is the per-task gap to GPT-2, which *is* the bar:

| task | @0.5B | needed for 42.2 | comment |
|---|---|---|---|
| HellaSwag | 27.3 | **+3.9** | near floor; the task that scales hardest with tokens |
| PIQA | 56.0 | **+6.1** | the other real climb |
| ARC-Easy | 38.7 | +1.4 | **assume 0** — flat across 0.5B→300B (above) |
| OpenBookQA | 28.4 | −0.8 | already ahead (both at chance) |
| WinoGrande | 50.7 | −0.9 | already ahead (both at chance) |

The five gaps sum to **+9.7 points**, i.e. the +1.94 of mean that separates us
from the bar. Taking the conservative line on ARC-Easy, essentially **all 9.7
must come from HellaSwag and PIQA** across a **120×** increase in tokens, with the
other three contributing nothing.

A sanity check on whether that is a lot: interpolating log-linearly between our
0.5B point and SmolLM2-135M's 2T point (the only two anchors that exist on this
harness) gives ~1.3 HellaSwag points and ~1.1 PIQA points per doubling, and
**0.5B→60B is 6.9 doublings** — implying **+9.0 HellaSwag and +7.6 PIQA = +16.6
summed**, against the **+9.7 required** with ARC-Easy assumed flat.

(At the `go 40B` fallback the same arithmetic is 6.3 doublings, +8.4 and +6.7 =
**+15.1** — still above the +9.7, which is why the 60B decision buys margin in
this argument rather than making or breaking it.)

**Treat that as suggestive and nothing more.** It is a two-point extrapolation
across 80×, one endpoint of which is a different model trained on different data
with a different recipe; the true curve is sigmoid, so a chord drawn from a
near-floor point can mislead in either direction. It is not evidence that the
bar will be cleared. It is evidence that the required gain is *within the range
that token scaling normally delivers* rather than outside it — which is the most
that can honestly be claimed before the 5B point lands.

**The 5B point from `abl-arch` is the real second anchor**, because it is on
*our* curve rather than someone else's, and it arrives tonight for ~$0.30. Two
points on our own trajectory turn the above from an analogy into a measured
slope. It goes in this issue before the operator answers.

#### What the numbers mean — the chance floor, which is not zero
<!-- appendix -->

Nothing in this repo has written down what a *dead* model scores on this
metric, and without it the 5-task mean is easy to misread. The tasks are
multiple choice, so chance is per-task:

| task | choices | chance |
|---|---|---|
| HellaSwag | 4 | 25.0 |
| ARC-Easy | 4 | 25.0 |
| PIQA | 2 | 50.0 |
| OpenBookQA | 4 | 25.0 |
| WinoGrande | 2 | 50.0 |
| **5-task mean** | | **35.0** |

So the usable scale runs **35.0 (chance) → 51.2 (SmolLM2-135M, 2T tokens)**, and
the bar to beat every 300B-class peer, **42.2**, sits **44% of the way up it**,
not 82% as the raw number suggests.

Two consequences worth stating before any score is read:

- **The peers are themselves at chance on two of the five tasks.** OpenBookQA
  25.0–27.8 against a 25.0 floor, WinoGrande 49.4–51.5 against 50.0. Pythia-160M
  is *exactly* at chance on OpenBookQA. Almost all the real separation between
  35.0 and 42.2 comes from **HellaSwag, ARC-Easy and PIQA** — those three are
  where this project is won or lost, and a gain on the other two is likely noise.
- **A single number near 35 does not distinguish "undertrained" from "broken".**
  That is precisely why the 0.5B and 5B points are being taken: a model that is
  merely undertrained still shows *movement* on the three discriminating tasks,
  and one that is broken sits flat at 35.0 on all five at both checkpoints.

#### Is 42.2 actually reachable at 60B tokens? — the crux of this $61.89
<!-- appendix -->

This gate has never answered the question it is really asking the operator to
fund. The peers we must beat trained on **300B tokens; `hero` trains on 60B**,
5× fewer. Stated that baldly it sounds hopeless. Our own measured peer table
says otherwise, and the argument needs no external citation — it is entirely in
`runs/eval/peer-table.md`:

| model | tokens | our 5-task mean |
|---|---|---|
| GPT-2 124M | **undisclosed** (~10–30B?) | **42.2** |
| OPT-125M | **180B** | 42.1 |
| GPT-neo-125M | 300B | 41.9 |
| Pythia-160M | 300B | 41.0 |
| SmolLM2-135M | 2T | 51.2 |

**Two of those budgets were checked against primary sources on 2026-08-10 and
were wrong** (`runs/eval/peer-token-budgets.md`). OPT-125M was listed at 300B; the
OPT corpus is **180B / 800 GB**, and 300B appears to have been assumed from the
neighbouring Pile-era models. And GPT-2's "~30B", which this draft previously
leaned on hardest, is not an estimate but an **absence** — OpenAI never published
it (*"The training duration was not disclosed"*). One epoch of WebText is ~10B
tokens; the epoch count is unpublished.

**So the argument is now made without GPT-2 carrying it.** Read OPT and Pythia
together: **180B scores 42.1, 300B scores 41.0.** A 1.7× token difference, in the
wrong direction, between two models of the same era and size. Across the whole
180B–300B span the spread is 41.0 to 42.2 and token count explains essentially
none of it. What moves the number is data quality and recipe — exactly what
separates the 2T SmolLM2 (curated modern web: FineWeb-Edu, DCLM, the same lineage
as our corpus) from the Pile-era models.

GPT-2 is still the top row and still the bar, and its budget is very likely the
*smallest* here — which strengthens the point rather than weakening it. It is
simply no longer the row the argument stands on, because a claim resting on an
undisclosed number is not one to ask $61.89 for.

That reframes the ask. `hero` is not "60B against their 300B"; it is **60B of
2024-25-grade curated data against 300B of 2020-21-grade data**, at a slightly
larger parameter count (160.5M vs 125–160M), with a modern recipe (Muon, WSD,
QK-norm, QAT). GPT-2 is the closest analogue in the table — comparable token
budget, comparable size — and it sits exactly on the bar.

**Honest verdict: plausibly reachable, not assured.** The evidence supports it,
but nobody should read the above as a prediction of success:

- ~~GPT-2's "~30B" is an estimate of WebText's size, not a published token
  count. It is the load-bearing row and it is the softest one.~~ **Checked
  2026-08-10 and it is worse than "soft": the number does not exist.** OpenAI
  disclosed no training duration for GPT-2. The argument above has been rebuilt
  on OPT-125M (180B, sourced) vs Pythia-160M (300B, sourced) so that it no longer
  depends on it — see `runs/eval/peer-token-budgets.md`. The bar itself is
  unaffected: 42.2 is a measurement on our harness, not a budget claim.
- The peers cluster in a 1.2-point band, so **the difference between beating all
  of them and beating none is ~1 point** — well inside the range that data
  mixture, an lr the sweep could not resolve, or a single-seed draw can move.
- **That ~1 point is also inside the benchmark's own sampling noise, which
  nothing here had ever measured.** Computed 2026-08-10 (`scripts/eval_noise.py`):
  the 5-task mean carries **±0.59 points** of pure binomial error — OpenBookQA
  is 500 items, so one question is 0.2 points of that column alone — and a
  difference must reach **1.65 points** to be two sigmas. Against the named
  peers that means a **2σ win needs 42.65 (Pythia), 43.55 (GPT-neo), 43.75
  (OPT)**. So a 42.5 finish would honestly be a *tie* with this peer group, not
  a win. It does not change what to run; it changes what may be claimed
  afterwards, and you should have it before spending rather than after.
  Two things it is not: it is sampling error only (seed variance is reported at
  2–3 points at this scale and is unmeasured here, so the true uncertainty is
  larger), and it is *unpaired*, which overstates the error of a difference
  because every model answers the same items. Removing that second one costs
  ~$1 of otherwise-idle GPU during this gate's wait — queued, see below.
  One encouraging read from the same arithmetic: our 0.5B probe is **1.0σ**
  behind Pythia-160M, i.e. already statistically indistinguishable from a
  300B-token peer at 1.25% of the budget.
- ~~Nothing here is measured on *our* weights yet.~~ **Updated 06:02Z: the 0.5B
  point is now measured — 40.2, above the 35.0 floor, with the discriminating
  tasks moving.** It rules out a broken pipeline, which was its job. It does not
  establish the bar is reachable, and the remaining climb is concentrated in
  exactly two tasks (HellaSwag, PIQA). The 5B point lands tonight.
- ~~It does not establish the bar is reachable.~~ **Updated 16:47Z: the 5B point
  clears the bar.** See immediately below. This is the single largest change to
  this gate since it was drafted.

#### RESULT — the 5B point landed at 16:47Z, and it clears the bar

`abl-arch` arm 1: the **hero architecture**, 4,994,316,288 tokens, WSD fully
decayed (final lr **0.257% of peak**, so a finished model rather than a snapshot),
QAT deliberately off for the ablation (`max(qat_active)` over all 519 metric rows
is 0, so this is clean fp32). Scored through the same harness, same full splits
and same metrics as the peer table:

| | HellaSwag | ARC-Easy | PIQA | OpenBookQA | WinoGrande | **mean** | tokens |
|---|---|---|---|---|---|---|---|
| chance | 25.0 | 25.0 | 50.0 | 25.0 | 50.0 | **35.0** | — |
| Daedalus @0.5B | 27.3 | 38.7 | 56.0 | 28.4 | 50.7 | **40.2** | 0.5B |
| GPT-2 124M — **the bar** | 31.2 | 40.1 | 62.1 | 27.6 | 49.8 | **42.2** | *undisclosed* |
| **Daedalus @5B** | **33.1** | **47.0** | **62.3** | **31.0** | **50.0** | **44.68** | **5B** |
| SmolLM2-135M | 43.2 | 58.8 | 68.7 | 33.0 | 52.2 | **51.2** | 2T |

**It clears 42.2 by 2.5 points, and it wins all five tasks individually against
GPT-2 124M** — on 5B tokens against peers trained on 180–300B. ARC-Easy is the
standout (+6.9 over the bar). The two tasks flagged above as the concentrated
remaining climb both moved: HellaSwag 27.3 → 33.1, PIQA 56.0 → 62.3.

**What this changes about the decision.** The gate was drafted asking for $61.89
on a quality argument that rested on a 0.5B point sitting **2 points below** the
bar. It now rests on a 5B point sitting **2.5 points above** it, produced by the
exact architecture, tokenizer, data mixture and schedule `hero` will use. The
verdict moves from *"plausibly reachable, not assured"* to **"materially
de-risked, still not assured"**.

**What it does not license, stated plainly.** It is not a forecast of the `hero`
number, and this document deliberately contains no projected 60B score:

- Two points on a token-scaling curve are a weak basis for a third. The 0.5B→5B
  slope is +4.46 per decade; taken literally over the further 1.08 decades to 60B
  it lands near 49.5, which would sit between the bar and SmolLM2-135M. **I do not
  claim that number.** Scaling curves flatten, and the two points come from runs
  that differ in more than token count (`sweep` probe vs `abl-arch` arm).
- Seed variance at this scale is reported at 2–3 points and remains unmeasured
  here (the second hero seed was deferred by operator decision). A 2.5-point
  margin is inside that band, so "clears the bar" is a single-seed statement.
- The peer error bars still apply: ±0.58, and the four 300B-class peers remain
  mutually within ~1σ of each other.
- **Decontamination is partial, and this is the caveat I would most want checked
  if I were reading this number sceptically.** The corpus was built with
  8/13-gram decontam against all five tasks (`data/manifest.json` records it),
  but `_build_eval_index` takes `limit=2000` items *per task*
  (`daedalus/dataprep.py:299,304`), so coverage is uneven:

  | task | items scored | items in the decontam index | coverage |
  |---|---:|---:|---:|
  | HellaSwag | 10,042 | 2,000 | **19.9%** |
  | ARC-Easy | 2,376 | 2,000 | 84.2% |
  | PIQA | 1,838 | 1,838 | 100% |
  | OpenBookQA | 500 | 500 | 100% |
  | WinoGrande | 1,267 | 1,267 | 100% |

  Documents matching an *uncovered* eval item were not filtered, so contamination
  could inflate the number, mostly via HellaSwag. Two things argue it is not what
  is driving this result: the largest margin over the bar is **ARC-Easy (+6.9),
  which is 84% covered**, while the worst-covered task (HellaSwag, +1.9) is our
  *smallest* win — the opposite of the pattern contamination would produce; and
  the peers being compared against (GPT-2, Pythia, OPT, GPT-neo) were trained on
  web corpora with no decontamination at all, so the comparison is not tilted in
  our favour by this. It remains an honest limitation of every Daedalus eval
  number, `hero`'s included, and it is cheap to close for the remaining corpus:
  the top-up can pass a higher `--eval-task-limit`, which would fully cover the
  3.50B new tokens (~25% of the final corpus). Recorded rather than silently
  fixed, because raising the limit enlarges the n-gram set every dataprep worker
  holds, and ADDENDUM 2 says measure that before changing it under a live run.

The honest summary is that the risk being bought has changed shape. Before, the
open question was whether this architecture and recipe could reach 42.2 at all.
It reaches it at one twelfth of the budget. What 60B buys on top is now the
uncertain part — and that is a much better bet than the one this gate was
originally going to ask you to take.

If the answer had been "no plan within this budget beats any credible peer",
that belonged at the top of this issue rather than buried here. It is not the
answer — and as of 16:47Z it is further from being the answer than when this was
drafted. The margin is still thin enough that this is a genuine bet rather than a
formality, but it is no longer a bet on an unmeasured premise.

### What stops a four-day run going wrong quietly
<!-- appendix -->

Durability answers "the box vanished". This answers "the run kept going after
it stopped being worth anything", which was the likelier failure and, until
2026-08-10 03:00Z, an unhandled one.

`watchdog.py` halted a diverged run by SIGTERMing the trainer and then exiting.
`run_with_resume` read that non-zero exit as a crash and restarted from the
**diverged** checkpoint — with no watchdog left running. A divergence at hour 20
would have trained a broken model for the remaining three days and exited 0,
reporting "finished in 2 attempt(s)". Now:

- the watchdog writes `runs/hero/watchdog-halt.json` on divergence and stall
  (never on a crash — restarting those is the supervisor's job), and
  `run_with_resume` stops on it before the backoff sleep and reports it as a
  halt. Tests: `test_a_watchdog_halt_is_not_resumed`,
  `test_a_crash_without_a_marker_still_resumes`,
  `test_a_halt_is_reported_as_a_halt_not_as_failed_attempts`.
- `--supervised` keeps the watchdog polling across a supervisor restart, so a
  recoverable crash on day one no longer leaves days two to four unwatched.
- `detect_stall`'s clock runs from the newer of `metrics.jsonl` and
  `train.pid`, so a restart is not counted as a stall — with a terminal halt
  that false positive would end the run rather than pause it. Tests:
  `test_a_fresh_pidfile_gives_a_restarted_attempt_its_own_grace_period`,
  `test_a_process_that_hangs_after_starting_still_stalls`.
- verified read-only against the live `sweep` probe: the pidfile holds the real
  training PID and the healthy-run path returns None and writes nothing.

`abl-arch` runs the same wiring first, so `hero` is not the first job to use it.

**And the dashboard you would watch it on survives a restart, as of 07:20Z.**
W&B mints a fresh run id on every `init()` — the run *name* is only a display
label — so each `run_with_resume` restart began a **new run at a new URL**,
freezing the link in this issue at the crash point while training carried on
unseen. Guaranteed on any restart, not probabilistic, and invisible to 765
tests because nothing pinned the init kwargs. Measured: two inits with identical
project/name/tags returned `ma562epj` and `vlr1n10i`. Now re-attaches — but only
when the process actually resumed a checkpoint, *not* on a run-name match, since
`sweep` was re-run under recycled names and appending a good curve to a
discarded one at the same step numbers would be worse than the bug. Verified
with two real `train.py` subprocesses landing in the same run `ae12d846`.
→ `runs/preflight/wandb-run-identity.md`, 6 tests.

### `abl-arch` result

<<TODO: paste `python scripts/abl_table.py --results runs/abl-arch/results.json`
verbatim — it renders both arms' val_bpb, CPU decode tok/s, Q4_0 deltas, the
Pareto verdict and the `hero` config the rule implies. Then add the sweep winner
+ is_a_comparison from scripts/check_sweep.py.>>

**Read the verdict against the rule, not against the numbers.** The rule was
fixed at 08:30Z on the 10th, with arm 1 a third of the way in and arm 2 not
started → `runs/preflight/abl-arch-decision-rule.md`. It exists because `sweep`
taught this project the lesson at a cost: its winner beat the runner-up by
**0.05%**, and a pre-registered tie rule is the only reason a noise winner did
not become a $41 decision. Quality is called against a **0.5% floor that is
inherited and unmeasured** — `train.py` has no `--seed` flag, so every run here
has shared seed 0 and nothing has ever measured seed sigma in bits-per-byte.
Decode is called against the bench's own reported stddev, which *is* measured.

### If the dense twin wins quality — the branch this gate does not currently ask
<!-- appendix -->

The decision box above recommends `daedalus-150m`, which is what the rule
returns for a tie, a hybrid win, or a missing score. **If dense wins by more
than 0.5%, the rule escalates and this ask changes**, so the alternative is
written out now rather than assembled at 06:45Z while the box burns $10.78/day:

> **Recommend instead:** you choose. Dense trains at **0.889×** the hybrid's
> rate — and that is now the *measured* ratio over two multi-hour runs
> (109,004 tok/s for arm 2 against arm 1's 122,612), not the preflight's
> 0.888 estimate, which it reproduces. So at **58B**, `hero` on dense is
> **~149.9 h / $67.31** against **133.3 h / $59.85** — **+$7.46 and +16.6 h** —
> and it runs those six unattended days at ~**28.4 GB of 32.6** rather than
> 24.29, the thinnest VRAM margin in the plan held for the longest this project
> has ever held one.
>
> It also gives up the CPU-decode half of the Pareto claim, which is the only
> axis on which SmolLM2-135M is beaten at all under the confirmed success
> definition. **It fits, but only just**: the +$7.46 comes straight out of the
> buffer, so *Projected credit*'s **≈$12.62** at completion becomes **≈$5.16**
> — and 16.6 extra hours is 16.6 more hours in which something can go wrong with
> no money to redo it. Dense at **51B** costs $59.18 and leaves **≈$13.29**,
> roughly the hybrid buffer back — but 51B is 12% fewer tokens, so it trades
> away part of the very quality advantage that would have triggered this branch.
> **Reply `go dense 58B` (≈$5.16 left) · `go dense 51B` (≈$13.29 left) ·
> `go hybrid anyway` · `hold`.**
>
> *(Re-priced 02:30Z. This paragraph read 60B / $69.70 / ≈$3.14 until then, and
> its cheaper option was a dense run at 55 billion tokens — the ask moved to 58B
> at 23:40Z and this branch did not follow, and 55B had no verified WSD schedule
> behind it. 58B and 51B both do. The retired option is spelled out rather than
> shown as a reply, so nobody answers with it.)*

That branch is a genuine finding if it happens — the ablation is allowed to come
out against the architecture the project is built on, and the writeup reports it
either way.

### Two things to read correctly in that output
<!-- appendix -->

- **`passes_threshold: false` on both arms is expected, not a failure.** Both
  run `qat_frac=0` by design — a quantized forward would invalidate the
  hybrid-vs-dense comparison. Measured on real trained checkpoints, Q4_0 costs
  **+2.576%** perplexity on one and **+1.558%** on another (a 500M-token
  probe), against a 1.0% threshold; last night's 0.03% came from smoke-trained
  arms whose weights were still near-Gaussian and quantize almost perfectly.
  Neither figure is `hero`'s, and they differ by checkpoint rather than by
  method — but both clear the bar in the same direction, which is the point:
  **QAT is load-bearing**, and `hero` is the only job that runs it. The flag is
  fatal in neither `export.py:329` nor `abl_arch.py:234`.
  **Both are underestimates** (2026-08-10 ~14:40Z): they were measured on an
  eval text carrying 153 literal `<|endoftext|>` strings that
  `llama-perplexity` does not parse, which diluted the delta by ~6% relative.
  Corrected on a real 4B-token checkpoint over one identical span,
  **+1.676% dirty → +1.781% clean**. The file is fixed, so tonight's arm
  figures are the clean ones and will not be comparable to the two above.
  → `runs/preflight/gguf-vs-pytorch-fidelity.md`
- **The two arms start at systematically different losses** (15.36 vs 12.80)
  purely because they are 768 vs 640 wide — zero-init residual projections make
  every block the identity at step 0. Not seed noise, not an architecture
  effect; self-corrects within the first few hundred of ~1e5 steps.

### Work queued for the gate wait itself
<!-- appendix -->

The box idles at $10.78/day from the moment this issue goes up until it is
answered, so that window is already budgeted for. This list is the authoritative
one — the items were scattered across `STATUS.md`, the timeline above and this
section, and the gate wait is the last window before a 5.9-day run.

**Blocking — `hero` must not start until these are done:**

1. ~~**The $2.25 corpus top-up** (issue #5), CPU only, `scripts/topup_for_60b.sh`.~~
   **Done** — moved off the critical path by running it *beside* `abl-arch` arm 2
   from 18:22Z on the 10th instead of serially in this wait window, so it floors
   nothing and cost no extra rent. It was the only item here that gated the
   *start time* rather than the decision.
   Without it the 4-epoch cap cannot be satisfied at 60B, so it switches off
   entirely and `everyday-conversations` repeats ~2,973 times. The script
   refuses to run beside training, and `train.py` now *grades* the condition
   (`max_epochs_seen > max_epochs`) rather than relying on `l1_skew_pts`, which
   reads a perfect **0.00** in exactly this case.
2. **Re-run the 7 CUDA-gated QAT tests.** They self-skip with *"a train.py
   process owns the GPU"*, so they have not run since 05:13Z today and will not
   while any arm trains — which means a suite reporting "1002 passed, 7 skipped"
   is not evidence about the one mechanism that runs at **hour ~131 of 137.9**,
   after ~$58 is already spent. They are also the tests that pin the compiled-
   path fix from `runs/preflight/qat-compile-lattice.md`, where inductor folded
   away the fp16 scale round-trip and silently moved the lattice off the one
   llama.cpp stores — a failure that raises nothing, leaves the loss curve
   unchanged, and keeps `qat_rel_rmse` reporting success.
   `python -m pytest tests/test_qat.py -q -rs` on the idle box; **zero skips is
   the pass condition**, not "no failures".

   **This had no owner until 20:33Z on the 10th** — the same gap the paired
   decode re-bench had 34 minutes earlier, and found the same way, by walking
   this checklist against `ps` rather than re-reading it. None of the seven armed
   jobs ran pytest, so a blocking precondition would have been satisfied only if
   I happened to remember it between the gate opening and your reply. Now armed
   as `scripts/qat_tests_when_quiet.sh`: it waits for `results.json` and then for
   a genuinely quiet box (the trainer, the val pass, the eval and the re-bench
   all own the GPU in turn after arm 2 stops), runs the tests, and writes
   **`runs/preflight/qat-gate-evidence.md`** with the verdict. It waits for the
   re-bench but does not require it to have *succeeded* — gating a blocking item
   on a non-blocking one is how a failure in the cheap thing cancels the
   important one.

   The verdict is computed from pytest's JUnit XML, not its summary line,
   because the first version parsed the prose and scored a clean run **FAIL**
   and a real failure as **0 failures** — wrong in both directions. Nine tests
   in `tests/test_qat_tests_when_quiet.py` cover it, including the one that
   matters: pytest exits **0** when every test skips, so `rc` alone can never be
   the pass condition.

   **`hero` now enforces this itself rather than trusting me to remember it.**
   `hero._cli` refuses with **rc 2** — before the split, the trainer or the
   watchdog — if that file is missing or does not record a PASS, writing the
   reason to `STATUS.md`, in the same shape as the mixture refusal beside it.
   `--allow-unverified-qat` overrides it and records the choice. The check is
   skipped when `--qat-frac 0` turns the mechanism off, so it cannot become a
   false blocker on a run that never quantises. Six tests in `tests/test_hero.py`
   cover the refusal, the override and the switched-off case.

   This matters because the previous guarantee was "the agent will check", and
   the loop supervisor has already died unnoticed once (issue #3). A blocking
   precondition whose only enforcement is my memory across sessions is not a
   blocking precondition.

**Non-blocking, worth the idle time:**

3. **Paired decode re-bench** — `scripts/rebench_arms.py`, ~3 min CPU. Required
   for the headline ratio to be paired rather than 12 h apart; `abl_table.py`
   marks it "indicative" if absent, so a skip is visible rather than silent.
4. **Score the shipped Q4_0 GGUF.** `eval.py` has no GGUF path, so every quality
   number so far describes the PyTorch model while the artifact you run is Q4_0
   — which costs ~1.8% perplexity (corrected; see above). The *other* half of
   that gap is now closed: PyTorch-vs-GGUF fidelity was measured for the first
   time on 2026-08-10 and the GGUF is faithful to **−1.99%**, paired, on a
   frozen checkpoint, with the residual being a 0.037% tokenizer-merge
   divergence rather than the graph. So what remains unmeasured is *task*
   accuracy through the GGUF, not whether the export is sound.
   `llama-perplexity` here supports
   `--hellaswag`, `--winogrande` and `--multiple-choice`; its conventions differ
   from lm-evaluation-harness so it cannot replace the peer table, but run over
   the fp16 and Q4_0 GGUFs of the same model it measures the shipping cost
   directly. At peer accuracies within a point of each other, that is the
   difference between clearing the bar and not.
5. **0.5B paired re-score + McNemar**, on the GPU that is finally free — the
   peer comparison is currently unpaired, which overstates the error of a
   difference since every model answers the same items.
6. Whatever `abl-arch`'s output raises.

### Contamination — measured at 22:45Z, so the 44.7 can be defended

The first challenge to any surprising small-model score is that it saw the test
set, and until tonight the answer was an argument. It is now a measurement:
`scripts/contam_scan.py` over 224.7M tokens (1.32% of the corpus), 174,932 whole
documents, classified with the same `ngram_set` the build filtered on.

| index | rate over training tokens | docs |
|---|---|---|
| `filtered` — **negative control** | **0.0000%** | **0 / 174,932** |
| `split_gap` — scored splits the build never indexed | 0.0004% | 1 / 174,932 |
| `limit_gap` — items past the build's `limit=2000` | 0.0012% | 1 / 174,932 |

The control returning exactly zero is what makes the other two rows mean
anything. It also exposed a defect nobody knew about: the build logs record a
183,359-gram index where today's code builds 214,682, because `334c86c` moved
ARC-Easy and OpenBookQA onto their `test` splits *after* most of the corpus was
built — so for most of this corpus the splits we score on were never filtered.
Its measured cost is the `split_gap` row: one document, matching one 13-gram of
ordinary physics prose.

**No action before `hero`.** The running top-up indexes today's splits at
`limit=1,000,000`, so every token it added is fully covered on both counts, and
0.0012% does not justify rebuilding a corpus. Full report:
`runs/preflight/contam-exposure.md`.

### One durability hole found and closed after those four were signed off
<!-- appendix -->

`train_step` skips the optimizer on a non-finite loss without incrementing
`self.step`, so once the **weights** are NaN every step is skipped and
`metrics.jsonl` goes quiet — and `watchdog.py` needs up to `--stall-min` (30
min) to notice. `maybe_checkpoint` runs on a **30-minute** gate. Those two
clocks race, and if the checkpoint one wins, `runs/hero/checkpoint.pt` is
overwritten with the NaN model and the `rolling` Hub revision follows it two
hours later.

That is precondition 1 and 2 failing together at hour 90, and the preconditions
as written did not cover it. All three durable writers now refuse a non-finite
model; the milestone refuses *without* consuming itself. Nothing about training
changes — only what gets persisted. 6 tests, four of which fail against the
pre-fix file.

### A second hole, found at 03:50Z by running the crash-recovery loop for the first time
<!-- appendix -->

`supervise.run_with_resume` — what restarts `train.py` after a crash, and so
what makes 5.6 days survivable — is imported by `hero.py` and **nothing else**,
and every test of it injects a fake `runner`. A real SIGKILL followed by a real
`--resume` had never happened. Rehearsed on CPU against a real trainer: **it
works.** Three bugs fell out of asking what attempt two actually does, all fixed
and tested, all verified failing on the pre-fix files:

1. **A resume was scored as a divergence — which is terminal** (`08173f9`). The
   rolling checkpoint gate fires on its first call, so a run's first checkpoint
   is at step 1 and its second 30 min later; a crash inside that window resumes
   correctly and then reports the step-20 loss against the pre-crash running
   mean, because `metrics.jsonl` spans attempts. On arm 1's real curve that is
   **9.3150 against a 7.4176 threshold** — so `hero` recovers at minute ~25 and
   is killed by its own watchdog at minute ~26, blaming the learning rate. Arm 2
   OOM'd twice at startup yesterday, so this is the observed case.
2. **Nothing ever checked the Hub uploader was still alive** (`235482f`) —
   precondition 1's mechanism. It now respawns on the same 2 h gate.
3. **This document's "measured" `abl-arch` cost dropped every attempt but the
   last** (`32b6f59`): `cost_usd` resets on resume and arm 2 was resumed twice,
   understating by **$0.52 / 1.1 h** — the direction that flatters a budget. The
   cost row below is now summed over attempts and says so. `COSTS.md` was never
   affected; it prices from wall clock.

Full note: `runs/preflight/supervisor-rehearsal.md`.

### A third hole, found at 04:40Z: the insurance had no alarm of its own
<!-- appendix -->

Both guards on precondition 1 watch the *process* — `_ensure_uploader` respawns
one that died, `_warn_if_hub_stalled` prints past 3× the cadence. An uploader
that is **alive and whose every upload is rejected** (quota, revoked token,
outage) trips neither: `upload_once` catches every exception by design and
leaves the payload pending, which is right for a transient failure and
indistinguishable from a permanent one. The worst case was strictly invisible —
`_warn_if_hub_stalled` returns early on `hub_stale_h is None`, and None means
*nothing has ever uploaded*, so uploads failing from step 1 report None for six
days. `scripts/hub_watch.py` (27 tests) now alarms on the symptom, armed inside
the existing credit loop, verified against the live arm-2 trainer.

Arming it found two live bugs, both proven against the pre-fix code: the loop's
`flock` fd was inherited by its own `sleep`, so an orphan held the lock for 30
min after the parent was killed — and since the documented relaunch check is
`ps | grep credit_watch_loop`, which reads empty, **following the launch
procedure at 07:52Z would have left `hero` with no watch at all**; and
`write_status` closed over its own markers, so the hub block *deleted the credit
alarm*.

**It also put a number on a quota nobody had checked.** The Hub keeps every
superseded LFS blob, so `rolling` costs 321 MB × every upload ever made.

| | |
|---|---|
| free-tier **private** quota | **100 GB** |
| private in use now | **57.4 GB** (corpus 34.19, abandoned `daedalus-shards` 13.07, checkpoints 10.09) |
| `hero` adds | ~67 rolling + 2 milestones = ~24.4 GB |
| **projected at run end** | **~81.7 GB of 100 GB — it fits** |

Reclaiming is **not a fast lever** (the Hub documents up to 36 h before a squash
returns quota), so it has to be caught on the projection. Separately,
`Unseen1980/daedalus-shards` (13.07 GB) is the abandoned first dataprep attempt,
superseded by `daedalus-corpus`; deleting it returns 13.07 GB. **I have not
deleted it** — irreversible, your account, your call. Nothing needs it.

### What this does not prove

Single seed, so the hybrid-vs-dense gap carries seed sigma. Distillation was
dropped (costed, recorded). Document-aligned packing is not implemented — now a
**measured and costed** deviation rather than a bare admission, see the section
below. Sampling with replacement leaves **18.09%** of the two maths sources'
unique tokens undrawn at their 1.71 epochs (1.83% at the 4.00-epoch sources);
re-priced at the 60B budget tonight, deliberately not fixed — rewriting data
sampling hours before launch is the larger risk. NoPE was skipped because it
breaks GGUF export. The bar remains: beat Pythia-160M
(41.0), GPT-neo-125M (41.9), OPT-125M (42.1) and GPT-2 124M (42.2) on the
**5-task mean measured on this harness**; concede SmolLM2-135M (51.2) on quality
while beating it decisively on CPU decode.

(Those five figures are the *measured* column. The published 8-task averages for
the same peers are 42.5 / 42.9 / 42.6 / — / 50.7, which include ARC-Challenge,
BoolQ and SIQA — tasks `AGENT.md` §3 deliberately does not run, so they are a bar
this project can never measure itself against.)

### The one blueprint item still missing — measured, priced, and deliberately not taken
<!-- appendix -->

`DAEDALUS-BLUEPRINT-v6.md:37` and `:93` lock **document-aligned packing via
FlexAttention**. It is not implemented: `model.py:133` is a plain causal
`scaled_dot_product_attention`, and `data.py:220-224` derives per-token document
ids that nothing consumes (its own docstring says *"for a **future**
document-aware attention mask"*). `hero` is the last chance to add it, so it was
settled rather than carried.

**The gap is larger than I expected.** Measured on the real shards — 2,000 real
2048-token windows per source — **37.3% of the causal (query, key) pairs this
model trains on cross a document boundary**, and 47.8% of positions have at least
one foreign token in context, at ~2.7 documents per sequence. Worst source
`cosmopedia-v2` at 66.3%, best `finepdfs-edu` at 17.3%.

**And it still should not be implemented**, because the one published ablation
that matches our setting on every axis measured no effect. HuggingFace's *Smol
Training Playbook* ran causal-vs-intra-document masking at **1B params / 45B
tokens on FineWeb-Edu + FineMath + Python-Edu** — three of our four largest
sources — scored on **HellaSwag, ARC, PIQA, OpenBookQA, WinoGrande, MMLU**, and
reports *"identical loss curves and downstream evaluation scores"*, one small PIQA
gain aside, concluding *"we don't observe a noticeable impact on short context
tasks."* They adopted it for SmolLM3 regardless — explicitly for the 4k→64k
**context extension**, which Daedalus does not do. Llama 3 reports the same shape:
limited impact short-context, significant for long-context.

The paper that introduced the technique (Zhao et al., ACL 2024) *does* show a real
win at 1.3B/150B/2K — −0.883 perplexity (−8.3%) and +6.98 points on few-shot ICL —
but reports **none of our five tasks**, and its headline is few-shot
classification, where the neighbouring documents *are* the in-context examples.
Ours is zero-shot cloze. Both results can be true at once, and **the bar is
written in the quantity HF measured**.

| | |
|---|---|
| cost | 4.0% throughput (Zhao et al.'s own figure) = ~5.7 GPU-h of `hero` = **~$2.55** |
| risk | editing the shared attention path ~4 days before a 5.9-day, $63.78 run |
| consistency | arm 1 has already trained 5B tokens without it |
| benefit on the five tasks in the bar | **not demonstrated** |

A trap worth naming: implementing it would probably *improve our own `val_bpb`*,
which is the number on the dashboard — and buy nothing on the bar. → full
measurements, sources and residual doubts in `runs/preflight/document-masking.md`.
**Nothing to approve here; recorded so the deviation is a decision with a price on
it rather than an omission.**

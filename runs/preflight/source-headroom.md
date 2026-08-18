# Which source runs out next? — measured, 2026-08-10 ~22:0xZ

`stack-edu-python` ran out of documents mid-build:

    stream exhausted at 1,210,964,651 tokens of a 1,350,000,000 budget
    -- this source has no more documents

That shortfall is permanent, and it fed directly into the `hero` mixture gate,
which clears 60B by only 0.64%. So "which source dries up next" stopped being
trivia. `STATUS.md` named the live version of the question:

> The risk is the source **running out of documents**, which is exactly what
> `stack-edu-python` just did, and which no amount of waiting fixes.

Measured with `scripts/source_headroom.py` (26 tests), against the live
manifest and the real Hub metadata, at the running top-up's budgets:

| source | verdict | files consumed | still needed | reachable (lower bound) | cover |
|---|---|---|---|---|---|
| `stack-edu-python` | **EXHAUSTED** | **10 / 10** | 139,035,349 | 0 | **0x** |
| `fineweb-edu` | SAFE | 4 / 2,410 | 1,874,997,391 | 1,432,363,223,975 | 764x |
| `finepdfs-edu` | **MET** | 1 / 100 | — | 55,737,238,402 | inf |
| `dclm-baseline` | MET | 29 / 27,838 | — | 3,742,416,666,870 | inf |
| `finephrase` | MET | 30 / 27,104 | — | 1,264,208,266,853 | inf |
| `finemath-3plus` | MET | 4 / 128 | — | 33,162,520,616 | inf |
| `infiwebmath-3plus` | MET | 3 / 64 | — | 20,194,637,223 | inf |
| `cosmopedia-v2` | MET | 3 / 104 | — | 23,768,554,072 | inf |
| `finewiki-en` | MET | 0 / 15 | — | 6,312,958,654 | inf |
| `everyday-conversations` | MET | 0 / 1 | — | — | inf |

`stack-edu-python` is the only short source, and it is the one already known to
be short. **Nothing else is at risk**, and one number explains why: the
exhausted source is the only one in the mixture that reads a *small* file set —
10 parquet files, 1.79 GB total. Every other source reads between 15 and 27,838
files.

## The 60B gate risk is now realised rather than projected

`finepdfs-edu` was the source `STATUS.md` flagged as deciding the launch:

> `finepdfs-edu` must reach ≥ 1,124,340,092 tokens (93.7% of its 1.20B budget)
> or `hero` refuses at 60B.

It finished its **full** budget — `done finepdfs-edu: tokens=1,200,001,535` —
so the threshold is met with 75.7M to spare, and it was never close to running
dry: 100 files / 298.67 GB of `eng_Latn`, of which it has consumed part of two.
Re-run against the corpus as it now stands, with `fineweb-edu` assumed to finish
the 1.167B it has left:

    corpus       17,577,556,578 tokens across 10 sources
    train split  17,226,005,446 (2.0% holdout carved)
    at 60B       l1_skew 9.01 pts (limit 10.0), worst repetition 4.0x
    verdict      hero would LAUNCH        largest budget: 60.39B (+0.64%)

Unchanged from the projection, but no longer resting on an assumption about
`finepdfs-edu`.

**Consequence for the gate's reply options.** The `go 58B` fallback exists to
cover "`finepdfs-edu` stops dead where it is". That source is now complete, and
the only unfinished source has 764x the headroom it needs, so the fallback
insures against a risk that is measured at zero rather than merely believed
small. It costs 2B tokens. It should not be chosen out of caution about this.

## The part worth keeping: the shortfall was predictable from metadata

`stack-edu-python` reads 10 files totalling 1.79 GB. At the density it actually
achieved that is ~1.21B tokens, so a **1.35B budget was never satisfiable** — no
retry, no extra RAM and no scheduling change could have produced it. One Hub
metadata call would have said so in about a second, before any document was
streamed. Instead it was discovered ~1 h into a live top-up.

`test_the_shortfall_was_predictable_from_a_prefix_of_the_source` pins that:
given only the first half of the source, the measured density already projects
the whole thing below its budget, so the tool reports AT_RISK while there is
still time to react rather than EXHAUSTED afterwards.

## Why the numbers lean the safe way

Two deliberate biases, so a SAFE verdict cannot be an artefact of optimism:

* **Density is a lower bound.** Tokens produced are divided by the bytes of
  every file *including* the partly-read current one, and remaining bytes
  *exclude* that file's untouched tail. Both errors point down.
* **Density is measured, not assumed.** It comes from the source's own output,
  so `fineweb-edu`'s `int_score >= 3` filter, the near-dup drop and the
  100k-char document cap are already priced in.

File lists come from `datasets.load_dataset_builder` — the same config /
`data_files` resolution `dataprep` streams through — not a path-prefix guess. A
heuristic that silently mismatched a config would report SAFE for a source it
had never looked at, which is the one failure this must not have. It resolves
`finepdfs-edu`'s `eng_Latn` to 100 files and `stack-edu-python`'s
`Python-all/partial-train/*.parquet` to 10, both confirmed independently against
the Hub file listing.

The `shard_idx` reduction is a **max** over the whole nested stream state,
because HF nests an inner `examples_iterable` beside a `previous_state` and
which one is ahead differs between sources. Checked against both live shapes:
`stack-edu-python` reduces to 10 (EXHAUSTED), `finepdfs-edu` to 0 at the time of
writing.

## Re-run it

    python scripts/source_headroom.py            # non-zero exit if any source is short

Worth running before any future corpus target is chosen, and it is cheap
(metadata only, no streaming, no GPU).

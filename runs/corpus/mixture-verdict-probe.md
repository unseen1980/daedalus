# Phase 7 mixture selection -- tag `probe`

**Verdict: keep-baseline `baseline`** -- best admissible arm 'quality-heavy' gains 0.0810% of aggregate BPB, under the preregistered 0.5000%

Measured on `/workspace/daedalus/data/holdout` at context 1024, full pass, every arm aggregated under one fixed weighting: `dclm-baseline` 0.326, `fineweb-edu` 0.543, `stack-edu-python` 0.130.

| arm | aggregate BPB | vs baseline | BPB dclm-baseline | BPB fineweb-edu | BPB stack-edu-python | admissible |
|---|---|---|---|---|---|---|
| `baseline` | 1.2486 | +0.00% | 1.3828 | 1.2441 | 0.9319 | baseline |
| `quality-heavy` | 1.2476 | +0.08% | 1.4035 | 1.2373 | 0.9005 | yes |
| `derived` | 1.2479 | +0.06% | 1.3899 | 1.2490 | 0.8879 | yes |

## Rule

- floors: `code` >= 0.0522, `web-raw` >= 0.1304
- max per-source regression vs baseline: 5.00%
- minimum aggregate gain before the mixture changes: 0.50%
- floored domains with no source under this root, so with nothing to constrain: `math`

## Shares the cap set, not the measurement

The derived arm tilts each blueprint share by `exp(excess/T)`, clipped. For these sources the clip bound, not the measured headroom, is what fixed the share -- read them as the most the rule would grant, not as a measured optimum:

- `stack-edu-python` saturated the upper bound: +0.2126 bits/byte of excess loss asks for 8.38x its blueprint share, and the cap allowed 2.00x.

## Arms

- `baseline` (dataprep.MIXTURE shares, renormalized over the sources present under the data root): dclm-baseline 0.3261, fineweb-edu 0.5435, stack-edu-python 0.1304 -- scorecard `runs/corpus/scorecards/mix-probe-baseline-bpb.json`, checkpoint `543f10936839`
- `quality-heavy` (raw-web sources scaled to 0.45 of their blueprint share, the freed mass redistributed over the filtered sources in proportion to theirs): dclm-baseline 0.1467, fineweb-edu 0.6881, stack-edu-python 0.1651 -- scorecard `runs/corpus/scorecards/mix-probe-quality-heavy-bpb.json`, checkpoint `0c7abd25adc7`
- `derived` (blueprint shares tilted by exp(excess/0.1), clipped to 2x, then floored at 0.4 of each floored domain's blueprint share): dclm-baseline 0.2849, fineweb-edu 0.5369, stack-edu-python 0.1782 -- scorecard `runs/corpus/scorecards/mix-probe-derived-bpb.json`, checkpoint `45cea76cad7d`

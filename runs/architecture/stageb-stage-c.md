# Phase 6 stage-C go/no-go, from stage-b (stageb)

Three preregistered conditions, all required: the previous stage separated its arms beyond its own parameter-matching residual, at least one arm carries no measured failure, and 1,006,632,960 tokens for at most 2 finalists plus the control still fit before finalization.

| condition | status | note |
| --- | :---: | :--- |
| discrimination | **FAIL** | BPB spans only 0.26 points across 4 arms, inside the 0.84 points their 2.48% parameter spread could explain on its own; the ordering is not separable from the grid's matching residual at this scale |
| finalists | **FAIL** | every non-control arm is blocked (['a3-kv4', 'a4-kv4', 'a6-kv4', 'a8-kv4']); a measured failure does not reopen at four times the budget |
| deadline | pass | 7.8 hours needed against 92.0 before finalization opens |

## Outcome

- **verdict: `no-go`**
- finalists: none
- unmet: ['discrimination', 'finalists']
- estimated training hours: 6.8

> stage-b does not hand on a stage C: ['discrimination', 'finalists'] unmet. This is a preregistered negative result and not a deferral -- the numbers that decided it are in `conditions`, and the shapes stand on the evidence already gathered. Re-running the decision at a looser condition after seeing this would be threshold-tuning.

## What a `go` would have required

- discrimination: a measured BPB spread wider than 0.84 points, against the 0.26 observed
- finalists: at least one non-control arm not blocked, of 4 gated
- deadline: 92.0 hours before finalization to cover 7.8

## Read from

- recommendation: `runs/architecture/stageb-recommendation.json`
- report: `runs/architecture/stageb-report.json`
- state: `runs/vast-program/state.json`

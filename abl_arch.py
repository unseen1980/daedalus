"""`abl-arch`: the project's headline experiment (AGENT.md SS4). Trains
daedalus-150m (hybrid) and dense-150m (its param-matched all-attention twin,
see config.py's PRESETS) on identical data for 5B tokens each, then reports
held-out val BPB and measured llama.cpp CPU decode tok/s for both. This
script only trains + measures + writes runs/abl-arch/results.json -- it does
NOT open the AGENT.md "[ASK HUMAN] ready for hero" issue itself; posting the
real results and waiting for the operator's decision is the calling agent's
job, done once after reviewing what this script actually produced.

Design notes:
  - "identical data, same seed, same steps": both configs train via a
    train.py *subprocess* (isolates torch.compile/CUDA context between runs,
    same reasoning as sweep.py) with the same --data-dir, --total-tokens,
    and seq/tok schedule. train.py has no --seed CLI flag -- TrainArgs.seed
    always defaults to 0 -- so both runs draw batches from the exact same
    MixtureBatchSource RNG stream given the same --data-dir, and the same
    total_tokens/micro_batch/seq schedule means the same step count too.

    **This holds only while neither arm is interrupted.** A resumed run is
    positioned by its `tokens_seen` (train.py's `set_position`), so it draws a
    different -- deliberately non-repeating -- stream from that point on. If
    one arm is resumed and the other runs straight through, "identical data"
    is no longer true and the head-to-head loses the property that makes it
    publishable. Each arm is ~12 h, so this is a live possibility rather than
    a theoretical one: if an arm dies, restart **both** from scratch rather
    than resuming one, or state the deviation with the result.
  - --mixture-dir defaults to the full dataprep.py mixture root (one
    subdirectory per source, each with its own manifest.json). Unlike
    sweep.py, which deliberately probes a single source (lr selection isn't
    sensitive to exact composition), abl-arch needs the real mixture --
    AGENT.md SS4 frames it as the project's headline, publishable result.
    train.py's Trainer auto-detects a mixture root vs. a single-source dir
    (see MixtureBatchSource's docstring); this script always passes a
    mixture root, never a single source.
  - held-out val BPB is computed over a per-source holdout split of the
    mixture (daedalus.data.make_mixture_holdout_split), token-weighted
    across sources -- eval.py's evaluate_bpb only understands one shard
    directory at a time, so this loads the checkpoint once and loops over
    each source's holdout split itself rather than reloading per source via
    eval.py's evaluate_checkpoint (as sweep.py does for its single source).
  - CPU decode tok/s is measured by exporting each trained checkpoint to
    Q4_0 GGUF (export.py) and running llama-bench (export.py's
    measure_decode_speed) -- the actual "measured llama.cpp CPU decode
    tok/s" AGENT.md SS4 asks for, not an estimate.
  - No "winner" is picked here (unlike sweep.py's best-by-val_bpb): AGENT.md
    frames abl-arch as a head-to-head report for a human decision, not an
    automatic selection.
"""
import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional

import watchdog as watchdog_mod
from daedalus.supervise import start_watchdog, stop_watchdog


def run_training(data_dir: str, config: str, total_tokens: int, run_name: str,
                 wandb_enabled: bool, extra_train_args: Optional[List[str]] = None,
                 max_attempts: int = 3, report: Optional[dict] = None,
                 watchdog: bool = True, stall_min: float = 30.0) -> str:
    """Launch one train.py subprocess for this config; returns its checkpoint
    path. Raises CalledProcessError if every attempt exits non-zero.

    Retries with --resume rather than losing the arm. Each arm is ~12 h and
    the whole thing runs unattended; a single transient failure used to
    propagate straight out of `subprocess.run(check=True)` into
    run_abl_arch's handler, which recorded the error and moved on to the next
    config -- discarding up to 12 h of training and ~$5.4 for a blip.
    train.py checkpoints every 30 min, so a resume costs at most that.

    Resuming is not free, and it is recorded rather than hidden. A resumed
    arm draws a different (deliberately non-repeating) data stream from the
    restart point, so it no longer sees byte-identical data to an arm that
    ran straight through -- the property that makes the head-to-head
    publishable. This module's docstring allows exactly two responses:
    restart both arms from scratch, or state the deviation with the result.
    Losing the arm entirely serves neither, so we take the second and write
    `resumed`/`attempts` into results.json, where the writeup has to confront
    it. Re-running both arms remains the rigorous option and stays the
    caller's decision, made with the numbers in hand.
    """
    run_dir = os.path.join("runs", run_name)
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    base_cmd = [sys.executable, "train.py",
                "--run-name", run_name, "--config", config,
                "--data-dir", data_dir, "--total-tokens", str(total_tokens),
                "--tags", "abl-arch"]
    if not wandb_enabled:
        base_cmd.append("--no-wandb")
    if extra_train_args:
        base_cmd += extra_train_args

    # A halt from a previous launch must not block this one; cleared here, at
    # the start of the arm, and never inside the retry loop below -- that loop
    # is exactly what the marker exists to stop.
    watchdog_mod.clear_halt_marker(run_dir)
    wd = start_watchdog(run_name, run_dir, total_tokens,
                        stall_min) if watchdog else None

    attempts, resumed = 0, False
    try:
        for attempt in range(1, max_attempts + 1):
            cmd = list(base_cmd)
            # Only resume from a checkpoint that exists: a first attempt that dies
            # before the 30-min mark leaves none, and train.py would silently
            # start from scratch anyway. Being explicit keeps `resumed` honest.
            this_resume = attempt > 1 and os.path.exists(ckpt_path)
            if this_resume:
                cmd += ["--resume", ckpt_path]
            attempts = attempt
            # Record before running, not after: the successful attempt exits the
            # loop via `break`, so anything set afterwards is skipped -- which had
            # a resumed-then-succeeded arm reporting resumed=False, the exact
            # silent-provenance failure this flag exists to prevent.
            resumed = resumed or this_resume
            print(f"=== training {config} (attempt {attempt}/{max_attempts}"
                  f"{', resuming' if this_resume else ''}): {' '.join(cmd)} ===",
                  flush=True)
            try:
                subprocess.run(cmd, check=True)
                break
            except subprocess.CalledProcessError as e:
                # A watchdog halt is not a crash. Retrying a divergence resumes the
                # diverged checkpoint -- and since the watchdog exits after halting,
                # the retry runs unwatched to the end of its 12 h arm and reports a
                # val_bpb that means nothing. Stop and say why.
                halt = watchdog_mod.read_halt_marker(run_dir)
                if halt is not None:
                    print(f"=== {config} halted by the watchdog "
                          f"({halt.get('kind')}): {halt.get('reason')}; "
                          f"not resuming ===", flush=True)
                    if report is not None:
                        report["attempts"] = attempts
                        report["resumed"] = resumed
                        report["halt"] = halt
                    raise RuntimeError(
                        f"halted by watchdog ({halt.get('kind')}): "
                        f"{halt.get('reason')}") from e
                if attempt == max_attempts:
                    print(f"=== {config} failed {max_attempts} attempts; giving up ===",
                          flush=True)
                    if report is not None:
                        report["attempts"] = attempts
                        report["resumed"] = resumed
                    raise
                print(f"=== {config} attempt {attempt} failed ({e}); "
                      f"retrying with --resume ===", flush=True)
    finally:
        # Before the eval/export that follows this call: the arm is trained, so
        # nothing is left to watch, and a lingering watchdog would read the
        # finished run's own metrics as a stall.
        stop_watchdog(wd)

    if report is not None:
        report["attempts"] = attempts
        report["resumed"] = resumed
        if resumed:
            report["identical_data_caveat"] = (
                "this arm was resumed after a failure, so from the restart "
                "point it drew a different data stream than an arm that ran "
                "straight through; the head-to-head is no longer "
                "byte-identical in data order")
    return ckpt_path


def mixture_sampling_weights(train_root: str, total_run_tokens: Optional[int],
                             max_epochs: float = 4.0) -> Dict[str, float]:
    """The per-source probabilities `MixtureBatchSource` will draw `train_root`
    with: blueprint shares, restricted to sources actually present, epoch-capped
    against what is on disk, renormalized. Mirrors that constructor so val_bpb
    is weighted the same way training samples."""
    from daedalus.dataprep import MIXTURE
    from train import cap_weights_by_epochs

    weights = {s.key: s.share for s in MIXTURE}
    present = {name: w for name, w in weights.items()
               if os.path.exists(os.path.join(train_root, name, "manifest.json"))}
    if not present:
        raise ValueError(f"no source under {train_root!r} has a manifest.json")
    total_weight = sum(present.values())
    probs = {n: w / total_weight for n, w in present.items()}
    if total_run_tokens:
        tokens_on_disk = {}
        for name in probs:
            with open(os.path.join(train_root, name, "manifest.json")) as f:
                tokens_on_disk[name] = int(json.load(f)["total_tokens"])
        probs = cap_weights_by_epochs(probs, tokens_on_disk, total_run_tokens,
                                      max_epochs)
    return probs


def eval_val_bpb(ckpt_path: str, holdout_root: str, config: str, device: str,
                 seq_len: int = 2048, bpb_batch_size: int = 8,
                 weights: Optional[Dict[str, float]] = None) -> dict:
    """Held-out BPB across every source under `holdout_root` (a
    daedalus.data.make_mixture_holdout_split output). Loads the checkpoint
    once and reuses it across sources, unlike sweep.py's single-source
    eval_probe (which goes through eval.py's evaluate_checkpoint, reloading
    the model -- fine for one source, wasteful across a whole mixture).

    `weights` should be the training sampler's per-source probabilities. This
    used to weight by each source's *holdout* token count, which is not the
    mixture: `make_holdout_split` reserves whole shard files, so a source's
    holdout share is set by the arbitrary size of its trailing partial shard.
    Measured on the real 9-source corpus that put stack-edu-python at 1.71x its
    training share, dclm-baseline at 2.06x and fineweb-edu at 0.65x -- so the
    headline val BPB described a code-and-wiki-heavy blend nobody trained on.
    Both arms shared the distortion, so the hybrid-vs-dense *ranking* stayed
    fair; the reported number did not mean what it said.
    """
    from daedalus.config import PRESETS
    from daedalus.data import get_tokenizer
    from daedalus.model import Daedalus
    from eval import evaluate_bpb_mixture
    from train import load_checkpoint

    tokenizer = get_tokenizer()
    cfg = PRESETS[config]
    model = Daedalus(cfg).to(device)
    load_checkpoint(ckpt_path, model, map_location=device)
    model.eval()

    return evaluate_bpb_mixture(model, holdout_root, seq_len, tokenizer,
                                device=device, batch_size=bpb_batch_size,
                                max_batches=None, weights=weights)


def export_and_bench(ckpt_path: str, config: str, out_dir: str, llama_cpp_dir: str,
                     ppl_text_file: Optional[str] = None, n_ctx: int = 512,
                     bench_n_gen: int = 128) -> dict:
    """checkpoint -> Q4_0 GGUF -> measured CPU decode tok/s, plus (when a text
    file is given) the fp16-vs-Q4_0 perplexity delta as export.py's own
    quantization sanity check.

    Decode speed is measured *unconditionally*. It used to sit inside
    `if ppl_text_file:`, because the Q4_0 GGUF it needs was produced as a
    side-effect of `verify_quantization`. `--ppl-text-file` defaults to None,
    so a default `abl-arch` invocation trained both models for ~$11 and then
    silently reported no decode numbers at all -- no error, just a missing key.
    That is the measured-CPU-decode half of the hybrid-vs-dense Pareto claim,
    i.e. the reason this experiment exists. Conversion and quantization are now
    separate from perplexity, so only the perplexity check depends on the text
    file.
    """
    from export import (MAX_PPL_DELTA_PCT, convert_to_gguf, export_hf_model,
                        export_tokenizer, measure_decode_speed,
                        measure_perplexity, perplexity_delta_pct, quantize_gguf)

    hf_dir = os.path.join(out_dir, "hf")
    export_hf_model(ckpt_path, config, hf_dir)
    export_tokenizer(hf_dir)

    fp16_path = os.path.join(out_dir, "model-f16.gguf")
    q4_0_path = os.path.join(out_dir, "model-q4_0.gguf")
    convert_to_gguf(hf_dir, fp16_path, llama_cpp_dir, outtype="f16")
    quantize_gguf(fp16_path, q4_0_path, llama_cpp_dir, qtype="Q4_0")
    result = {"hf_dir": hf_dir, "fp16_gguf": fp16_path, "q4_0_gguf": q4_0_path}

    if ppl_text_file:
        fp16_ppl = measure_perplexity(fp16_path, llama_cpp_dir, ppl_text_file, n_ctx)
        q4_0_ppl = measure_perplexity(q4_0_path, llama_cpp_dir, ppl_text_file, n_ctx)
        delta_pct = perplexity_delta_pct(fp16_ppl, q4_0_ppl)
        result.update({"fp16_ppl": fp16_ppl, "q4_0_ppl": q4_0_ppl,
                       "delta_pct": delta_pct,
                       "passes_threshold": delta_pct < MAX_PPL_DELTA_PCT})
    else:
        print("=== abl-arch: no --ppl-text-file, SKIPPING the fp16-vs-Q4_0 "
              "perplexity check (decode speed is still measured) ===", flush=True)

    result["decode_speed"] = measure_decode_speed(q4_0_path, llama_cpp_dir,
                                                  n_gen=bench_n_gen)
    return result


# Below this, the probes agree to within their own run-to-run noise and the
# sweep has not actually chosen anything. Kept identical to
# `scripts/check_sweep.py`'s constant of the same name -- that script prints the
# verdict, this one acts on it, and a test pins them equal so they cannot drift.
NOISE_FRAC = 0.005


def _winner_margin(data: dict) -> Optional[float]:
    """How far the best probe beats the *runner-up*, relative to the best, or
    None if fewer than two probes scored.

    Margin, not the full-grid spread `check_sweep.py` reports, and the
    difference decides real cases. Tonight's grid is the example: probes at
    0.01 and 0.02 came in 0.43% apart -- a tie -- so if the third probe at 0.04
    lands 2% worse, the *range* is 2% and looks decisive while the choice
    actually being made, 0.02 over 0.01, is still noise. A bad third probe
    would then have unlocked deviating from the blueprint on a distinction
    nothing measured. Only the gap to the runner-up bears on which lr to train
    at.

    Deliberately total: this runs at the launch of a ~$11, ~24 h unattended
    job, so a malformed or older best.json must fall through to the previous
    behaviour rather than raise.
    """
    try:
        vals = sorted(v for v in (p.get("val_bpb") for p in
                                  (data.get("probes") or []))
                      if isinstance(v, (int, float)) and v == v and v > 0)
        if len(vals) < 2:
            return None
        return (vals[1] - vals[0]) / vals[0]
    except Exception:
        return None


def resolve_muon_lr(muon_lr: Optional[float],
                    best_json: Optional[str]) -> tuple:
    """Decide which muon_lr the two arms train at, and say where it came from.

    `abl-arch` is sequenced immediately after `sweep`, whose entire purpose is
    to produce a trustworthy lr -- but nothing used to carry that number
    across: `_cli()` never populated `run_abl_arch`'s `extra_train_args`, so
    both arms trained at train.py's default 0.02 no matter what the sweep
    said. The first sweep picked 0.01, so this was live rather than
    theoretical, and it fails silently: the head-to-head stays internally
    valid (both twins share the lr) but its val BPB no longer reflects the
    regime `hero` will actually train in, and results.json recorded nothing
    about which lr produced it.

    Returns (extra_train_args, provenance). A None lr from both sources
    leaves train.py's own default in charge and is reported as such, rather
    than being silently re-encoded here -- one default is enough.
    """
    if muon_lr is not None and best_json:
        raise ValueError("pass --muon-lr or --best-json, not both: the "
                         "explicit value would silently shadow the swept one")
    if muon_lr is not None:
        return ["--muon-lr", repr(float(muon_lr))], {
            "muon_lr": float(muon_lr), "source": "--muon-lr"}
    if best_json:
        with open(best_json) as f:
            data = json.load(f)
        lr = data.get("best_lr")
        if lr is None:
            raise ValueError(f"{best_json!r} has no 'best_lr' key -- did the "
                             "sweep finish? Every probe failing leaves the "
                             "file without a winner.")
        prov = {"muon_lr": float(lr), "source": best_json,
                "sweep_git_commit": data.get("git_commit"),
                "sweep_best_val_bpb": data.get("best_val_bpb")}

        # When the grid does not discriminate, follow the blueprint, not the
        # noise. `scripts/check_sweep.py` already detects this case -- and only
        # *warns*, exit 0, so the chain proceeds and hands `best_lr` straight to
        # a ~$11 ablation and then to a ~$44 hero. Measured tonight, the first
        # two probes came in 0.43% apart (1.09178 at lr 0.01 against 1.087067 at
        # 0.02), inside this threshold, so the likeliest outcome of the sweep is
        # a winner that means nothing.
        #
        # DAEDALUS-BLUEPRINT-v6.md locks Muon lr 0.02 (SS36 "Muon lr 0.02
        # (sweep {0.01, 0.02, 0.04})", SS93). A tie is not evidence for
        # deviating from a locked decision -- it is evidence the probe budget
        # cannot tell these lrs apart -- so the blueprint keeps precedence and
        # the sweep only overrides it when it has actually measured something.
        #
        # Falls through to train.py's own default rather than re-encoding 0.02
        # here: one default is enough, and `test_blueprint_default_lr_is_train_py_default`
        # pins that it is still the blueprint's. The winner is kept in the
        # provenance either way, so results.json records what the sweep said as
        # well as what was trained.
        margin = _winner_margin(data)
        if margin is not None and margin < NOISE_FRAC:
            prov.update({
                "muon_lr": None,
                "source": "train.py default (blueprint 0.02)",
                # `source` is overwritten above, so keep the path too --
                # otherwise results.json records that the blueprint won without
                # recording which sweep it beat.
                "sweep_file": best_json,
                "sweep_best_lr": float(lr),
                "sweep_winner_margin": margin,
                "why": (f"the winner beats the runner-up by only "
                        f"{margin*100:.2f}%, inside the {NOISE_FRAC*100:.1f}% "
                        f"noise floor, so the sweep did not discriminate; "
                        f"keeping the blueprint's locked lr rather than a "
                        f"winner chosen by noise")})
            print(f"=== sweep tie (winner beats runner-up by "
                  f"{margin*100:.2f}% < {NOISE_FRAC*100:.1f}%): ignoring "
                  f"best_lr={lr} and keeping the blueprint's 0.02 ===",
                  flush=True)
            return [], prov
        prov["sweep_winner_margin"] = margin
        return ["--muon-lr", repr(float(lr))], prov
    return [], {"muon_lr": None, "source": "train.py default"}


def _release_gpu() -> None:
    """Return this process's cached CUDA blocks to the driver.

    Never raises: it is called between arms of a ~$11 job, and a cleanup step
    is not allowed to be what ends one. A no-op without torch or without CUDA.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:                                   # pragma: no cover
        print(f"WARNING: could not release GPU memory ({e!r})", flush=True)


def run_abl_arch(mixture_dir: str, configs: List[str], total_tokens: int,
                 holdout_frac: float, split_root: str, out_path: str,
                 llama_cpp_dir: str, ppl_text_file: Optional[str] = None,
                 device: str = "cuda", wandb_enabled: bool = True,
                 bench_n_gen: int = 128,
                 extra_train_args: Optional[List[str]] = None,
                 lr_provenance: Optional[dict] = None,
                 run_prefix: str = "abl-arch") -> dict:
    from daedalus.data import make_mixture_holdout_split

    train_root = os.path.join(split_root, "train")
    holdout_root = os.path.join(split_root, "holdout")
    make_mixture_holdout_split(mixture_dir, train_root, holdout_root,
                               holdout_frac=holdout_frac)
    # Weight val_bpb the way the sampler draws, not by holdout shard sizes.
    # Degrades to the old holdout-token weighting rather than raising: this
    # runs before either arm trains, and a refinement to how a metric is
    # averaged must never be what ends a ~$11, ~24 h unattended job.
    try:
        val_weights = mixture_sampling_weights(train_root, total_tokens)
    except Exception as e:
        print(f"WARNING: falling back to holdout-token val weights ({e})",
              flush=True)
        val_weights = None

    # Both arms log a bounded held-out bpb *during* training, not only once at
    # the end. Two reasons, the second larger than the first:
    #
    #  - The ablation is decided on val_bpb, and without this an arm runs ~12 h
    #    showing nothing but training loss on the phone dashboard. A curve that
    #    is flat, NaN or at chance is worth catching at step 500 (~$0.05 in) and
    #    not at the end of the arm (~$5.40 in).
    #  - `hero.py` has passed --val-dir since 298c059, so the four-day, ~$43.70
    #    run depends on a code path that has never executed on a GPU -- the
    #    holdout is a *mixture root*, and `evaluate_bpb_mixture` exists only
    #    because plain `evaluate_bpb` raised on one. `Trainer._val_bpb` swallows
    #    every exception by design, so a mistake here does not fail loudly: it
    #    logs `val_bpb: null` behind a WARNING for the whole run. abl-arch
    #    exercises it first, on the same hardware, for free.
    #
    # Memory-safe by construction on the thin-margin dense arm (measured 29.55
    # of 32.6 GB at micro-batch 16, the least headroom anywhere in the plan):
    # validation is forward-only under no_grad at val_batch_size 8, never more
    # rows than the micro-batch the preflight picks from 16/12/8, and it retains
    # no activations -- so an eval pass allocates strictly less than the training
    # step that just finished, whichever micro-batch wins.
    #
    # Identical for both arms, like every other train flag here: the head-to-head
    # is only publishable if the arms differ in architecture alone. Note this is
    # a bounded per-source sample for the live curve; the number the ablation is
    # decided on stays `eval_val_bpb`'s full holdout pass (max_batches=None).
    train_args = list(extra_train_args or []) + ["--val-dir", holdout_root]

    runs: Dict[str, dict] = {}
    for config in configs:
        run_name = f"{run_prefix}-{config}"
        entry = {"config": config, "run_name": run_name}
        try:
            ckpt_path = run_training(train_root, config, total_tokens, run_name,
                                     wandb_enabled, train_args,
                                     report=entry)
            entry["checkpoint"] = ckpt_path
            entry.update(eval_val_bpb(ckpt_path, holdout_root, config, device,
                                      weights=val_weights))
            entry["export"] = export_and_bench(
                ckpt_path, config, os.path.join("runs", run_name, "export"),
                llama_cpp_dir, ppl_text_file=ppl_text_file, bench_n_gen=bench_n_gen)
        except Exception as e:
            # Catches eval/export failures too, not just a failed training
            # subprocess -- a checkpoint that trained fine but whose eval or
            # export step crashes must still leave results.json written with
            # whatever it has, not lose the (expensive) completed training.
            entry["error"] = str(e)
        runs[config] = entry
        summary = {k: v for k, v in entry.items() if k != "per_source_val_bpb"}
        print(f"=== {config}: {json.dumps(summary, indent=2)} ===", flush=True)

        # Hand the GPU back before the next arm trains.
        #
        # `eval_val_bpb` and `export_and_bench` run in *this* process, so their
        # allocations stay in PyTorch's caching allocator after they return --
        # reserved from the driver's point of view, invisible to the next arm.
        # This process then blocks in `subprocess.run` waiting for that arm,
        # still holding them.
        #
        # Measured, on 2026-08-10: after arm 1's export this process held
        # **4.18 GB** while arm 2 trained, leaving 620 MiB free on a 32.6 GB
        # card. Arm 1 never paid that cost -- nothing had been exported when it
        # ran -- so the dense arm was training with 4.18 GB less than the arm it
        # is being compared against, and it needs *more*: 18 attention blocks
        # against the hybrid's 6. It OOM'd twice and died, and recovering it
        # meant killing this process to free the memory (see
        # scripts/run_dense_arm.sh and scripts/finish_dense_arm.py).
        #
        # An arm that is measured on a smaller GPU than the arm it is compared
        # against is not a controlled experiment, so this is a correctness fix
        # for the ablation, not only a robustness one.
        _release_gpu()

    result = {"runs": runs, "mixture_dir": mixture_dir,
             "total_tokens_per_run": total_tokens, "configs": configs,
             "lr": lr_provenance or {"muon_lr": None,
                                     "source": "train.py default"}}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_path}")
    return result


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--mixture-dir", default="data/shards",
                   help="dataprep.py mixture root (one subdir per source)")
    p.add_argument("--configs", default="daedalus-150m,dense-150m")
    p.add_argument("--total-tokens", type=int, default=5_000_000_000)
    p.add_argument("--holdout-frac", type=float, default=0.02)
    p.add_argument("--split-root", default="data/shards-abl-split")
    p.add_argument("--out", default="runs/abl-arch/results.json")
    p.add_argument("--llama-cpp-dir", default=os.environ.get("LLAMA_CPP_DIR", "vendor/llama.cpp"))
    p.add_argument("--ppl-text-file", default=None,
                   help="text file for the fp16-vs-Q4_0 perplexity check and "
                        "the CPU decode-speed benchmark; both are skipped "
                        "without it")
    p.add_argument("--bench-n-gen", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--muon-lr", type=float, default=None,
                   help="muon lr for BOTH arms; omit to use train.py's default")
    p.add_argument("--best-json", default=None,
                   help="sweep.py output (e.g. runs/sweep/best-wsdfix.json); "
                        "both arms train at its 'best_lr'. Mutually exclusive "
                        "with --muon-lr.")
    p.add_argument("--run-prefix", default="abl-arch",
                   help="run dirs are runs/<prefix>-<config>. Change it for a "
                        "smoke run: the retry path resumes from "
                        "runs/<prefix>-<config>/checkpoint.pt if one exists, so "
                        "a leftover checkpoint from a throwaway run would be "
                        "silently picked up by a real arm's second attempt and "
                        "poison a ~$11 head-to-head. sweep.py has the same flag "
                        "for the same reason.")
    p.add_argument("--micro-batch", type=int, default=None,
                   help="gradient-accumulation granularity for BOTH arms. "
                        "dense-150m needs ~33%% more activation memory than "
                        "the hybrid (24 attention layers vs 6, FF 24x2304 vs "
                        "18x2048), so the batch that fits one may not fit the "
                        "other; scripts/preflight_batch.sh measures it. Must "
                        "be the same for both arms -- it sets `accum`, which "
                        "sets tokens/step, which sets the step count, and "
                        "'same steps' is the property that makes the "
                        "head-to-head publishable.")
    args = p.parse_args()
    configs = args.configs.split(",")
    extra_train_args, lr_provenance = resolve_muon_lr(args.muon_lr, args.best_json)
    if args.micro_batch is not None:
        extra_train_args = extra_train_args + ["--micro-batch", str(args.micro_batch)]
        lr_provenance["micro_batch"] = args.micro_batch
    print(f"=== muon_lr: {lr_provenance} ===", flush=True)
    result = run_abl_arch(
                args.mixture_dir, configs, args.total_tokens, args.holdout_frac,
                args.split_root, args.out, args.llama_cpp_dir,
                ppl_text_file=args.ppl_text_file, device=args.device,
                wandb_enabled=not args.no_wandb, bench_n_gen=args.bench_n_gen,
                extra_train_args=extra_train_args, lr_provenance=lr_provenance,
                run_prefix=args.run_prefix)

    # results.json is written either way -- a completed arm must never be lost
    # because its twin failed -- but the *exit status* has to tell the truth.
    # Exiting 0 with both arms errored makes a total failure look like a
    # finished job to the chain log, the heartbeat and anything gating on it,
    # and this is the last unattended step of a ~25 h night.
    failed = sorted(c for c, r in result.get("runs", {}).items() if "error" in r)
    if failed:
        print(f"=== FAILED arms: {', '.join(failed)} "
             f"(results.json still written to {args.out}) ===", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    _cli()

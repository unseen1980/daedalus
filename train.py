"""Daedalus training loop (AGENT.md SS3, item 2).

Muon + AdamW (daedalus/muon.py), WSD lr with linear decay-to-zero, staged
batch-tokens/seq-len ramps, bf16 autocast + optional torch.compile, gradient
accumulation, checkpoint/resume, metrics to runs/<RUN_NAME>/metrics.jsonl,
and W&B logging that degrades to offline instead of crashing a run.

Resume contract: a checkpoint captures model + Muon + AdamW state and the
absolute step/tokens_seen counters, so continuing training after a restart
reproduces an uninterrupted run bit-for-bit *given the same subsequent
batches* (see tests/test_train.py). It does NOT replay the exact shuffled
data order from a real sharded dataset after a restart -- only a fresh
shuffle from the resumed step. That's a deliberate scope cut: real-world
large-scale trainers (llm.c, nanochat) make the same trade, because
functional correctness (no corrupted state, training continues improving)
is what actually matters, not bit-identical data replay across a restart.

What that scope cut must NOT mean is re-serving the *same* windows. The
samplers are therefore positioned by `tokens_seen` (`set_position`, called
once after a resume): a restart draws a fresh, deterministic, and
*different* index stream, rather than restarting the old one from index 0.
Before that existed, a `hero` interrupted at the halfway mark spent its
whole second half on the first half's data -- a silent quality loss with
nothing wrong in any log. See `ShardBatchSource.set_position`.
"""
import argparse
import contextlib
import io
import json
import math
import os
import random
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, replace as dataclass_replace
from typing import Dict, List, Optional, Sequence, Tuple

# Must run before `import torch` -- torch._inductor.config reads this env var
# at import time to size its parallel compile-worker pool (one process per
# CPU core by default, ~470 MB each -- 16 workers stacked with a concurrent
# job's own RSS caused the near-miss in STATUS.md's "Near-miss" section).
# Capping to 4 keeps the worst-case compile-time spike to ~1.9 GB instead of
# ~7.5 GB, without materially slowing single-shape compiles like this one.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")

import torch

from daedalus.config import PRESETS, DaedalusConfig
from daedalus.model import Daedalus
from daedalus import ckpt_uploader
from daedalus import qat as qat_mod
from daedalus.muon import (build_optimizers, conv_proj_wd_schedule,
                           decay_start_step, momentum_warmup, wsd_lr)
from daedalus.wandb_logger import WandbLogger

DEFAULT_COST_PER_HOUR = 0.449  # this box (RTX 5090 incl. storage); see COSTS.md


# --------------------------------------------------------------- schedules ---

def ramp_progress(tokens_seen: int, ramp_tokens: int) -> float:
    """Fraction of a token-budget ramp completed, clamped to [0, 1]."""
    if ramp_tokens <= 0:
        return 1.0
    return min(1.0, max(0.0, tokens_seen / ramp_tokens))


def seq_len_schedule(tokens_seen: int, ramp_tokens: int, start: int = 1024,
                     end: int = 2048, round_to: int = 128) -> int:
    """Linear seq-len ramp, snapped to a coarse grid so torch.compile only
    has to hold a handful of distinct shapes (runs/smoke found compile's own
    memory overhead -- not activation memory -- is what gates large batches)."""
    p = ramp_progress(tokens_seen, ramp_tokens)
    raw = start + p * (end - start)
    lo, hi = min(start, end), max(start, end)
    snapped = round(raw / round_to) * round_to
    return int(min(hi, max(lo, snapped)))


def batch_tokens_schedule(tokens_seen: int, ramp_tokens: int,
                          start: int = 128_000, end: int = 512_000) -> int:
    """Linear ramp of the *effective* tokens per optimizer step, realized via
    gradient accumulation over a fixed micro-batch size."""
    p = ramp_progress(tokens_seen, ramp_tokens)
    return int(start + p * (end - start))


def grad_accum_steps(target_tokens: int, micro_batch: int, seq_len: int) -> int:
    per_micro = micro_batch * seq_len
    return max(1, round(target_tokens / per_micro))


def estimate_total_steps(total_tokens: int, ramp_tokens: int, micro_batch: int,
                         seq_start: int, seq_end: int, tok_start: int,
                         tok_end: int) -> int:
    """Exact number of optimizer steps a `total_tokens` budget will take, by
    replaying the same seq/batch ramp the training loop uses.

    This must be exact, because it is the `total` passed to `wsd_lr` and
    therefore decides where the decay phase starts and how far it gets. The
    obvious shortcut -- `total_tokens / mean(tok_start, tok_end)` -- is wrong
    by ~1.42x: the batch-token ramp finishes after only `ramp_frac` (10%) of
    the budget, so the run spends ~90% of its tokens at `tok_end`, not at the
    midpoint. Overestimating the step count that way made every run stop at
    ~0.66 of peak lr instead of decaying to zero, which is the entire point
    of WSD (D2Z, arXiv 2502.15938) -- measured, not theorised: see
    tests/test_train.py::test_wsd_schedule_actually_decays_to_zero.
    """
    seen, steps = 0, 0
    while seen < total_tokens:
        seq = seq_len_schedule(seen, ramp_tokens, seq_start, seq_end)
        batch_tokens = batch_tokens_schedule(seen, ramp_tokens, tok_start, tok_end)
        seen += micro_batch * seq * grad_accum_steps(batch_tokens, micro_batch, seq)
        steps += 1
    return max(1, steps)


# --------------------------------------------------------------- checkpoint ---

def save_checkpoint(path: str, model, muon, adamw, step: int, tokens_seen: int,
                    cfg: DaedalusConfig, extra: Optional[dict] = None,
                    save_optimizer: bool = True,
                    weights_dtype: Optional[torch.dtype] = None) -> str:
    """Write a checkpoint atomically.

    `weights_dtype` casts the *floating-point* weights on the way out (integer
    buffers are left alone). Used for the ~2 h Hub copy: bf16 halves 642 MB to
    321 MB, which is what AGENT.md SS0.4's "weights-only (~300MB) at intervals"
    asks for. Never used for the local rolling checkpoint or the milestone --
    those stay fp32, because they are what a run actually resumes from.
    """
    state = qat_mod.strip_qat_state_dict(model.state_dict())
    if weights_dtype is not None:
        state = {k: (v.to(weights_dtype) if v.is_floating_point() else v)
                 for k, v in state.items()}
    payload = {
        # QAT (last ~5% of hero) renames every quantized weight to
        # `...parametrizations.weight.original`. Strip it back so the artifact
        # keeps the model's normal key layout and stays loadable by a plain
        # model -- resume must work in exactly the window where an interruption
        # costs the most.
        "model": state,
        "step": step,
        "tokens_seen": tokens_seen,
        "config": asdict(cfg),
        "extra": extra or {},
    }
    if save_optimizer:
        payload["muon"] = muon.state_dict()
        payload["adamw"] = adamw.state_dict()
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)  # never leave a half-written checkpoint on crash
    return path


def load_checkpoint(path: str, model, muon=None, adamw=None,
                    map_location: str = "cpu") -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if muon is not None and "muon" in payload:
        muon.load_state_dict(payload["muon"])
    if adamw is not None and "adamw" in payload:
        adamw.load_state_dict(payload["adamw"])
    return {
        "step": payload["step"],
        "tokens_seen": payload["tokens_seen"],
        "config": payload.get("config", {}),
        "extra": payload.get("extra", {}),
    }


# ------------------------------------------------------------------ metrics ---

def run_dir_for(args) -> str:
    """Where a run writes, from its args alone.

    Named because two sides have to agree on it and only one of them used to
    know it. A supervised launcher hands `run_with_resume` a checkpoint path so
    it can resume from it; if that path is not the one the `Trainer` writes, the
    marker sits beside a file that never appears, every relaunch starts from
    step 0, and nothing anywhere reports a problem. Resolving it in one place
    lets a caller ask instead of assume.
    """
    return args.run_dir or os.path.join("runs", args.run_name)


def checkpoint_path_for(args) -> str:
    """The rolling checkpoint a run writes and resumes from."""
    return os.path.join(run_dir_for(args), "checkpoint.pt")


def append_metrics(run_dir: str, record: dict) -> str:
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "metrics.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return path


def bits_per_byte(loss_nats: float, n_tokens: int, n_bytes: int) -> float:
    """Held-out bits-per-byte: nats -> bits, normalized by raw text bytes
    rather than token count, so it's comparable across tokenizers/vocab
    sizes (AGENT.md SS3 eval.py spec; also logged during training as val_bpb).
    """
    if n_bytes <= 0:
        return float("nan")
    return (loss_nats * n_tokens) / (n_bytes * math.log(2))


# ------------------------------------------------------------------- wandb ---
# WandbLogger lives in daedalus/wandb_logger.py (torch-free -- see that
# module's docstring for why) and is imported above so existing callers
# (including tests) can keep using `train.WandbLogger`.

WANDB_ID_FILE = "wandb-run-id.txt"

# Total absolute deviation from the blueprint mixture, in percentage points,
# above which the run says so loudly. Chosen to be quiet at every budget the
# corpus was built for (3.97-3.99 pts from 10B to 40B, essentially all of it
# the recorded everyday-conversations deviation) and loud at the first budget
# that breaks it (29.91 pts at 50B).
MAX_MIXTURE_SKEW_PTS = 10.0

# Multiple of the rolling-upload cadence above which "nothing has reached the
# Hub" is reported as a problem rather than logged as a number. Healthy
# staleness sawtooths up to 1x the cadence, so 3x is clear of the peak.
HUB_STALE_FACTOR = 3.0


def resolve_wandb_run_id(run_dir: str, resumed: bool) -> Tuple[str, str]:
    """Pick the W&B run id for this process. Returns `(run_id, resume_mode)`.

    The distinction being drawn is **"the supervisor restarted me mid-run"**
    vs **"someone started a fresh run that happens to reuse the name"**, and
    the two want opposite behaviour:

    - *Resumed* (`supervise.run_with_resume` relaunched `train.py --resume`
      after a crash): re-attach to the same run, so the URL in STATUS.md and
      the hero gate issue keeps updating and the loss curve stays continuous.
      Without this every crash in `hero`'s ~92 h silently strands the operator
      on a dead URL while training carries on elsewhere.
    - *Fresh*: mint a new id. This project has a concrete case of why that
      matters -- `sweep` was thrown out and re-run under the same run names
      after the WSD bug. Re-attaching there would have appended the good curve
      to the discarded one at the *same step numbers*, producing one chart
      that silently interleaves two different experiments.

    A resume with no id file (the run dir was rebuilt, or a checkpoint was
    restored from the Hub onto a clean box -- the disaster-recovery case)
    cannot re-attach to a run whose id is unknown, so it mints a fresh one and
    says so rather than failing.

    Note a resumed run re-logs the steps between the checkpoint and the last
    point W&B received; W&B keeps the earlier value for a repeated step, so a
    restart shows as a brief flat spot rather than as a fork in the history.
    """
    path = os.path.join(run_dir, WANDB_ID_FILE)
    if resumed:
        try:
            with open(path) as fh:
                saved = fh.read().strip()
            if saved:
                print(f"W&B: re-attaching to run id {saved} (resumed)")
                return saved, "allow"
        except FileNotFoundError:
            pass
        print(f"WARNING: resumed but no {WANDB_ID_FILE} in {run_dir}; "
              f"starting a new W&B run (history will not be continuous)")
    run_id = uuid.uuid4().hex[:8]
    os.makedirs(run_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(run_id)
    os.replace(tmp, path)
    return run_id, "allow"


# -------------------------------------------------------------- batch source ---

def stream_seed(base_seed: int, position: int) -> int:
    """Deterministic (base_seed, stream position) -> torch seed, via splitmix64.

    Deliberately *not* `hash((base_seed, position))`: Python randomizes hashing
    per process (PYTHONHASHSEED) for str/bytes, and while int hashing happens to
    be stable today, a resume is by definition a new process -- the one place a
    process-dependent seed would silently stop being reproducible. splitmix64 is
    pure integer arithmetic and gives the same answer in every process, on every
    machine, forever. Adjacent positions must also land far apart, so that a
    resume at `tokens_seen` and a resume at `tokens_seen + 1` don't draw nearly
    the same windows; splitmix64's avalanche is what buys that.
    """
    mask = (1 << 64) - 1
    x = (int(base_seed) * 0x9E3779B97F4A7C15 + int(position)) & mask
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & mask
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & mask
    x ^= x >> 31
    return x & 0x7FFF_FFFF_FFFF_FFFF   # torch.Generator.manual_seed wants >= 0


class SyntheticBatchSource:
    """Random-token batches for smoke/dev runs and tests -- no data on disk."""

    def __init__(self, vocab_size: int, micro_batch: int, device: str, seed: int = 0):
        self.vocab_size = vocab_size
        self.micro_batch = micro_batch
        self.device = device
        self.seed = seed
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    def set_position(self, tokens_seen: int) -> None:
        """See `ShardBatchSource.set_position`. Random tokens carry no
        information either way, but keeping the interface uniform across all
        three sources means the Trainer never has to ask what kind it has."""
        self.gen = torch.Generator(device="cpu").manual_seed(
            stream_seed(self.seed, tokens_seen))

    def get_batch(self, seq_len: int) -> torch.Tensor:
        x = torch.randint(0, self.vocab_size, (self.micro_batch, seq_len),
                          generator=self.gen)
        return x.to(self.device)


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


class ShardBatchSource:
    """Wraps PackedTokenDataset/make_loader, rebuilding the DataLoader when
    the scheduled seq_len changes. See the module docstring for the resume
    scope cut on data-order replay.

    Every rebuild advances a monotone stream position and re-derives the
    sampler seed from it (`stream_seed`), so no two segments of a run draw the
    same window indices. `set_position` jumps that counter to `tokens_seen`,
    which is what makes a resume continue rather than repeat.

    num_workers defaults to 0: reads are cheap memmap slices (no image
    decoding or heavy per-sample work), so multi-process workers add no
    speed here and a worker pool wrapped by an infinite generator (`_cycle`)
    that's never explicitly closed can trip a `PyGILState_Release` error at
    interpreter shutdown. Single-process loading sidesteps that entirely.
    """

    def __init__(self, shard_dir: str, micro_batch: int, device: str,
                num_workers: int = 0, seed: int = 0):
        self.shard_dir = shard_dir
        self.micro_batch = micro_batch
        self.device = device
        self.num_workers = num_workers
        self.seed = seed
        self._stream = 0
        self._seq_len = None
        self._it = None

    def set_position(self, tokens_seen: int) -> None:
        """Move the sampler to the stream position implied by `tokens_seen`.

        The Trainer calls this once, after a resume. Without it `_rebuild`
        re-seeded from the constant `self.seed`, so a restarted run replayed
        the same window indices it had already trained on -- `hero` is a
        multi-day run that *will* be interrupted, and one interrupted at the
        halfway mark would have spent its entire second half re-reading the
        first half's data. Nothing in the loss curve or any log would have
        looked wrong; it would just have quietly produced a worse model.

        Positions are token counts (billions) while within-run rebuilds
        increment by one from zero, so a resumed segment cannot collide with
        the segments that preceded it.
        """
        self._stream = int(tokens_seen)
        self._seq_len = None      # force a rebuild before the next batch

    def _rebuild(self, seq_len: int) -> None:
        from daedalus.data import make_loader
        self._stream += 1
        g = torch.Generator().manual_seed(stream_seed(self.seed, self._stream))
        loader = make_loader(self.shard_dir, seq_len, self.micro_batch,
                             shuffle=True, num_workers=self.num_workers,
                             generator=g)
        self._seq_len = seq_len
        self._it = _cycle(loader)

    def get_batch(self, seq_len: int) -> torch.Tensor:
        if seq_len != self._seq_len:
            self._rebuild(seq_len)
        return next(self._it).to(self.device)


def cap_weights_by_epochs(weights: Dict[str, float],
                          tokens_on_disk: Dict[str, int],
                          total_run_tokens: int,
                          max_epochs: float = 4.0) -> Dict[str, float]:
    """Clamp mixture shares so no source is repeated more than `max_epochs`.

    A source's target share is a statement about the *mixture*, not about how
    much data exists. Sampling by share alone means an under-built source is
    silently repeated as many times as it takes to fill its quota. That is not
    hypothetical here: `everyday-conversations` is exhausted at 403,573 tokens
    (~2.2k rows is the whole dataset), so a 2% share of a 40B-token run would
    draw it 800M times over -- roughly 2,000 epochs of the same 2,000
    conversations.

    Repetition up to ~4 epochs is close to free (Muennighoff et al. 2023,
    arXiv 2305.16264: "training with up to 4 epochs of repeated data yields
    negligible changes to loss compared to having unique data"); past that the
    returns decay to zero and heavy repetition of a tiny slice starts doing
    real damage. So each source is capped at
    `max_epochs * tokens_on_disk / total_run_tokens` and the freed mass is
    redistributed, by water-filling, over the sources that still have headroom.

    If *every* source is capped -- the whole corpus is too small for this run,
    not one source lagging -- the target shares are returned **unchanged**, with
    a warning. In that regime you cannot both hit the mixture and bound
    repetition, and mixture balance is the one that matters more: uniform
    over-repetition degrades gracefully, whereas renormalizing to the caps would
    reproduce whatever skew happens to be on disk (right now that would mean
    training on 21% FinePhrase and 5% FineWeb-Edu instead of 7% and 37.5%). The
    warning is the signal to go build more data, and it must not be silent.
    """
    caps = {
        name: (max_epochs * tokens_on_disk.get(name, 0) / total_run_tokens
               if total_run_tokens > 0 else float("inf"))
        for name in weights
    }
    total = sum(weights.values())
    free = {n: w / total for n, w in weights.items()}
    capped: Dict[str, float] = {}

    while free:
        budget = 1.0 - sum(capped.values())
        pool = sum(free.values())
        if pool <= 0:
            break
        scaled = {n: w / pool * budget for n, w in free.items()}
        over = [n for n, w in scaled.items() if w > caps[n]]
        if not over:
            capped.update(scaled)
            break
        for n in over:
            capped[n] = caps[n]
            del free[n]

    target = {n: w / total for n, w in weights.items()}
    if sum(capped.values()) < 1.0 - 1e-9:
        detail = ", ".join(
            f"{n} {target[n] * total_run_tokens / tokens_on_disk[n]:.1f}x"
            for n in sorted(target) if tokens_on_disk.get(n))
        print(f"WARNING: the corpus is too small for a {total_run_tokens:,}-token "
             f"run at {max_epochs} epochs/source -- EVERY source is over the "
             f"limit, so the target mixture is kept as-is and repetition is "
             f"accepted rather than reweighting to whatever is on disk. "
             f"Implied epochs: {detail}. Build more data.")
        return target
    return capped


def resolve_mixture(data_root: str,
                    total_run_tokens: Optional[int] = None,
                    max_epochs: float = 4.0,
                    weights: Optional[Dict[str, float]] = None,
                    verbose: bool = True):
    """Decide what mixture will actually be sampled from `data_root`.

    Factored out of `MixtureBatchSource.__init__` so a *preflight* can ask the
    question without building samplers. `hero.py` needs the answer before it
    spends $61.89 and six days, and the only way to get it used to be to
    construct the real loader -- which opens a memmap per source and needs a
    device. Two implementations of "what will this run train on" is exactly the
    drift that makes a preflight worthless, so there is one.

    Returns `(names, target_probs, probs, tokens_on_disk)`, where `probs` is
    post-cap and `target_probs` is the renormalized blueprint mixture.
    """
    if weights is None:
        from daedalus.dataprep import MIXTURE
        weights = {s.key: s.share for s in MIXTURE}
    present = {name: w for name, w in weights.items()
               if os.path.exists(os.path.join(data_root, name, "manifest.json"))}
    if not present:
        raise ValueError(
            f"no source under {data_root!r} has a manifest.json "
            f"(looked for {sorted(weights)})")
    missing = sorted(set(weights) - set(present))
    if missing and verbose:
        print(f"MixtureBatchSource: {missing} not found under {data_root!r}, "
              f"renormalizing remaining weights over {sorted(present)}")
    total_weight = sum(present.values())
    names = sorted(present)
    probs = {n: present[n] / total_weight for n in names}
    target_probs = dict(probs)

    tokens_on_disk: Dict[str, int] = {}
    if total_run_tokens:
        for name in names:
            with open(os.path.join(data_root, name, "manifest.json")) as f:
                tokens_on_disk[name] = int(json.load(f)["total_tokens"])
        if verbose:
            capped = cap_weights_by_epochs(probs, tokens_on_disk,
                                           total_run_tokens, max_epochs)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                capped = cap_weights_by_epochs(probs, tokens_on_disk,
                                               total_run_tokens, max_epochs)
        for name in names:
            before, after = probs[name], capped[name]
            if abs(after - before) > 1e-6 and verbose:
                disk = tokens_on_disk[name]
                epochs = before * total_run_tokens / disk if disk else float("inf")
                print(f"MixtureBatchSource: {name} share "
                      f"{before:.4f} -> {after:.4f} (only {disk:,} tokens on "
                      f"disk; uncapped share would be {epochs:.1f} epochs, "
                      f"limit {max_epochs})")
        probs = capped
    return names, target_probs, probs, tokens_on_disk


def summarize_mixture(names, target_probs, probs, tokens_on_disk,
                      total_run_tokens, max_epochs) -> Dict[str, object]:
    """The body of `MixtureBatchSource.mixture_summary()`, callable without a
    sampler. See that method's docstring for what each field means and why
    `max_epochs_seen` exists alongside `l1_skew_pts`."""
    eff = dict(probs) if isinstance(probs, dict) else dict(zip(names, probs))
    per_source = {}
    for n in names:
        disk = tokens_on_disk.get(n)
        target = target_probs[n]
        row = {"target_share": round(target, 6),
               "effective_share": round(eff[n], 6),
               "tokens_on_disk": disk,
               "capped": False}
        if disk and total_run_tokens:
            epochs = eff[n] * total_run_tokens / disk
            row["epochs"] = round(epochs, 3)
            row["capped"] = epochs >= max_epochs - 1e-6
        per_source[n] = row
    capped = sorted(n for n, r in per_source.items() if r["capped"])
    epochs_seen = {n: r["epochs"] for n, r in per_source.items() if "epochs" in r}
    worst = max(epochs_seen, key=epochs_seen.get) if epochs_seen else None
    return {
        "per_source": per_source,
        "capped_sources": capped,
        "l1_skew_pts": round(
            100.0 * sum(abs(eff[n] - target_probs[n]) for n in names), 4),
        "max_epochs_seen": round(epochs_seen[worst], 3) if worst else None,
        "most_repeated_source": worst,
        "total_tokens_on_disk": sum(tokens_on_disk.values()) or None,
        "total_run_tokens": total_run_tokens,
        "max_epochs": max_epochs,
    }


def mixture_preflight(data_root: str, total_run_tokens: int,
                      max_epochs: float = 4.0,
                      weights: Optional[Dict[str, float]] = None,
                      verbose: bool = False) -> Dict[str, object]:
    """`mixture_summary()` for a corpus, without building the loader.

    Reads one `manifest.json` per source and nothing else, so it is cheap
    enough to run at launch and needs no GPU. `hero.py` gates on it.
    """
    names, target_probs, probs, on_disk = resolve_mixture(
        data_root, total_run_tokens, max_epochs, weights, verbose=verbose)
    return summarize_mixture(names, target_probs, probs, on_disk,
                             total_run_tokens, max_epochs)


def parse_mixture_weights(pairs: Optional[Sequence[str]]
                          ) -> Optional[Dict[str, float]]:
    """`["fineweb-edu=0.7", ...]` as shares, or None when none were given.

    The shares must sum to 1. `resolve_mixture` renormalizes anyway, so the
    check buys nothing arithmetically -- it buys the one error renormalization
    hides. A phase-7 arm is three or four shares typed on a command line, and a
    dropped source or a mistyped digit produces a set that sums to 0.85 and is
    then quietly rescaled into a mixture nobody chose, with the arm's own
    artifact recording the weights it *asked* for. Requiring the sum makes that
    a refusal at launch instead of a result at the end.
    """
    if not pairs:
        return None
    weights: Dict[str, float] = {}
    for pair in pairs:
        name, sep, raw = str(pair).partition("=")
        name = name.strip()
        if not sep or not name:
            raise ValueError(f"expected NAME=FRACTION, got {pair!r}")
        if name in weights:
            raise ValueError(f"--mixture-weight {name!r} given twice")
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(f"share for {name!r} is not a number: {raw!r}") from None
        if value < 0 or not math.isfinite(value):
            raise ValueError(f"share for {name!r} must be finite and >= 0, got {value}")
        weights[name] = value
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"--mixture-weight shares sum to {total:.6f}, not 1.0. They are "
            f"renormalized over whatever is on disk, so a set that does not sum "
            f"to 1 is usually a source left out rather than a mixture asked for: "
            f"{dict(sorted(weights.items()))}")
    return weights


class MixtureBatchSource:
    """Samples whole micro-batches from multiple per-source shard directories
    by mixture proportion, so `abl-arch`/`hero` can train on the real data
    mixture -- `dataprep.py` writes one shard set per source under
    `out_root/<source_key>/`, never merged (see STATUS.md's "mixture-loader
    gap" note). Each `get_batch()` call draws its whole micro-batch from one
    source (weighted random choice), never mixing rows from different
    sources within a batch -- keeps `ShardBatchSource`'s contract (a single
    flat `[seq_len]` tensor per row) unchanged.

    `data_root` must contain one subdirectory per source, each with its own
    `manifest.json`, as written by `daedalus.dataprep.run_dataprep`. Weights
    default to `daedalus.dataprep.MIXTURE`'s shares, restricted to sources
    actually present under `data_root` and renormalized to sum to 1 -- so a
    partially-built mixture (e.g. dataprep still running, or a substituted
    source dropped) trains on whatever's available rather than crashing.

    Same resume scope cut as `ShardBatchSource`/the module docstring: the
    sequence of source picks is a `random.Random` stream, not checkpointed, so
    a resume reproduces training given the same subsequent batches but not the
    exact pre-restart mixture-and-shuffle order. `set_position` re-seeds it,
    and every per-source sampler under it, from `tokens_seen`, so a resume
    draws new data rather than repeating what it already trained on.
    """

    def __init__(self, data_root: str, micro_batch: int, device: str,
                weights: Optional[Dict[str, float]] = None,
                num_workers: int = 0, seed: int = 0,
                total_run_tokens: Optional[int] = None,
                max_epochs: float = 4.0):
        # Shared with `mixture_preflight()` so what hero.py gates on and what
        # this actually samples cannot diverge. `target_probs` keeps the
        # pre-cap ask so `mixture_summary()` can report it next to the
        # effective share -- otherwise the only record of a cap biting is a
        # print() at startup, which on a six-day run scrolls past once and
        # reaches neither W&B nor STATUS.md.
        (self.names, self.target_probs, probs,
         self.tokens_on_disk) = resolve_mixture(
            data_root, total_run_tokens, max_epochs, weights)
        self.total_run_tokens = total_run_tokens
        self.max_epochs = max_epochs
        self.probs = [probs[n] for n in self.names]
        self.sources = {
            name: ShardBatchSource(os.path.join(data_root, name), micro_batch,
                                   device, num_workers=num_workers, seed=seed + i)
            for i, name in enumerate(self.names)
        }
        self.seed = seed
        self.rng = random.Random(seed)

    def mixture_summary(self) -> Dict[str, object]:
        """What this sampler will actually draw, next to what was asked for.

        Reported into the W&B run config so the mixture a run trained on is
        recoverable from the dashboard rather than only from a startup line in
        a log file. That distinction is not academic: a single DNS failure on
        2026-08-10 left `dclm-baseline` at 1.57B of 2.25B, and because
        `cap_weights_by_epochs` clamps a short source to `4 x on_disk /
        run_tokens`, a 40B run would have sampled 53.2% web against a 60%
        target -- the shortfall water-filled onto code and finephrase. It
        printed one line and raised nothing. See
        `runs/preflight/mixture-cap-vs-hero-budget.md`.

        `l1_skew_pts` is the total absolute deviation from the target mixture
        in percentage points, so one number on the dashboard says whether the
        corpus is delivering the blueprint's mixture at this run's size.
        """
        # "Capped" means pinned at the epoch limit -- this source has no more
        # data to give -- not merely "its share moved". Those differ:
        # water-filling *raises* the share of sources with headroom, so keying
        # off `eff != target` would report a source as capped for absorbing
        # someone else's shortfall. A capped source can also sit above its own
        # target if it absorbed mass before hitting its own ceiling, so
        # comparing against `target` is wrong in both directions.
        #
        # The mixture has *two* failure modes and `l1_skew_pts` only sees one.
        # When the cap binds it reweights, and the skew number rises. When no
        # allocation can satisfy the cap at all -- every source over the limit,
        # `cap_weights_by_epochs`'s all-capped fallback -- the target shares are
        # returned unchanged, so the skew is **0.00 by construction**: its best
        # possible value at the one budget where repetition is bounded by
        # nothing. That is hero at 60B against today's 14.218B corpus, where
        # `everyday-conversations` takes its full 2% from a dataset exhausted at
        # 403,573 tokens -- ~2,973 epochs of ~2,200 conversations.
        #
        # So report the worst repetition directly. When the cap works, a capped
        # source is pinned at exactly `max_epochs` and every other source sits
        # below it, so `max_epochs_seen > max_epochs` is true precisely in the
        # unbounded case and never otherwise.
        return summarize_mixture(self.names, self.target_probs,
                                 dict(zip(self.names, self.probs)),
                                 self.tokens_on_disk, self.total_run_tokens,
                                 self.max_epochs)

    def set_position(self, tokens_seen: int) -> None:
        """Position every per-source sampler *and* the source-pick stream (see
        `ShardBatchSource.set_position`). Each sub-source was built with its own
        `seed + i`, so mixing the same position into each still gives them
        distinct streams."""
        for src in self.sources.values():
            src.set_position(tokens_seen)
        self.rng = random.Random(stream_seed(self.seed, tokens_seen))

    def get_batch(self, seq_len: int) -> torch.Tensor:
        name = self.rng.choices(self.names, weights=self.probs, k=1)[0]
        return self.sources[name].get_batch(seq_len)


# --------------------------------------------------------------------- git ---

# This function runs **inside the training loop**, synchronously, every ~10
# minutes of a multi-day run. "Never raises" is therefore not enough: a hang is
# not an exception, and the `except` below cannot see one. An unbounded
# `git push` that black-holes -- a dropped TCP connection with no RST, or the
# `gh auth git-credential` helper wedging -- would freeze the whole loop with
# the GPU idle for the remainder of the run, silently, because the very thing
# that would have reported it is the commit that is stuck.
#
# Measured on this box: `git push` to origin round-trips in 0.70-0.82 s; the
# local commands are ~3 ms each.
#
# The two bounds are deliberately asymmetric:
#
#  * The push is the only step that touches the network, so it gets the bound
#    that matters (300 s, ~370x measured), plus git's own low-speed abort so a
#    stalled *transfer* ends as a clean error the `except` handles rather than
#    as a SIGKILL. `GIT_TERMINAL_PROMPT=0` turns a credential prompt into an
#    immediate failure instead of a wait on a stdin nobody is typing into.
#  * The local commands get a very wide bound (120 s, ~40,000x measured)
#    because killing them is the risky direction: a SIGKILLed `git commit` can
#    leave `.git/index.lock` behind, which would break every later git
#    operation in this repo -- including the heartbeat and the auto-commit
#    supervisor. They cannot stall on the network, so the bound is a backstop
#    against a wedged filesystem and nothing else.
GIT_LOCAL_TIMEOUT_S = 120.0
GIT_PUSH_TIMEOUT_S = 300.0
# Abort a push whose transfer drops below 1 KB/s for 60 s. Complements the hard
# timeout: this catches a stall mid-transfer cleanly, the timeout catches one
# that never gets as far as HTTP (DNS, connect, or a wedged credential helper).
_GIT_PUSH_STALL_OPTS = ["-c", "http.lowSpeedLimit=1000",
                        "-c", "http.lowSpeedTime=60"]


def git_commit_and_push(repo_dir: str, message: str, paths) -> bool:
    """Best-effort: stage `paths`, commit if there's anything staged, push.
    Never raises, and never blocks indefinitely -- a git/network hiccup must
    not kill a training run, and must not stall one either (see above).

    Silently skips paths that don't exist yet (e.g. metrics.jsonl before the
    first metrics write) rather than failing the whole `git add` and losing
    legitimate changes to the paths that do exist.
    """
    existing = [p for p in paths if os.path.exists(os.path.join(repo_dir, p))]
    if not existing:
        return False
    try:
        subprocess.run(["git", "add", *existing], cwd=repo_dir, check=True,
                       capture_output=True, timeout=GIT_LOCAL_TIMEOUT_S)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo_dir,
                              capture_output=True, timeout=GIT_LOCAL_TIMEOUT_S)
        if diff.returncode == 0:
            return False  # nothing staged
        subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo_dir,
                       check=True, capture_output=True,
                       timeout=GIT_LOCAL_TIMEOUT_S)
        subprocess.run(["git", *_GIT_PUSH_STALL_OPTS, "push", "-q", "origin", "main"],
                       cwd=repo_dir, check=True, capture_output=True,
                       timeout=GIT_PUSH_TIMEOUT_S,
                       env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
        return True
    except Exception as e:
        # TimeoutExpired lands here too, so a wedged push costs one publish
        # interval and a warning line, not the run. The commit it made is
        # local and the next interval pushes it.
        print(f"WARNING: git commit/push failed ({e}); continuing")
        return False


class IntervalGate:
    """Fires at most once per `interval_sec` of wall clock -- used to pace
    checkpointing/git-push/Hub-upload without threading a clock through the
    training loop by hand."""

    def __init__(self, interval_sec: float, clock=time.time):
        self.interval_sec = interval_sec
        self.clock = clock
        self.last = None

    def ready(self) -> bool:
        now = self.clock()
        if self.last is None or now - self.last >= self.interval_sec:
            self.last = now
            return True
        return False


# ---------------------------------------------------------------- training ---

@dataclass
class TrainArgs:
    run_name: str
    config: str = "daedalus-150m"
    data_dir: Optional[str] = None       # None -> synthetic random-token data
    total_tokens: int = 5_000_000_000
    max_steps: Optional[int] = None      # overrides total_tokens for smoke/tests
    micro_batch: int = 16
    seq_start: int = 1024
    seq_end: int = 2048
    tok_start: int = 128_000
    tok_end: int = 512_000
    ramp_frac: float = 0.1
    muon_lr: float = 0.02
    adam_lr: float = 3e-4
    # None = the shipped single-Muon-group split. See `build_optimizers`.
    conv_proj_wd: Optional[float] = None
    # Phase 5's two varying arms. `conv_proj_wd_end=None` is a constant, which
    # is what the shipped 0.1 and the weak-constant arm are, so leaving these
    # alone reproduces the existing behaviour exactly.
    conv_proj_wd_end: Optional[float] = None
    conv_proj_wd_ramp_frac: float = 0.0
    conv_proj_wd_hold_frac: float = 0.0
    warmup_steps: int = 300
    decay_frac: float = 0.45
    # Config knobs a *run* may override without editing the preset. Both default
    # to None meaning "whatever `PRESETS[config]` says", so every existing run
    # keeps the shipped 1024-token loss chunk and no block checkpointing.
    #
    # They are per-run rather than per-preset because they are memory/throughput
    # trades, not model definition: `to_hf_dict` already strips both, so two
    # runs that differ only here export byte-identical configs. A QAT recovery
    # run is the case that needs them -- the fake-quant parametrization adds a
    # dequantized copy of every linear weight to the forward graph, so the
    # activation headroom that fitted the original pretraining run no longer
    # does at the same batch shape.
    loss_chunk_size: Optional[int] = None
    gradient_checkpointing: Optional[bool] = None
    # Which tokenizer produced the ids in `--data-dir`/`--val-dir`. `None` means
    # the preset's own value, which is `None` -- SmolLM2 -- for every shipped
    # preset, so no existing run changes. It is stripped by `to_hf_dict` like
    # the two above, so it does not reach the exported config either.
    tokenizer: Optional[str] = None
    grad_clip: float = 1.0
    run_dir: Optional[str] = None
    ckpt_every_sec: float = 1800.0       # 30 min, per AGENT.md hero spec
    push_every_sec: float = 600.0        # 10 min, per AGENT.md SS0.1/SS5.3
    metrics_every_steps: int = 20
    log_every_steps: int = 20
    compile: bool = True
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_enabled: bool = True
    # Whether fit() closes the W&B run when it returns. post.py sets this
    # False because a DPO round follows the SFT loop in the same process:
    # finishing here made every DPO metric land on a closed run and be dropped
    # with a warning, so the operator's dashboard just stopped mid-job.
    finish_wandb: bool = True
    cost_per_hour: float = DEFAULT_COST_PER_HOUR
    resume: Optional[str] = None
    # Load *weights only* and start a new run at step 0 with fresh optimizer
    # state and a fresh WSD schedule. This is what fine-tuning means, and it is
    # emphatically not what `resume` means: `resume` restores step, tokens_seen
    # and both optimizers so an interrupted run continues where it stopped.
    #
    # Passing hero's checkpoint as `resume` -- which post.py did -- restores
    # step=610000/tokens_seen=40e9 into a run whose budget is far smaller, so
    # fit() breaks at the top of its very first iteration and the whole SFT
    # stage silently does nothing (measured: 0 steps, lr multiplier 0.000000,
    # and a milestone upload fired on step 0). Nothing raises.
    #
    # Both may be set at once, and `resume` wins when its checkpoint exists:
    # that is exactly what a crash-restart of a fine-tune wants -- continue the
    # fine-tune if it got anywhere, otherwise start it again from the base
    # weights -- and it lets a supervisor relaunch the same command line
    # unchanged.
    init_from: Optional[str] = None
    tags: Optional[List[str]] = None
    max_source_epochs: float = 4.0   # see cap_weights_by_epochs
    # Explicit per-source shares for a mixture root, or None for
    # `dataprep.MIXTURE`'s blueprint. Phase 7 compares mixtures under equal
    # compute, which needs the mixture to be an argument of the *run* rather
    # than a constant of the corpus: every other way of varying it -- a second
    # data root per arm, an edited MIXTURE, a patched loader -- also varies
    # something that is supposed to be held.
    #
    # None, not the blueprint's shares, so every existing run resolves its
    # mixture exactly where it did before and a mixture root that is missing a
    # source keeps renormalizing over what is present.
    mixture_weights: Optional[Dict[str, float]] = None
    # Held-out bits-per-byte during training (AGENT.md SS5.2 lists val_bpb as a
    # required metrics field). Without it a multi-day run has no generalization
    # signal at all -- watchdog.py can only see the training loss, which looks
    # healthy right up until it doesn't.
    val_dir: Optional[str] = None        # holdout shard dir; None disables
    val_every_steps: int = 500
    val_batches: int = 8                 # bounded sample, not a full pass
    val_batch_size: int = 8
    # Quantization-aware training against llama.cpp's exact Q4_0 grid, over the
    # final `qat_frac` of the run (blueprint: ~5%, i.e. 0.05). 0 disables it.
    # Every job except `hero` must leave this at 0 -- a quantized forward would
    # invalidate `sweep`'s lr comparison and `abl-arch`'s hybrid-vs-dense result.
    qat_frac: float = 0.0
    # Hub checkpoint durability (AGENT.md SS0.2/SS0.4). Nothing on this box
    # survives a recycle, so a multi-day run that only checkpoints locally is
    # uninsured. Off when `hub_repo` is None, which keeps tests and smoke runs
    # offline by default.
    hub_repo: Optional[str] = None
    hub_every_sec: float = 7200.0        # ~2 h weights-only rolling copy
    hub_poll_sec: float = 300.0          # uploader's own directory-poll cadence
    hub_uploader: bool = True            # spawn the out-of-band upload process
    # Consecutive skipped updates before `fit` raises `NonFiniteStall` rather
    # than spinning. A transient bad batch recovers in one or two steps; a run
    # that has skipped this many in a row is not going to finish, and neither
    # of fit()'s break conditions can fire while it keeps skipping.
    max_consecutive_skips: int = 25

    def __post_init__(self):
        # A ramp without a group to ramp is the same silent no-op
        # `build_optimizers` already refuses for `conv_proj_wd` itself: the run
        # looks configured, trains the shipped schedule, and only the arm's
        # result says otherwise -- by which point the GPU hours are spent.
        if self.conv_proj_wd is None and (
                self.conv_proj_wd_end is not None
                or self.conv_proj_wd_ramp_frac
                or self.conv_proj_wd_hold_frac):
            raise ValueError(
                "conv_proj_wd_end/ramp_frac/hold_frac need conv_proj_wd set: "
                "without it there is no conv-projection group to schedule and "
                "the ramp would silently do nothing")
        if not 0.0 <= self.conv_proj_wd_hold_frac <= self.conv_proj_wd_ramp_frac <= 1.0:
            raise ValueError(
                f"conv-proj-wd ramp must satisfy 0 <= hold_frac "
                f"({self.conv_proj_wd_hold_frac}) <= ramp_frac "
                f"({self.conv_proj_wd_ramp_frac}) <= 1")
        if self.conv_proj_wd_end is not None and self.conv_proj_wd_ramp_frac == 0.0:
            raise ValueError(
                "conv_proj_wd_end is set but conv_proj_wd_ramp_frac is 0, so "
                "the decay would jump to the end value at step 0; pass a ramp "
                "fraction, or drop --conv-proj-wd-end for a constant")


class NonFiniteStall(RuntimeError):
    """Every recent step was skipped, so the run cannot make progress.

    `train_step` returns early on a non-finite loss *without* advancing `step`
    or `tokens_seen` -- correct, because a skipped update trained nothing and
    should not be billed against the budget. But `fit`'s two break conditions
    are `step >= max_steps` and `tokens_seen >= total_tokens`, and neither can
    ever be reached if every step is skipped. The loop then spins: it burns
    GPU, appends an identical metrics row each time, and never ends.

    Measured on the first Phase 3 smoke run, whose released-checkpoint weights
    produced a NaN loss from step one (see `qat._safe_reciprocal`): 2,794
    skipped updates and 0.18 GPU-hours before it was killed by hand. Nothing in
    the process would have stopped it -- `--max-steps 3` was set.

    Raising hands the decision to the supervisor, which is the component that
    knows whether to retry, halt, or move to the next arm. `watchdog.py` would
    also have caught this eventually, but only a supervised run has one, and
    "eventually" is the wrong unit for a loop that cannot progress.
    """


class NoOpResume(RuntimeError):
    """A resume that would train nothing, raised instead of exiting 0.

    `fit()` breaks at the top of its first iteration when
    `tokens_seen >= total_tokens`, which is correct for a run that has
    genuinely finished and is being relaunched by `supervise`/`boot_resume`.
    It is *also* what happens when someone points a fresh run at a foreign
    checkpoint under too small a budget -- and there the process prints
    `resumed from ...`, writes no metrics row, and exits 0. Measured twice
    on this project:

    - `post.py` passed `hero`'s checkpoint as `resume` and its whole SFT
      stage silently did nothing (see `TrainArgs.init_from`).
    - the model card's branch-from-milestone command omitted `--total-tokens`,
      so it inherited the 5e9 default against a 30.5e9-token milestone
      (`runs/preflight/branch-command.md`).

    The two cases are told apart by whether this run directory has any
    history of its own: a crash-resume continues a run that has been writing
    `metrics.jsonl` for hours, a mistaken branch starts in an empty one. That
    discriminator is what keeps this off `hero`'s own resume path.
    """


def _config_for(args: TrainArgs) -> DaedalusConfig:
    """`PRESETS[args.config]` with this run's memory overrides applied.

    Returns the preset object itself when nothing is overridden, so the common
    case allocates nothing and `Trainer(...).cfg is PRESETS[name]` stays true
    for every existing caller and test.
    """
    cfg = PRESETS[args.config]
    overrides = {}
    if args.loss_chunk_size is not None:
        if args.loss_chunk_size < 0:
            raise ValueError(
                f"--loss-chunk-size must be >= 0 (0 disables chunking), "
                f"got {args.loss_chunk_size}")
        overrides["loss_chunk_size"] = args.loss_chunk_size
    if args.gradient_checkpointing is not None:
        overrides["gradient_checkpointing"] = bool(args.gradient_checkpointing)
    # `None` keeps the preset's own value, which is `None` -- SmolLM2 -- for
    # every shipped preset. Phase 4's probes pass the candidate vocabulary they
    # were packed under, so `_val_bpb` decodes the holdout with the tokenizer
    # that produced its ids rather than with SmolLM2 regardless.
    if args.tokenizer is not None:
        overrides["tokenizer"] = args.tokenizer
    return dataclass_replace(cfg, **overrides) if overrides else cfg


class Trainer:
    def __init__(self, args: TrainArgs):
        self.args = args
        self.run_dir = run_dir_for(args)
        self.ckpt_path = checkpoint_path_for(args)

        torch.manual_seed(args.seed)
        # `dataclasses.replace`, never mutation: `PRESETS` holds one shared
        # `DaedalusConfig` instance per name, so assigning to `self.cfg.<field>`
        # would rewrite the preset for every later `Trainer`, `eval.py` and
        # `export.py` in the same process. The overrides below are exactly the
        # fields `to_hf_dict` strips, so the exported config is unaffected.
        self.cfg = _config_for(args)
        self.model = Daedalus(self.cfg).to(args.device)
        self.muon, self.adamw, self.opt_stats = build_optimizers(
            self.model, muon_lr=args.muon_lr, adam_lr=args.adam_lr,
            conv_proj_wd=getattr(args, "conv_proj_wd", None))

        self.step = 0
        self.tokens_seen = 0
        self.start_time = time.time()
        self._peak_mem_seen = 0.0
        self._tokenizer = None          # lazily loaded, only if val_dir is set
        # Cumulative count of optimizer updates dropped because the loss was
        # non-finite. A Phase 3 gate reads "no skipped non-finite updates", and
        # the only previous evidence was a WARNING line in a log nobody keeps:
        # every skip logged a `loss: nan` row indistinguishable from a row the
        # metrics interval simply happened to land on. Carried in the durable
        # record so the gate can be decided from `metrics.jsonl` alone.
        self._skipped_updates = 0
        # Skips since the last update that actually landed. Reset on any
        # successful step, so an occasional bad batch never accumulates toward
        # the stall limit -- only an inability to make progress does.
        self._consecutive_skips = 0

        # `hub://owner/repo/path?rev=branch` is materialised to a local file
        # first, so a restore from the Hub takes exactly the same code path as
        # a restore from disk -- there is no separate, less-tested branch for
        # the case that only ever runs after the box has been lost.
        resume_path = ckpt_uploader.resolve_resume(
            args.resume, os.path.join(self.run_dir, "restored"),
            token=os.environ.get("HF_TOKEN_WRITE"))
        self._resumed = False
        if resume_path and os.path.exists(resume_path):
            info = load_checkpoint(resume_path, self.model, self.muon, self.adamw,
                                   map_location=args.device)
            self.step = info["step"]
            self.tokens_seen = info["tokens_seen"]
            # Carry the skip count across the restart. Resetting it to 0 would
            # let a run that skipped updates before a crash report a clean
            # `skipped_updates: 0` afterwards, and the Phase 3 finiteness gate
            # reads exactly that field.
            self._skipped_updates = int(
                (info.get("extra") or {}).get("skipped_updates", 0))
            self._resumed = True
            print(f"resumed from {args.resume}: step={self.step} "
                 f"tokens_seen={self.tokens_seen}")
            self._refuse_noop_resume()
        elif args.init_from:
            # Weights only: `muon`/`adamw` are deliberately not passed, so the
            # optimizers keep the fresh state built above and step/tokens_seen
            # stay at 0. See TrainArgs.init_from.
            init_path = ckpt_uploader.resolve_resume(
                args.init_from, os.path.join(self.run_dir, "init"),
                token=os.environ.get("HF_TOKEN_WRITE"))
            if not (init_path and os.path.exists(init_path)):
                # Unlike `resume`, which is routinely absent on a first launch,
                # `init_from` is an explicit "start from these weights". Falling
                # back to random init would fine-tune a fresh model for hours
                # and report success, so this is fatal on purpose.
                raise FileNotFoundError(
                    f"--init-from checkpoint not found: {args.init_from}")
            load_checkpoint(init_path, self.model, map_location=args.device)
            print(f"initialized weights from {args.init_from}: "
                 f"step=0 tokens_seen=0, fresh optimizer state and LR schedule")

        # tok_per_sec is windowed since the last metrics log, not cumulative --
        # must be (re)based on tokens_seen *after* a possible resume above, or
        # the first post-resume log would report a bogus multi-day-average spike.
        self._last_log_tokens = self.tokens_seen
        self._last_log_time = self.start_time
        # Which step the last metrics row covers, and the stats of the last
        # completed step, so `fit` can force a final row without duplicating
        # one the interval already wrote. None until the first of each.
        self._last_metrics_step: Optional[int] = None
        self._last_stats: Optional[dict] = None

        self.net = torch.compile(self.model) if args.compile else self.model

        ramp_tokens = int(args.total_tokens * args.ramp_frac) if args.max_steps is None \
            else max(1, args.max_steps // 4) * args.micro_batch * args.seq_start
        self.ramp_tokens = max(ramp_tokens, 1)

        # Cached, not recomputed per step: it's a pure function of args, and
        # replaying an 87k-step ramp on every train_step would cost more than
        # the step itself.
        self.total_steps = args.max_steps if args.max_steps is not None else \
            estimate_total_steps(args.total_tokens, self.ramp_tokens,
                                 args.micro_batch, args.seq_start, args.seq_end,
                                 args.tok_start, args.tok_end)

        is_mixture_root = bool(args.data_dir) and not os.path.exists(
            os.path.join(args.data_dir, "manifest.json"))
        if args.mixture_weights and not is_mixture_root:
            # Refused rather than ignored. Every other batch source samples one
            # corpus, so weights handed to one are a no-op -- and a no-op here
            # is a phase-7 arm that reports itself as `only-stack-edu-python`,
            # trains on whatever the single directory held, and produces a
            # perfectly finite BPB for a mixture it never sampled. Same silent
            # no-op TrainArgs.__post_init__ already refuses for a conv-proj
            # weight-decay ramp with no group to ramp.
            raise ValueError(
                f"--mixture-weight was given but --data-dir "
                f"{args.data_dir or '<none>'} is not a mixture root "
                f"(a directory of per-source subdirectories, each with its own "
                f"manifest.json). The weights would be silently ignored.")
        if args.data_dir:
            if not is_mixture_root:
                self.batch_source = ShardBatchSource(args.data_dir, args.micro_batch,
                                                     args.device, seed=args.seed)
            else:
                # No manifest.json directly under data_dir -> treat it as a
                # mixture root (one subdirectory per source, each with its
                # own manifest.json), auto-detected so callers don't need a
                # separate CLI flag for the single-source vs. mixture case.
                self.batch_source = MixtureBatchSource(
                    args.data_dir, args.micro_batch, args.device, seed=args.seed,
                    weights=args.mixture_weights,
                    total_run_tokens=args.total_tokens,
                    max_epochs=args.max_source_epochs)
        else:
            self.batch_source = SyntheticBatchSource(self.cfg.vocab_size,
                                                      args.micro_batch, args.device,
                                                      seed=args.seed)

        # A resumed run must not re-serve the windows the pre-restart segment
        # already trained on -- position the sampler by tokens_seen. No-op at
        # tokens_seen == 0, so a fresh run is unaffected.
        if self.tokens_seen:
            self.batch_source.set_position(self.tokens_seen)

        self._qat_on = qat_mod.is_qat_active(self.model)

        self.ckpt_gate = IntervalGate(args.ckpt_every_sec)
        self.push_gate = IntervalGate(args.push_every_sec)

        # --- Hub durability (AGENT.md SS0.2) -------------------------------
        self.outbox = ckpt_uploader.outbox_dir(self.run_dir)
        # Fires immediately on the first call, so a run that dies in its first
        # two hours still leaves something on the Hub. The rolling copy is
        # cheap; waiting for the first full interval is a gap for no reason.
        self.hub_gate = IntervalGate(args.hub_every_sec)
        self.milestone_step = decay_start_step(self.total_steps, args.decay_frac)
        self.milestone_path = os.path.join(self.run_dir, "milestone.json")
        # Read off disk rather than kept in memory: hero.py restarts train.py
        # after a crash, and a milestone written before the crash must not be
        # written a second time at a step that is no longer the branch point.
        self._milestone_done = os.path.exists(self.milestone_path)
        self._milestone_gate = IntervalGate(600.0)   # backoff between retries
        # One line per run, not one per 30-minute gate: a diverged run keeps
        # looping until the watchdog halts it, and a repeated ERROR would bury
        # the non-finite loss row that says what actually happened.
        self._diverged_logged = False
        # Re-warn about a hub stall at most once per extra hour, so a genuine
        # outage stays visible across a four-day run without every metrics
        # line repeating it.
        self._hub_warned_at_h = 0.0
        self.uploader_proc = None

        wandb_config = {"train_args": asdict(args), "model_config": asdict(self.cfg),
                        "optimizer_split": self.opt_stats}
        # Put the sampled mixture on the dashboard. Best-effort on purpose:
        # W&B reporting must never be able to stop a training run (AGENT.md
        # §5.1), and this is the last thing added before the run starts.
        try:
            if isinstance(self.batch_source, MixtureBatchSource):
                summary = self.batch_source.mixture_summary()
                wandb_config["data_mixture"] = summary
                if summary["capped_sources"]:
                    print(f"MixtureBatchSource: mixture L1 skew "
                          f"{summary['l1_skew_pts']:.2f} pts from target; capped: "
                          f"{summary['capped_sources']}")
                # Graded, because the line above is not: at hero's intended 40B
                # the skew is 3.99 pts and at 50B it is 29.91, and both printed
                # the same shape of line. The 50B case is the dangerous one --
                # the epoch cap binds hard and the web backbone collapses
                # (fineweb-edu 37.5% -> 30.0%, dclm 22.5% -> 18.0%, finephrase
                # nearly doubling) -- and it is reached by raising a *token
                # budget*, a lever that looks unrelated to data. "It went well,
                # let's extend it" is the natural request after a good hero.
                # See runs/preflight/mixture-vs-token-budget.md.
                if summary["l1_skew_pts"] > MAX_MIXTURE_SKEW_PTS:
                    print(f"WARNING: the sampled mixture is {summary['l1_skew_pts']:.2f} "
                          f"pts from the blueprint target (limit "
                          f"{MAX_MIXTURE_SKEW_PTS} pts). The corpus is too small "
                          f"for {summary['total_run_tokens']:,} tokens at "
                          f"{summary['max_epochs']} epochs/source, so capped "
                          f"sources are being replaced by whatever has headroom. "
                          f"Mixture balance matters more to benchmarks than "
                          f"token count -- reduce the budget or build more data.")
                # Graded separately, because the check above cannot see this
                # one: in the all-capped fallback the skew is 0.00 -- perfect --
                # while repetition is bounded by nothing at all. 40B is quiet
                # (worst source 4.00 epochs) and 60B against today's corpus is
                # loud (2,973). See runs/preflight/mixture-at-60b.md and
                # issue #5.
                seen = summary.get("max_epochs_seen")
                if seen is not None and seen > summary["max_epochs"] + 1e-6:
                    print(f"WARNING: repetition is UNBOUNDED -- "
                          f"'{summary['most_repeated_source']}' is sampled "
                          f"{seen:,.1f} times over against a {summary['max_epochs']}-"
                          f"epoch limit. No allocation can satisfy the cap at "
                          f"{summary['total_run_tokens']:,} tokens over "
                          f"{summary['total_tokens_on_disk']:,} on disk, so the "
                          f"target mixture is kept and nothing bounds repeats. "
                          f"l1_skew_pts reads {summary['l1_skew_pts']:.2f} here "
                          f"and is not measuring this. Top up the corpus or "
                          f"lower the budget before spending the run.")
        except Exception as e:                                # pragma: no cover
            print(f"WARNING: could not summarize the data mixture for W&B ({e!r})")
        # Kept on the Trainer so it is assertable without a live W&B run --
        # WandbLogger deliberately keeps no state when disabled.
        self.wandb_config = wandb_config
        # Re-attach to the same W&B run when this process is a supervisor
        # restart, so a crash mid-`hero` does not strand the operator on a
        # frozen URL. See resolve_wandb_run_id.
        self.wandb_run_id, wandb_resume = resolve_wandb_run_id(
            self.run_dir, self._resumed)
        self.wandb = WandbLogger(
            project=args.wandb_project or os.environ.get("WANDB_PROJECT", "daedalus"),
            entity=args.wandb_entity or os.environ.get("WANDB_ENTITY"),
            name=f"{args.run_name}",
            config=wandb_config, tags=[args.run_name, *(args.tags or [])],
            enabled=args.wandb_enabled,
            run_id=self.wandb_run_id, resume=wandb_resume,
        )

    def _lr_multiplier(self, total_steps: int) -> float:
        return wsd_lr(self.step, total_steps, warmup=self.args.warmup_steps,
                     decay_frac=self.args.decay_frac)

    def _conv_proj_wd(self, total_steps: int) -> Optional[float]:
        """This step's conv-projection decay, or None when the group does not
        exist.

        Returning None rather than the shipped 0.1 matters: with
        `--conv-proj-wd` unset there *is* no second Muon group, and writing a
        decay into `param_groups[1]` would either raise or -- worse, if the
        index ever moved -- retune the 76.9% of Muon's parameters this
        experiment is supposed to hold fixed.
        """
        args = self.args
        if getattr(args, "conv_proj_wd", None) is None:
            return None
        return conv_proj_wd_schedule(
            self.step, total_steps, args.conv_proj_wd,
            end=getattr(args, "conv_proj_wd_end", None),
            ramp_frac=getattr(args, "conv_proj_wd_ramp_frac", 0.0),
            hold_frac=getattr(args, "conv_proj_wd_hold_frac", 0.0))

    def _estimated_total_steps(self) -> int:
        return self.total_steps

    def train_step(self) -> dict:
        args = self.args
        total_steps = self._estimated_total_steps()
        seq_len = seq_len_schedule(self.tokens_seen, self.ramp_tokens,
                                   args.seq_start, args.seq_end)
        batch_tokens = batch_tokens_schedule(self.tokens_seen, self.ramp_tokens,
                                             args.tok_start, args.tok_end)
        accum = grad_accum_steps(batch_tokens, args.micro_batch, seq_len)

        mult = self._lr_multiplier(total_steps)
        for g in self.muon.param_groups:
            g["lr"] = args.muon_lr * mult
            g["momentum"] = momentum_warmup(self.step, warmup=args.warmup_steps)
        for g in self.adamw.param_groups:
            g["lr"] = args.adam_lr * mult
        conv_wd = self._conv_proj_wd(total_steps)
        if conv_wd is not None:
            self.muon.param_groups[
                self.opt_stats["conv_proj_group_index"]]["weight_decay"] = conv_wd

        self.muon.zero_grad(set_to_none=True)
        self.adamw.zero_grad(set_to_none=True)

        loss_sum = 0.0
        for _ in range(accum):
            batch = self.batch_source.get_batch(seq_len)
            # A source may yield either `x` (pretraining: every token is its
            # own target) or `(x, y)` (SFT: y carries -100 on prompt and pad
            # positions). Keeping both here lets post.py reuse this Trainer --
            # optimizers, WSD schedule, checkpointing, resume -- instead of
            # duplicating the most delicate loop in the project.
            x, y = batch if isinstance(batch, tuple) else (batch, batch)
            with torch.autocast(device_type="cuda" if args.device == "cuda" else "cpu",
                                dtype=torch.bfloat16, enabled=(args.device == "cuda")):
                _, loss, _ = self.net(x, targets=y)
            if not torch.isfinite(loss):
                self._skipped_updates += 1
                print(f"WARNING: non-finite loss at step {self.step}, skipping "
                      f"update ({self._skipped_updates} skipped so far)")
                self.muon.zero_grad(set_to_none=True)
                self.adamw.zero_grad(set_to_none=True)
                return {"step": self.step, "loss": float("nan"), "skipped": True,
                       "seq_len": seq_len, "accum": accum}
            (loss / accum).backward()
            loss_sum += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                    args.grad_clip)
        self.muon.step()
        self.adamw.step()

        self.tokens_seen += args.micro_batch * seq_len * accum
        self.step += 1

        if args.device == "cuda":
            self._peak_mem_seen = max(self._peak_mem_seen,
                                      torch.cuda.max_memory_allocated() / 1e9)

        metrics = {
            "step": self.step, "loss": loss_sum / accum, "skipped": False,
            "seq_len": seq_len, "accum": accum, "lr_mult": mult,
            "grad_norm": float(grad_norm),
        }
        if conv_wd is not None:
            # Logged because a schedule that silently failed to apply is
            # indistinguishable from an arm that did not work, and phase 5
            # decides between arms.
            metrics["conv_proj_wd"] = conv_wd
        return metrics

    def _elapsed_h(self) -> float:
        return (time.time() - self.start_time) / 3600

    def _val_weights(self) -> Optional[Dict[str, float]]:
        """The per-source probabilities this run's sampler draws with, or None
        when training on a single source. Read off `MixtureBatchSource` itself
        rather than recomputed from `dataprep.MIXTURE`, so the epoch cap and
        the renormalization over present-only sources are reflected exactly."""
        src = getattr(self, "batch_source", None)
        names, probs = getattr(src, "names", None), getattr(src, "probs", None)
        if not names or probs is None or len(names) != len(probs):
            return None
        return dict(zip(names, probs))

    def _refuse_noop_resume(self) -> None:
        """See `NoOpResume`. Only fires for a resume into a run directory with
        no metrics history, so a supervisor relaunching a finished run still
        exits 0 quietly the way every caller expects."""
        args = self.args
        if args.max_steps is not None or self.tokens_seen < args.total_tokens:
            return
        if os.path.exists(os.path.join(self.run_dir, "metrics.jsonl")):
            return   # this run's own history: a crash-resume, not a branch
        # Both recovery paths resume from `<run_dir>/checkpoint.pt` -- their own
        # (`supervise.py:311`, `abl_arch.py:110`) -- so a checkpoint from inside
        # this run's directory is a relaunch however little history it left.
        try:
            own = os.path.realpath(self.run_dir) == os.path.realpath(
                os.path.dirname(args.resume or ""))
        except OSError:
            own = False
        if own:
            return
        raise NoOpResume(
            f"--resume restored {self.tokens_seen:,} tokens_seen (step "
            f"{self.step:,}) into a run with no metrics history, but "
            f"--total-tokens is {args.total_tokens:,}. The training loop "
            f"would stop before its first step and exit 0 having trained "
            f"nothing. Pass --total-tokens greater than "
            f"{self.tokens_seen:,} to continue this checkpoint, or "
            f"--init-from instead of --resume to fine-tune from its weights "
            f"at step 0.")

    def _val_bpb(self) -> Optional[float]:
        """Bounded held-out bits-per-byte, or None if validation is off / not
        due / failed. Never raises: a broken holdout dir must not kill a
        multi-day run any more than a W&B outage does."""
        args = self.args
        if not args.val_dir or self.step % args.val_every_steps != 0:
            return None
        try:
            # lazy: eval imports train
            from eval import evaluate_bpb_mixture
            if self._tokenizer is None:
                from daedalus.data import get_tokenizer
                # `cfg.tokenizer` (None for every shipped preset, so this is
                # unchanged for them) rather than always SmolLM2. val_bpb
                # converts nats-per-token through the *bytes those tokens stand
                # for*, and the byte count comes from decoding them -- so a run
                # over a 32,768-vocabulary holdout decoded with SmolLM2 reports
                # bits per byte for a corpus that does not exist.
                self._tokenizer = get_tokenizer(self.model.cfg.tokenizer)
            # self.model, not self.net: the compiled graph is specialized to the
            # training shapes and a different eval batch would force a recompile.
            #
            # Mixture-aware because `hero.py` passes the *root* of a
            # `make_mixture_holdout_split` output, which has no top-level
            # manifest.json -- plain `evaluate_bpb` raised FileNotFoundError
            # there, and the `except` below turned that into `val_bpb: null`
            # for the whole four-day run. Weighted by the source probabilities
            # the sampler actually draws with, so val_bpb estimates BPB under
            # the training distribution rather than under whatever the shard
            # boundaries happened to leave in the holdout.
            return evaluate_bpb_mixture(
                self.model, args.val_dir, args.seq_end,
                self._tokenizer, device=args.device,
                batch_size=args.val_batch_size,
                max_batches=args.val_batches,
                weights=self._val_weights())["val_bpb"]
        except Exception as e:
            print(f"WARNING: val_bpb failed at step {self.step} ({e}); continuing")
            return None

    def _progress(self) -> float:
        """Fraction of the run completed, in [0, 1]."""
        args = self.args
        if args.max_steps is not None:
            return min(1.0, self.step / max(1, args.max_steps))
        return min(1.0, self.tokens_seen / max(1, args.total_tokens))

    def maybe_enable_qat(self) -> bool:
        """Switch the forward pass onto the Q4_0 ship grid for the run's tail.

        Called every step; `enable_qat` is idempotent and the guard is a float
        compare, so the steady-state cost is nothing. Switching on mid-run is
        safe for the optimizers because `parametrize` keeps the same
        `Parameter` objects (proven by `test_qat.py`), so Muon/AdamW moments
        carry over rather than silently resetting at 95% of a four-day run.

        `torch.compile` will recompile once on the step this fires. That is a
        few seconds, once, against ~2 hours of QAT phase.
        """
        if not qat_mod.qat_active_at(self._progress(), self.args.qat_frac):
            return False
        if self._qat_on:
            return False
        applied = qat_mod.enable_qat(self.model)
        self._qat_on = True
        err = qat_mod.quantization_error(self.model)
        print(f"QAT ON at step {self.step} ({self._progress():.1%} of the run): "
              f"{len(applied)} tensors on the llama.cpp "
              f"{qat_mod.grid_id()} grid; "
              f"pre-QAT relative RMSE {err['qat_rel_rmse']:.4f}")
        self.wandb.log({"qat_started_step": self.step,
                        "qat_rel_rmse": err["qat_rel_rmse"],
                        "qat_tensors": err["qat_tensors"],
                        "qat_elements": err["qat_elements"]},
                       step=self.step)
        return True

    def weights_are_finite(self) -> bool:
        """Whether every parameter is still finite.

        One `|=` accumulated on-device and a single sync at the end, rather
        than a `bool()` per parameter -- ~200 GPU->CPU syncs every 30 minutes
        is free, but it is free in the same way that not measuring it would
        have been, and this runs inside the loop.
        """
        bad = None
        with torch.no_grad():
            for p in self.model.parameters():
                flag = ~torch.isfinite(p).all()
                bad = flag if bad is None else (bad | flag)
        return bad is None or not bool(bad)

    def _refuse_if_diverged(self, what: str) -> bool:
        """True when `what` must not be written, because the model is NaN.

        A non-finite model is unrecoverable, so persisting it has no upside --
        and one very real downside. The rolling checkpoint is *the* crash-resume
        artifact and it is overwritten in place on a 30-minute gate, while
        `watchdog.py` needs up to `--stall-min` (30 min) to notice that steps
        have stopped advancing: `train_step` skips the optimizer on a non-finite
        loss without incrementing `self.step`, so once the weights are NaN every
        subsequent step is skipped and `metrics.jsonl` goes quiet. Those two
        30-minute clocks race, and the checkpoint one can win -- turning a
        recoverable NaN at hour 90 of `hero` into a NaN in the only artifact the
        run can resume from, and, two hours later, in the Hub copy as well.

        The last good checkpoint is worth more than a fresh bad one. Refusing
        costs nothing when nothing is wrong (`weights_are_finite` is a single
        pass) and the run is over either way once this fires -- the watchdog
        halts it on the non-finite loss it will already have logged.
        """
        if self.weights_are_finite():
            return False
        if not self._diverged_logged:
            print(f"ERROR: model weights are non-finite at step {self.step}; "
                  f"refusing to overwrite {what} with a diverged model. The "
                  f"last good checkpoint is preserved; watchdog.py will halt "
                  f"this run.")
            self._diverged_logged = True
        return True

    def maybe_checkpoint(self, force: bool = False) -> None:
        if force or self.ckpt_gate.ready():
            if self._refuse_if_diverged("the rolling checkpoint"):
                return
            save_checkpoint(self.ckpt_path, self.model, self.muon, self.adamw,
                            self.step, self.tokens_seen, self.cfg,
                            extra={"skipped_updates": self._skipped_updates})

    # ------------------------------------------------------ Hub durability ---

    def _stage(self, filename: str, path_in_repo: str, revision: str,
               kind: str, save_optimizer: bool,
               weights_dtype: Optional[torch.dtype]) -> str:
        """Write a checkpoint into the outbox and seal it for upload.

        The write is `save_checkpoint`'s usual tmp-then-rename, and only then
        does `stage` emit the sidecar the uploader keys on -- so the uploader
        can never read a half-written file, however it interleaves with us.
        """
        os.makedirs(self.outbox, exist_ok=True)
        path = os.path.join(self.outbox, filename)
        save_checkpoint(path, self.model, self.muon, self.adamw, self.step,
                        self.tokens_seen, self.cfg,
                        save_optimizer=save_optimizer,
                        weights_dtype=weights_dtype)
        ckpt_uploader.stage(
            self.outbox, path, path_in_repo, revision=revision,
            meta={"kind": kind, "step": self.step,
                  "tokens_seen": self.tokens_seen, "run_name": self.args.run_name})
        return path

    def maybe_hub_upload(self, force: bool = False) -> Optional[str]:
        """Stage the ~2 h weights-only rolling copy. Staging is a local write;
        the transfer itself happens in the uploader process, so a slow or hung
        link costs no training time.

        Never raises. A backup is insurance against losing the run -- it must
        not become a way to *end* it. A full disk or a transient I/O error
        during the extra 321 MB write would otherwise take down a four-day job
        that was training perfectly well.
        """
        if not self.args.hub_repo:
            return None
        if not (force or self.hub_gate.ready()):
            return None
        if self._refuse_if_diverged("the rolling Hub checkpoint"):
            return None
        self._ensure_uploader()
        try:
            # Run-scoped path: one repo holds every job's checkpoints, and
            # `abl-arch`'s two arms plus `hero` must not overwrite each other's
            # rolling slot.
            return self._stage(
                f"weights-step{self.step:09d}.pt",
                path_in_repo=f"rolling/{self.args.run_name}/weights.pt",
                revision="rolling", kind="rolling", save_optimizer=False,
                weights_dtype=torch.bfloat16)
        except Exception as e:
            print(f"WARNING: staging rolling hub checkpoint failed ({e}); "
                  f"training continues, next attempt in "
                  f"{self.args.hub_every_sec / 3600:.1f} h")
            return None

    def maybe_milestone(self) -> Optional[str]:
        """At the step WSD leaves the stable phase, write the branch point:
        full fp32 weights *and* both optimizer states, on its own revision.

        Checked at the top of the loop, so `step == milestone_step` means
        exactly "the stable phase is complete, the next update is the first
        decayed one" -- the state a future run would want to branch from.
        `>=` rather than `==` so a resume that lands past the step still
        produces the artifact instead of silently skipping it.
        """
        if self._milestone_done or self.step < self.milestone_step:
            return None
        # A failed attempt must not be retried every step -- each one writes
        # 1.4 GB, so a persistent failure (a full disk being the likely one)
        # would turn into continuous thrashing on top of whatever broke.
        if not self._milestone_gate.ready():
            return None
        # Retried on the gate rather than marked done: the branch point is the
        # one artifact worth waiting for, and a resume from the last good
        # checkpoint may still reach this step with finite weights.
        if self._refuse_if_diverged("the milestone branch point"):
            return None
        revision = f"{self.args.run_name}-stable-end-step{self.step}"
        record = {
            "revision": revision, "step": self.step,
            "tokens_seen": self.tokens_seen,
            "total_steps": self.total_steps,
            "decay_frac": self.args.decay_frac,
            "lr_mult_at_branch": self._lr_multiplier(self.total_steps),
            "muon_lr": self.args.muon_lr, "adam_lr": self.args.adam_lr,
            "config": self.args.config, "repo": self.args.hub_repo,
            "path_in_repo": f"milestone/{self.args.run_name}/checkpoint.pt",
            "created_at": time.time(),
        }
        try:
            if self.args.hub_repo:
                self._stage("milestone-checkpoint.pt",
                            path_in_repo=record["path_in_repo"],
                            revision=revision, kind="milestone",
                            save_optimizer=True, weights_dtype=None)
            else:
                # No Hub configured: still write the branch point locally
                # rather than lose it. The artifact is the point; the upload
                # is transport.
                save_checkpoint(
                    os.path.join(self.run_dir, "milestone-checkpoint.pt"),
                    self.model, self.muon, self.adamw, self.step,
                    self.tokens_seen, self.cfg)
        except Exception as e:
            # Losing the branch point costs future flexibility; ending the run
            # costs the run. Retry on the gate rather than raise.
            print(f"WARNING: milestone checkpoint at step {self.step} failed "
                  f"({e}); will retry, training continues")
            return None
        with open(self.milestone_path, "w") as f:
            json.dump(record, f, indent=2)
        self._milestone_done = True
        print(f"[milestone] end of stable phase at step {self.step} "
              f"({self.tokens_seen:,} tokens) -> revision {revision}")
        self.wandb.log({"milestone_step": self.step}, step=self.step)
        return revision

    def _hub_health(self) -> dict:
        """What the phone needs to tell a working uploader from a stalled one.

        `hub_uploaded_step` is the step of the last checkpoint that actually
        landed on the Hub. If it stops advancing while `step` climbs, uploads
        are failing -- which is otherwise invisible for four days, since every
        upload failure is deliberately non-fatal.
        """
        if not self.args.hub_repo:
            return {}
        health = {"hub_pending": 0, "hub_uploaded_step": None,
                  "hub_lag_steps": None, "hub_stale_h": None}
        try:
            health["hub_pending"] = len(ckpt_uploader.pending_uploads(self.outbox))
            with open(os.path.join(self.outbox,
                                   ckpt_uploader.STATE_FILENAME)) as f:
                uploads = json.load(f).get("uploads", {})
            steps = [u.get("step") for u in uploads.values()
                     if u.get("kind") == "rolling" and u.get("step") is not None]
            if steps:
                health["hub_uploaded_step"] = max(steps)
                health["hub_lag_steps"] = self.step - max(steps)
            stamps = [u.get("uploaded_at") for u in uploads.values()
                      if u.get("uploaded_at")]
            if stamps:
                health["hub_stale_h"] = round((time.time() - max(stamps)) / 3600, 3)
        except (OSError, json.JSONDecodeError, ValueError):
            pass  # nothing uploaded yet, or the state file is mid-write
        self._warn_if_hub_stalled(health)
        return health

    def _warn_if_hub_stalled(self, health: dict) -> None:
        """Grade the staleness, don't just record it.

        These numbers were already logged and nothing read them, which is how
        the 2026-08-10 wedge stayed invisible: the uploader stopped delivering
        at 11:01Z and training carried on reporting a growing `hub_lag_steps`
        as though it were a normal number. Same failure shape as the mixture
        skew before it got a threshold -- a metric on the dashboard is not an
        alert.

        Measured in hours rather than steps because steps/hour changes with the
        batch and sequence ramps, so a step-based bound means different things
        at the start and end of a run. Healthy staleness sawtooths up to
        `hub_every_sec`; 3x that is clear of the peak and still catches a stall
        within a few hours of a 92 h run.
        """
        stale = health.get("hub_stale_h")
        if stale is None:
            return
        limit_h = HUB_STALE_FACTOR * self.args.hub_every_sec / 3600.0
        if stale > limit_h and stale > self._hub_warned_at_h + 1.0:
            self._hub_warned_at_h = stale
            print(f"WARNING: no checkpoint has reached the Hub for {stale:.1f} h "
                  f"(cadence is {self.args.hub_every_sec / 3600:.1f} h, alerting "
                  f"above {limit_h:.1f} h). {health.get('hub_pending', 0)} payload(s) "
                  f"pending. If this persists the run is uninsured: a lost box "
                  f"costs everything since the last upload.")

    def _start_uploader(self) -> None:
        """Spawn the out-of-band uploader. Its absence must never stop
        training, so every failure here is a warning."""
        if not self.args.hub_repo or not self.args.hub_uploader:
            return
        if not os.environ.get("HF_TOKEN_WRITE"):
            print("WARNING: hub_repo set but HF_TOKEN_WRITE is not; "
                  "checkpoints will be staged locally and not uploaded")
            return
        os.makedirs(self.outbox, exist_ok=True)
        cmd = [sys.executable, "-m", "daedalus.ckpt_uploader",
               "--outbox", self.outbox, "--repo", self.args.hub_repo,
               "--interval-s", str(self.args.hub_poll_sec)]
        try:
            self.uploader_proc = subprocess.Popen(cmd)
            print(f"[hub] uploader pid {self.uploader_proc.pid} -> "
                  f"{self.args.hub_repo}")
        except Exception as e:
            print(f"WARNING: could not start ckpt uploader ({e}); continuing")

    def _ensure_uploader(self) -> None:
        """Respawn the uploader if nothing is serving the outbox any more.

        Nothing checked this. `_start_uploader` runs once at startup and the
        handle is never consulted again, so an uploader that dies mid-run --
        OOM-killed, or crashed on a bad payload -- takes the run's insurance
        with it silently. `_warn_if_hub_stalled` would eventually print, but a
        printed warning in a 5.9-day log is not a repair, and AGENT.md SS0.2 is
        blunt about the consequence: if it isn't pushed, it doesn't exist.

        The check has to be `uploader_is_live(outbox)` and not
        `self.uploader_proc.poll()`. After any crash the process actually
        delivering is the orphan the previous attempt left holding the lock, and
        *this* attempt's uploader exited on that lock immediately and by design
        -- so the child handle reads "dead" for the rest of the run while
        uploads are entirely healthy. Respawning on the handle alone would fork
        a redundant uploader every two hours for six days.

        Runs on the same 2 h gate as the upload it protects, so detection
        latency matches the cadence. Never raises: this is insurance, and
        insurance must not be a way to end the run.
        """
        if not self.args.hub_repo or not self.args.hub_uploader:
            return
        if not os.environ.get("HF_TOKEN_WRITE"):
            return
        proc = self.uploader_proc
        if proc is not None and proc.poll() is None:
            return                       # our own child is still running
        try:
            if ckpt_uploader.uploader_is_live(self.outbox):
                return                   # an earlier attempt's orphan is serving it
        except Exception as e:           # pragma: no cover - defensive
            print(f"WARNING: could not check uploader liveness ({e})")
            return
        print("WARNING: no ckpt uploader is serving "
              f"{self.outbox}; respawning. Checkpoints staged since the last "
              f"successful upload are still on disk and will be picked up.")
        self._start_uploader()

    def _stop_uploader(self) -> None:
        """Drain, then stop. The final pass is synchronous on purpose: training
        is over, so blocking costs nothing, and the last checkpoint is the one
        most worth having off the box."""
        proc, self.uploader_proc = self.uploader_proc, None
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        token = os.environ.get("HF_TOKEN_WRITE")
        # No token means no transport, and the pass would be a network call
        # that can only fail. Checking here rather than only in
        # `_start_uploader` is what keeps the test suite (and any smoke run
        # that sets a repo without credentials) genuinely offline.
        if not self.args.hub_repo or not token:
            return
        # Bounded, because this pass is synchronous: a wedged transfer here
        # hangs the *trainer* at the end of a run, so `abl_arch.py` never sees
        # its arm finish and the chain stops with the GPU idle. The 2026-08-10
        # wedge was in the daemon, but this call has the identical failure mode.
        try:
            summary = ckpt_uploader.upload_once_bounded(
                self.outbox, self.args.hub_repo, token=token)
            print(f"[hub] final upload pass: {summary}")
        except Exception as e:
            print(f"WARNING: final hub upload pass failed ({e})")

    def maybe_push(self, force: bool = False) -> None:
        if not (force or self.push_gate.ready()):
            return
        # Only publish runs that live inside this repo.
        #
        # This stages STATUS.md from the repo root, so a run_dir *outside* the
        # repo -- which is every pytest tmp_path -- would commit and push
        # whatever STATUS.md happened to contain at that moment, under that
        # run's name. Not hypothetical: four `test: step 610000, 40000000000
        # tokens` commits reached origin/main this way, one per full-suite run.
        # They came from `test_init_from_actually_trains_where_resume_would_do_
        # nothing`, where fit() breaks immediately (step 610000 >= max_steps 5)
        # so metrics.jsonl is never written and STATUS.md is left as the only
        # staged path. Each one published a mid-edit STATUS.md to main.
        repo = os.path.abspath(os.getcwd())
        run_abs = os.path.abspath(self.run_dir)
        if not (run_abs == repo or run_abs.startswith(repo + os.sep)):
            # Warn rather than skip silently: an unattended run that has quietly
            # stopped publishing is the "alive but doing nothing" failure this
            # project has already been bitten by twice.
            if not getattr(self, "_warned_run_dir_outside_repo", False):
                self._warned_run_dir_outside_repo = True
                print(f"WARNING: run_dir {run_abs} is outside the repo at {repo}; "
                      "skipping git publish for this run")
            return
        git_commit_and_push(
            repo,
            f"{self.args.run_name}: step {self.step}, {self.tokens_seen} tokens",
            # milestone.json rides along because it is the *record* of the
            # branch point, and until now it reached git only when an agent
            # happened to hand-commit it (f8cda13, 18cc9f2 -- both by hand).
            # It is written once, ~3 days into `hero`, and losing it costs
            # more than the 1.4 GB checkpoint it describes:
            #
            #  - a recovery onto a fresh box clones a `runs/hero/` with no
            #    milestone.json, so `_milestone_done` (l.976) reads False and
            #    the milestone **re-fires at the post-decay step it resumed
            #    at**, publishing a checkpoint with lr_mult < 1.0 as the
            #    "end of stable phase" branch point;
            #  - `export.py:_load_milestone` reads this file, so the model
            #    card would either advertise that wrong branch point or, with
            #    the file simply gone, say no branch point exists at all --
            #    losing hard precondition 4's deliverable outright.
            #
            # `git_commit_and_push` skips paths that do not exist, so this is
            # inert until the milestone actually fires.
            [os.path.join(self.run_dir, "metrics.jsonl"),
             os.path.join(self.run_dir, "milestone.json"), "STATUS.md"],
        )

    def log_step(self, stats: dict, force: bool = False) -> None:
        """`force` writes the row regardless of the interval, for the final
        step of a run. Without it the durable record stops at the last multiple
        of `metrics_every_steps`, which is short of the token target unless the
        run happens to end on one -- see `fit`."""
        args = self.args
        # A forced call must not duplicate what the interval already did for
        # this step: two rows for one step double-count in every consumer that
        # replays metrics.jsonl, and a second identical log line reads as though
        # the final step ran twice. Checked first so it suppresses both.
        if force and self._last_metrics_step == self.step:
            return
        if force or self.step % args.log_every_steps == 0:
            print(f"step {stats['step']:7d}  loss {stats['loss']:.4f}  "
                 f"seq {stats['seq_len']:5d}  accum {stats['accum']:3d}  "
                 f"tokens {self.tokens_seen:,}")
        if force or self.step % args.metrics_every_steps == 0:
            now = time.time()
            window_tokens = self.tokens_seen - self._last_log_tokens
            window_sec = max(now - self._last_log_time, 1e-9)
            val_bpb = self._val_bpb()
            record = {
                "step": self.step, "tokens": self.tokens_seen,
                "loss": stats["loss"], "val_bpb": val_bpb,
                "lr": args.muon_lr * stats.get("lr_mult", 1.0),
                "tok_per_sec": window_tokens / window_sec,
                "elapsed_h": self._elapsed_h(),
                "cost_usd": self._elapsed_h() * args.cost_per_hour,
                "grad_norm": stats.get("grad_norm"),
                "peak_mem_GB": self._peak_mem_seen,
                "qat_active": int(self._qat_on),
                "skipped_updates": self._skipped_updates,
            }
            if val_bpb is not None:
                # Which forward produced `val_bpb`. `self.model` carries the
                # fake-quant parametrizations while QAT is on, so val_bpb is
                # measured *through the Q4_0 lattice* then and in fp32
                # otherwise. The two are not comparable, and nothing in the
                # number says which one it is -- a recovery run whose val_bpb
                # sits above the pretraining run's is expected, not a
                # regression. Recorded rather than inferred, because whoever
                # reads metrics.jsonl later will not have `qat_frac` to hand.
                record["val_forward"] = "quantized" if self._qat_on else "float"
                record["val_grid"] = qat_mod.grid_id() if self._qat_on else None
            if self._qat_on:
                # Should fall toward zero as the weights settle onto the grid --
                # the direct evidence QAT is working. O(params), so only while
                # it is actually on.
                err = qat_mod.quantization_error(self.model)
                record["qat_rel_rmse"] = err["qat_rel_rmse"]
                record["qat_tensors"] = err["qat_tensors"]
            record.update(self._hub_health())
            self._last_log_tokens, self._last_log_time = self.tokens_seen, now
            self._last_metrics_step = self.step
            append_metrics(self.run_dir, record)
            self.wandb.log(record, step=self.step)

    def fit(self) -> None:
        args = self.args
        target_tokens = args.total_tokens
        pidfile = os.path.join(self.run_dir, "train.pid")
        os.makedirs(self.run_dir, exist_ok=True)
        with open(pidfile, "w") as f:
            f.write(str(os.getpid()))  # watchdog.py reads this to check liveness
        self._start_uploader()
        try:
            while True:
                if args.max_steps is not None and self.step >= args.max_steps:
                    break
                if args.max_steps is None and self.tokens_seen >= target_tokens:
                    break
                # Before train_step, so the branch point is the state at the
                # end of the stable phase rather than one decayed update into
                # the decay phase.
                self.maybe_milestone()
                self.maybe_enable_qat()
                try:
                    stats = self.train_step()
                except StopIteration:
                    # A finite batch source ran out -- post.py's SFT stream is
                    # one-shot by design. Treat it as the run ending, not as a
                    # crash: letting it propagate skips the forced checkpoint,
                    # the final Hub upload and wandb.finish() below, so an
                    # otherwise-complete fine-tune would lose everything since
                    # the last rolling checkpoint. Pretraining sources are
                    # infinite, so this cannot fire there.
                    print(f"batch source exhausted at step {self.step}; "
                         f"finishing run")
                    break
                if stats.get("skipped"):
                    self._consecutive_skips += 1
                    if self._consecutive_skips >= args.max_consecutive_skips:
                        # Log the row first: the evidence for *why* the run
                        # stopped has to outlive the exception.
                        self.log_step(stats, force=True)
                        raise NonFiniteStall(
                            f"{self._consecutive_skips} consecutive non-finite "
                            f"updates at step {self.step} "
                            f"({self._skipped_updates} total). Neither "
                            f"max_steps nor total_tokens can be reached while "
                            f"every step is skipped, so this run cannot "
                            f"progress. Check the input weights and the shard "
                            f"token ids -- `scripts/recovery_preflight.py` "
                            f"tells the two apart.")
                else:
                    self._consecutive_skips = 0
                self._last_stats = stats
                self.log_step(stats)
                self.maybe_checkpoint()
                self.maybe_hub_upload()
                self.maybe_push()
            # The run's finishing state, forced into the durable record. The
            # loop breaks at the *top* of an iteration, so the last interval row
            # lands up to `metrics_every_steps - 1` steps earlier and reports
            # fewer tokens than the target -- 4,994,316,288 of 5,000,000,000 for
            # `abl-arch` arm 1, because 10,391 is not a multiple of 20. Three
            # things compare that row against the target and all three read a
            # finished run as unfinished: `watchdog.detect_completion` (which
            # then falls through to `detect_stall`, whose halt marker makes
            # `abl_arch.py` refuse to retry), `scripts/eval_arm1_when_done.sh`,
            # and the writeup. The sweep probes ended at step 1040 and hid it.
            if self._last_stats is not None:
                self.log_step(self._last_stats, force=True)
            self.maybe_checkpoint(force=True)
            self.maybe_hub_upload(force=True)
            self.maybe_push(force=True)
            if args.finish_wandb:
                self.wandb.finish()
        finally:
            self._stop_uploader()
            if os.path.exists(pidfile):
                os.remove(pidfile)


def parse_args(argv=None) -> TrainArgs:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default=os.environ.get("RUN_NAME", "train"))
    p.add_argument("--config", default="daedalus-150m")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--total-tokens", type=int, default=5_000_000_000)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--seq-start", type=int, default=1024)
    p.add_argument("--seq-end", type=int, default=2048)
    p.add_argument("--tok-start", type=int, default=128_000)
    p.add_argument("--tok-end", type=int, default=512_000)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--resume", default=None,
                   help="continue an interrupted run: restores step, "
                        "tokens_seen and both optimizers")
    p.add_argument("--init-from", default=None,
                   help="start a NEW run from these weights (fine-tuning): "
                        "step 0, fresh optimizers, fresh LR schedule. "
                        "Ignored when --resume finds a checkpoint.")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--adam-lr", type=float, default=3e-4)
    p.add_argument("--conv-proj-wd", type=float, default=None,
                   help="weight decay for the ShortConv in_proj/out_proj "
                        "matrices, as a separate Muon group (issue #7's "
                        "channel-death fix; 0.0 disables decay there). "
                        "Default None = shipped single-group behaviour. "
                        "Changes the optimizer state_dict layout, so it cannot "
                        "be applied to a run already in progress.")
    p.add_argument("--conv-proj-wd-end", type=float, default=None,
                   help="ramp --conv-proj-wd to this value instead of holding "
                        "it constant. Default None = constant, the shipped "
                        "behaviour. Requires --conv-proj-wd.")
    p.add_argument("--conv-proj-wd-ramp-frac", type=float, default=0.0,
                   help="fraction of the run by which --conv-proj-wd-end is "
                        "reached (0.1 = by 10%% of steps)")
    p.add_argument("--conv-proj-wd-hold-frac", type=float, default=0.0,
                   help="fraction of the run to hold --conv-proj-wd before the "
                        "ramp begins; must be <= --conv-proj-wd-ramp-frac")
    p.add_argument("--device", default=None,
                   help="cuda/cpu; defaults to cuda when available. Mainly so "
                        "the end-to-end subprocess resume test can pin cpu "
                        "rather than depending on the box having a free GPU.")
    p.add_argument("--metrics-every-steps", type=int, default=None,
                   help="cadence for runs/<name>/metrics.jsonl (default 20)")
    p.add_argument("--tags", default=None, help="comma-separated extra W&B tags")
    p.add_argument("--val-dir", default=None,
                   help="held-out shard dir for periodic val_bpb; off if unset")
    p.add_argument("--val-every-steps", type=int, default=500)
    p.add_argument("--hub-repo",
                   default=os.environ.get("DAEDALUS_HF_MODEL_REPO",
                                          ckpt_uploader.DEFAULT_MODEL_REPO),
                   help="private HF model repo for checkpoint durability "
                        "(AGENT.md SS0.2). Pass an empty string to disable.")
    p.add_argument("--hub-every-sec", type=float, default=7200.0,
                   help="cadence for the weights-only rolling Hub copy")
    p.add_argument("--no-hub-uploader", action="store_true",
                   help="stage checkpoints but do not spawn the background "
                        "upload process (the final drain at exit still runs "
                        "when HF_TOKEN_WRITE is set)")
    p.add_argument("--qat-frac", type=float, default=0.0,
                   help="fraction of the run's tail spent quantization-aware on "
                        "llama.cpp's exact Q4_0 grid (blueprint: 0.05 for hero). "
                        "Leave at 0 for sweep/abl-arch -- a quantized forward "
                        "would invalidate those comparisons.")
    p.add_argument("--warmup-steps", type=int, default=None,
                   help="linear LR warmup length in steps (default 300). A "
                        "recovery run measured in hundreds of steps rather "
                        "than hundreds of thousands would otherwise spend most "
                        "of its budget still warming up.")
    p.add_argument("--decay-frac", type=float, default=None,
                   help="fraction of the run spent decaying LR to zero "
                        "(default 0.45). Also moves the milestone checkpoint, "
                        "which is written at the end of the stable phase.")
    p.add_argument("--loss-chunk-size", type=int, default=None,
                   help="tokens per chunk in the fused loss head (default: the "
                        "config's 1024; 0 restores the single-shot path). "
                        "Lower it to cut loss-head memory, which is what a "
                        "QAT forward needs on top of the usual activations.")
    p.add_argument("--mixture-weight", action="append", default=[],
                   metavar="NAME=FRACTION",
                   help="explicit share for one source under a mixture "
                        "--data-dir; repeatable, and the shares must sum to 1. "
                        "Without any, the blueprint mixture in "
                        "daedalus.dataprep.MIXTURE is used, which is what every "
                        "run before phase 7 did. A source given 0 stays on disk "
                        "and is never drawn, which is how a single-source arm "
                        "shares one data root with the mixture arms it is "
                        "compared against.")
    p.add_argument("--tokenizer", default=None,
                   help="path or Hub name of the tokenizer that produced the "
                        "ids in --data-dir/--val-dir (default: the config's, "
                        "which is SmolLM2 for every shipped preset). Only "
                        "val_bpb reads it, but it reads it to convert "
                        "nats-per-token into bits per *byte*, so a holdout "
                        "decoded with the wrong vocabulary reports a figure "
                        "for a corpus that does not exist.")
    p.add_argument("--gradient-checkpointing", dest="gradient_checkpointing",
                   action="store_true", default=None,
                   help="recompute block activations in backward instead of "
                        "storing them: ~30%% slower steps for a large drop in "
                        "activation memory. Default: the config's setting.")
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                   action="store_false",
                   help="force block checkpointing off even if the config "
                        "enables it.")
    p.add_argument("--ramp-frac", type=float, default=None,
                   help="fraction of the budget the batch/seq ramps span "
                        "(default 0.1). Only needed to *retarget* a run that is "
                        "already past its ramp: `scripts/early_finish.py` passes "
                        "the original run's ramp_tokens as a fraction of the new "
                        "budget, so estimate_total_steps replays the ramp that "
                        "actually happened. Without it a shortened budget "
                        "under-counts the ramp's steps and the schedule reaches "
                        "lr 0 before the token target -- measured at 1,690-4,250 "
                        "steps of training at lr exactly 0.")
    a = p.parse_args(argv)
    kwargs = dict(run_name=a.run_name, config=a.config, data_dir=a.data_dir,
                  total_tokens=a.total_tokens, max_steps=a.max_steps,
                  micro_batch=a.micro_batch, seq_start=a.seq_start,
                  seq_end=a.seq_end, tok_start=a.tok_start, tok_end=a.tok_end,
                  compile=not a.no_compile, resume=a.resume,
                  init_from=a.init_from,
                  wandb_enabled=not a.no_wandb,
                  muon_lr=a.muon_lr, adam_lr=a.adam_lr,
                  conv_proj_wd=a.conv_proj_wd,
                  conv_proj_wd_end=a.conv_proj_wd_end,
                  conv_proj_wd_ramp_frac=a.conv_proj_wd_ramp_frac,
                  conv_proj_wd_hold_frac=a.conv_proj_wd_hold_frac,
                  tags=a.tags.split(",") if a.tags else None,
                  val_dir=a.val_dir, val_every_steps=a.val_every_steps,
                  qat_frac=a.qat_frac, hub_repo=a.hub_repo or None,
                  hub_every_sec=a.hub_every_sec,
                  hub_uploader=not a.no_hub_uploader)
    # Omit rather than pass None, so TrainArgs' own cuda-if-available default
    # stays the single source of truth for the unspecified case.
    if a.device is not None:
        kwargs["device"] = a.device
    if a.metrics_every_steps is not None:
        kwargs["metrics_every_steps"] = a.metrics_every_steps
    if a.ramp_frac is not None:
        kwargs["ramp_frac"] = a.ramp_frac
    if a.warmup_steps is not None:
        kwargs["warmup_steps"] = a.warmup_steps
    if a.decay_frac is not None:
        kwargs["decay_frac"] = a.decay_frac
    # These two stay None when unset -- None *is* the "use the preset" value
    # for them, unlike the flags above whose default lives on TrainArgs.
    kwargs["loss_chunk_size"] = a.loss_chunk_size
    kwargs["gradient_checkpointing"] = a.gradient_checkpointing
    kwargs["tokenizer"] = a.tokenizer
    kwargs["mixture_weights"] = parse_mixture_weights(a.mixture_weight)
    return TrainArgs(**kwargs)


def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.fit()


if __name__ == "__main__":
    main()

"""Score a *random-init* model through the real eval harness.

Why this exists: the whole project is now reported against a 5-task mean, and
the hero gate rests partly on the 0.5B probe scoring 40.2 against a stated
chance floor of 35.0. That floor is *arithmetic* -- 1/n_choices per task -- and
arithmetic is not a measurement. If the harness leaks (an answer visible in the
context, a length-normalisation that rewards one choice systematically, a
mis-built choice set), an untrained model scores above 35.0 too, and every
comparison in `runs/eval/peer-table.md` is inflated by an unknown constant.

A random-init model is the control that tells the two apart, and it is the only
way to test the floor without training something. Expected result: ~35.0 mean,
each task within noise of its own chance value.

Usage:
    python scripts/eval_chance_control.py --task-limit 500
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daedalus.config import PRESETS          # noqa: E402
from daedalus.model import Daedalus          # noqa: E402

CHANCE = {"hellaswag": 25.0, "arc_easy": 25.0, "piqa": 50.0,
          "openbookqa": 25.0, "winogrande": 50.0}


def write_random_checkpoint(path: str, cfg_name: str, seed: int) -> str:
    """A checkpoint of an untrained model, in train.py's own payload format so
    it goes through the identical load path a real checkpoint does."""
    torch.manual_seed(seed)
    model = Daedalus(PRESETS[cfg_name])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "step": 0, "tokens_seen": 0,
                "config": {"preset": cfg_name, "note": "random init, control"}},
               path)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="daedalus-150m")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--task-limit", type=int, default=500,
                   help="examples per task; the floor needs precision, not the "
                        "full split -- 500 gives ~+-2 points at 95%%")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt", default="runs/eval/random-init-control.pt")
    p.add_argument("--out", default="runs/eval/chance-control.json")
    args = p.parse_args()

    import eval as ev
    from daedalus.data import get_tokenizer

    write_random_checkpoint(args.ckpt, args.config, args.seed)
    tokenizer = get_tokenizer()
    tasks = ev.load_all_tasks(limit=args.task_limit)

    res = ev.evaluate_checkpoint(args.ckpt, args.config, tokenizer, tasks,
                                 device=args.device)

    rows, deltas = [], []
    for name, chance in CHANCE.items():
        got = res.get(name)
        if got is None:
            continue
        got *= 100
        rows.append({"task": name, "chance": chance, "random_init": got,
                     "delta": got - chance})
        deltas.append(got - chance)

    mean = sum(r["random_init"] for r in rows) / len(rows)
    chance_mean = sum(r["chance"] for r in rows) / len(rows)
    out = {"config": args.config, "seed": args.seed,
           "task_limit": args.task_limit, "per_task": rows,
           "mean": mean, "chance_mean": chance_mean,
           "mean_delta": mean - chance_mean,
           "max_abs_task_delta": max(abs(d) for d in deltas)}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"{'task':<14}{'chance':>8}{'random':>9}{'delta':>8}")
    for r in rows:
        print(f"{r['task']:<14}{r['chance']:>8.1f}{r['random_init']:>9.1f}"
              f"{r['delta']:>+8.1f}")
    print(f"{'MEAN':<14}{chance_mean:>8.1f}{mean:>9.1f}{mean - chance_mean:>+8.1f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

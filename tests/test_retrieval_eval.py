"""Tests for the retrieval evaluation backends and their scorecard output.

Both backends have to answer the *same* items so a paired comparison between
PyTorch FP16 and a stock llama.cpp GGUF is valid, and neither may quietly
succeed when the model, the binary, or a flag is wrong. These tests pin the
argv, the parsing, the determinism settings, and the control gate.
"""

import json
import subprocess

import pytest

from daedalus.retrieval import (
    OracleBackend,
    make_copy_control_items,
    make_passkey_items,
    score_items,
)
from daedalus.scorecard import ArtifactRef, load_scorecard


class WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)


@pytest.fixture
def tokenizer():
    return WhitespaceTokenizer()


HELP_TEXT = """
usage: llama-cli [options]
  -m,    --model FNAME
  -f,    --file FNAME
  -n,    --predict N
  -c,    --ctx-size N
  -t,    --threads N
  -ngl,  --gpu-layers N
         --temp N
         --top-k N
  -s,    --seed SEED
         --no-warmup
  -no-cnv, --no-conversation
         --no-display-prompt
         --simple-io
"""


# ------------------------------------------------------------ control gate ---

def test_verify_items_accepts_well_formed_items(tokenizer):
    from scripts.retrieval_eval import verify_items

    items = (make_passkey_items(tokenizer, depths=(256,), per_depth=4, seed=1)
             + make_copy_control_items(tokenizer, per_item=4, seed=1))

    assert verify_items(items)["exact_match"] == 1.0


def test_verify_items_refuses_to_run_a_model_on_broken_items(tokenizer):
    from scripts.retrieval_eval import ControlFailure, verify_items
    from daedalus.retrieval import RetrievalItem

    items = make_passkey_items(tokenizer, depths=(256,), per_depth=4, seed=1)
    broken = [RetrievalItem(**{**item.__dict__,
                               "prompt": item.prompt.replace(item.needle_text, "")})
              for item in items]

    with pytest.raises(ControlFailure, match="control"):
        verify_items(broken)


# -------------------------------------------------------- llama.cpp backend ---

def _fake_runner(record, stdout="", returncode=0, stderr=""):
    def runner(command, **kwargs):
        record.append({"command": list(command), "kwargs": kwargs})
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)
    return runner


def test_llama_cpp_backend_probes_help_once_and_caches_flags(tmp_path, tokenizer):
    from scripts.retrieval_eval import LlamaCppBackend

    calls = []
    backend = LlamaCppBackend(gguf_path=tmp_path / "m.gguf",
                              binary=tmp_path / "llama-cli",
                              runner=_fake_runner(calls, stdout=HELP_TEXT))
    backend.generate(make_passkey_items(tokenizer, depths=(256,), per_depth=1,
                                        seed=1)[0])
    backend.generate(make_passkey_items(tokenizer, depths=(256,), per_depth=1,
                                        seed=2)[0])

    help_calls = [call for call in calls if "--help" in call["command"]]
    assert len(help_calls) == 1


def test_llama_cpp_backend_runs_greedy_and_prints_only_the_completion(
        tmp_path, tokenizer):
    from scripts.retrieval_eval import LlamaCppBackend

    calls = []
    backend = LlamaCppBackend(gguf_path=tmp_path / "m.gguf",
                              binary=tmp_path / "llama-cli", threads=8,
                              n_ctx=4096, max_new_tokens=12, seed=99,
                              runner=_fake_runner(calls, stdout=HELP_TEXT))
    item = make_passkey_items(tokenizer, depths=(256,), per_depth=1, seed=1)[0]

    backend.generate(item)

    command = [call for call in calls if "--help" not in call["command"]][0]["command"]
    assert command[0] == str(tmp_path / "llama-cli")
    assert "-m" in command and str(tmp_path / "m.gguf") in command
    # Greedy and reproducible: temperature zero, top-k 1, a fixed seed.
    assert command[command.index("--temp") + 1] == "0"
    assert command[command.index("--top-k") + 1] == "1"
    assert command[command.index("-s") + 1] == "99"
    assert command[command.index("-n") + 1] == "12"
    assert command[command.index("-c") + 1] == "4096"
    assert command[command.index("-t") + 1] == "8"
    # CPU-only, no chat template, and stdout carries the completion alone.
    assert command[command.index("-ngl") + 1] == "0"
    assert "-no-cnv" in command
    assert "--no-display-prompt" in command
    assert "--no-warmup" in command


def test_llama_cpp_backend_closes_stdin_so_a_prompt_cannot_hang_it(
        tmp_path, tokenizer):
    """Every child gets a closed stdin, including the `--help` probe.

    Observed against the pinned build: `llama-cli` sat in conversation mode
    waiting on stdin and every item died on the 60 s timeout, with no output to
    say why. Which flag turns conversation off has been spelled three ways
    upstream, so the flag probe alone is not a guarantee -- a closed stdin is,
    because the child reads EOF and exits instead of waiting for a person who
    is not there.
    """
    from scripts.retrieval_eval import LlamaCppBackend

    calls = []
    backend = LlamaCppBackend(gguf_path=tmp_path / "m.gguf",
                              binary=tmp_path / "llama-cli",
                              runner=_fake_runner(calls, stdout=HELP_TEXT))

    backend.generate(make_passkey_items(tokenizer, depths=(256,), per_depth=1,
                                        seed=1)[0])

    assert calls, "the backend made no subprocess call"
    for call in calls:
        assert call["kwargs"].get("stdin") is subprocess.DEVNULL, call["command"]


def test_llama_cpp_backend_accepts_either_spelling_of_no_conversation(
        tmp_path, tokenizer):
    """`--no-conversation` counts, not just the `-no-cnv` short form.

    The pinned build advertises only the long spelling, so a probe that looked
    for the short one alone found nothing, left conversation mode on, and hung.
    """
    from scripts.retrieval_eval import LlamaCppBackend

    calls = []
    long_only = HELP_TEXT.replace("-no-cnv, --no-conversation",
                                  "       --no-conversation")
    backend = LlamaCppBackend(gguf_path=tmp_path / "m.gguf",
                              binary=tmp_path / "llama-cli",
                              runner=_fake_runner(calls, stdout=long_only))

    backend.generate(make_passkey_items(tokenizer, depths=(256,), per_depth=1,
                                        seed=1)[0])

    command = [call for call in calls if "--help" not in call["command"]][0]["command"]
    assert "--no-conversation" in command
    assert "-no-cnv" not in command


def test_llama_cpp_backend_refuses_a_binary_that_cannot_leave_conversation_mode(
        tmp_path, tokenizer):
    """No conversation switch means no non-interactive guarantee -- so stop.

    Without one, `llama-cli` sits at its `> ` prompt and every item burns the
    full generation timeout before failing, so a sweep of 88 items wastes 88
    minutes to learn one fact. The error names the flags the binary *did*
    advertise, because the next question is always "then what does it call it?".
    """
    from scripts.retrieval_eval import LlamaCppBackend

    without = HELP_TEXT.replace("-no-cnv, --no-conversation", "")
    backend = LlamaCppBackend(gguf_path=tmp_path / "m.gguf",
                              binary=tmp_path / "llama-cli",
                              runner=_fake_runner([], stdout=without))

    with pytest.raises(RuntimeError, match="conversation"):
        backend.generate(make_passkey_items(tokenizer, depths=(256,),
                                            per_depth=1, seed=1)[0])


def test_conversation_flag_may_be_waived_for_a_binary_that_never_had_one(
        tmp_path, tokenizer):
    """The refusal is a default, not a wall.

    A genuinely old `llama-cli` predates conversation mode entirely and is
    non-interactive already; refusing it would block a legitimate binary on a
    flag it has no reason to carry.
    """
    from scripts.retrieval_eval import LlamaCppBackend

    calls = []
    without = HELP_TEXT.replace("-no-cnv, --no-conversation", "")
    backend = LlamaCppBackend(gguf_path=tmp_path / "m.gguf",
                              binary=tmp_path / "llama-cli",
                              require_non_interactive=False,
                              runner=_fake_runner(calls, stdout=without))

    backend.generate(make_passkey_items(tokenizer, depths=(256,), per_depth=1,
                                        seed=1)[0])

    command = [call for call in calls if "--help" not in call["command"]][0]["command"]
    assert "-no-cnv" not in command


def test_llama_cpp_backend_falls_back_to_single_turn(tmp_path, tokenizer):
    """`-st` counts when the build has dropped the `-no-cnv` family.

    The pinned build advertises only `-st, --single-turn`; without accepting it
    the backend refuses a binary it can in fact drive.
    """
    from scripts.retrieval_eval import LlamaCppBackend

    calls = []
    modern = HELP_TEXT.replace("-no-cnv, --no-conversation",
                               "-st,   --single-turn")
    backend = LlamaCppBackend(gguf_path=tmp_path / "m.gguf",
                              binary=tmp_path / "llama-cli",
                              runner=_fake_runner(calls, stdout=modern))

    backend.generate(make_passkey_items(tokenizer, depths=(256,), per_depth=1,
                                        seed=1)[0])

    command = [call for call in calls if "--help" not in call["command"]][0]["command"]
    assert "-st" in command
    assert "-no-cnv" not in command


def test_show_prompt_keeps_the_echo_so_the_real_prompt_is_readable(
        tmp_path, tokenizer):
    """`--show-prompt` drops `--no-display-prompt`, and nothing else.

    `-st` still applies the model's chat template, so "what did llama.cpp
    actually feed the model" is a question a base-model harness has to be able
    to answer rather than assume.
    """
    from scripts.retrieval_eval import LlamaCppBackend

    calls = []
    backend = LlamaCppBackend(gguf_path=tmp_path / "m.gguf",
                              binary=tmp_path / "llama-cli", show_prompt=True,
                              runner=_fake_runner(calls, stdout=HELP_TEXT))

    backend.generate(make_passkey_items(tokenizer, depths=(256,), per_depth=1,
                                        seed=1)[0])

    command = [call for call in calls if "--help" not in call["command"]][0]["command"]
    assert "--no-display-prompt" not in command
    assert "-no-cnv" in command


def test_llama_cpp_backend_reports_the_flags_it_resolved(tmp_path, tokenizer):
    """The resolved flag set is provenance, not a debug aid.

    Two runs of "stock llama.cpp" that differ only in whether the binary
    advertised `--no-conversation` are not the same measurement, and nothing
    else in the scorecard would record the difference.
    """
    from scripts.retrieval_eval import LlamaCppBackend

    backend = LlamaCppBackend(gguf_path=tmp_path / "m.gguf",
                              binary=tmp_path / "llama-cli",
                              runner=_fake_runner([], stdout=HELP_TEXT))

    assert "-no-cnv" in backend.resolved_flags()
    assert "--no-display-prompt" in backend.resolved_flags()


def test_llama_cpp_backend_passes_the_prompt_by_file_not_argv(tmp_path, tokenizer):
    from scripts.retrieval_eval import LlamaCppBackend

    calls = []
    seen = {}

    def runner(command, **kwargs):
        if "--help" in command:
            return subprocess.CompletedProcess(command, 0, HELP_TEXT, "")
        prompt_file = command[command.index("-f") + 1]
        seen["prompt"] = open(prompt_file).read()
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "4821", "")

    backend = LlamaCppBackend(gguf_path=tmp_path / "m.gguf",
                              binary=tmp_path / "llama-cli", runner=runner)
    item = make_passkey_items(tokenizer, depths=(256,), per_depth=1, seed=1)[0]

    backend.generate(item)

    # Byte-exact, so what the model saw is what the scorecard recorded.
    assert seen["prompt"] == item.prompt


def test_llama_cpp_backend_strips_end_of_text_and_timing_noise(tmp_path, tokenizer):
    from scripts.retrieval_eval import LlamaCppBackend

    stdout = " 4821 [end of text]\n"
    calls = []

    def runner(command, **kwargs):
        if "--help" in command:
            return subprocess.CompletedProcess(command, 0, HELP_TEXT, "")
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout,
                                           "llama_perf_context_print: eval time = 1ms")

    backend = LlamaCppBackend(gguf_path=tmp_path / "m.gguf",
                              binary=tmp_path / "llama-cli", runner=runner)
    item = make_passkey_items(tokenizer, depths=(256,), per_depth=1, seed=1)[0]

    assert backend.generate(item) == "4821"


def test_llama_cpp_backend_omits_flags_the_binary_does_not_advertise(
        tmp_path, tokenizer):
    from scripts.retrieval_eval import LlamaCppBackend

    calls = []
    older_help = HELP_TEXT.replace("--no-warmup", "")
    backend = LlamaCppBackend(gguf_path=tmp_path / "m.gguf",
                              binary=tmp_path / "llama-cli",
                              runner=_fake_runner(calls, stdout=older_help))
    backend.generate(make_passkey_items(tokenizer, depths=(256,), per_depth=1,
                                        seed=1)[0])

    command = [call for call in calls if "--help" not in call["command"]][0]["command"]
    assert "--no-warmup" not in command
    assert "--no-display-prompt" in command


def test_llama_cpp_backend_discards_a_chat_ui_banner_and_prompt_echo(
        tmp_path, tokenizer):
    """A build that prints its whole chat UI to stdout still yields the answer.

    The pinned binary prints a spinner, ASCII art, a build/model block and a
    command list, ignores `--no-display-prompt`, echoes the prompt after `> `,
    and closes with a throughput line. Filtering known noise markers left
    "Loading model..." as the recorded completion for every item -- a score of
    zero that is indistinguishable from a model that cannot retrieve.
    """
    from scripts.retrieval_eval import LlamaCppBackend

    item = make_passkey_items(tokenizer, depths=(256,), per_depth=1, seed=1)[0]
    stdout = (
        "Loading model... |\b-\b\n"
        "██ ██  banner art\n\n"
        "build      : b1-7584430\n"
        "model      : /root/daedalus/gguf/hero-base-f16.gguf\n"
        "ftype      : F16\n\n"
        "available commands:\n"
        "  /exit or Ctrl+C     stop or exit\n\n\n"
        f"> {item.prompt}\n\n"
        "4821\n\n"
        "[ Prompt: 613.2 t/s | Generation: 85.5 t/s ]\n\n\n"
        "Exiting...")

    def runner(command, **kwargs):
        if "--help" in command:
            return subprocess.CompletedProcess(command, 0, HELP_TEXT, "")
        return subprocess.CompletedProcess(command, 0, stdout, "")

    backend = LlamaCppBackend(gguf_path=tmp_path / "m.gguf",
                              binary=tmp_path / "llama-cli", runner=runner)

    assert backend.generate(item) == "4821"


def test_llama_cpp_backend_keeps_an_answer_that_repeats_the_prompt(
        tmp_path, tokenizer):
    """Anchor on the *first* echo, not the last match.

    These models repeat their prompt back. Anchoring on the last occurrence
    would treat the model's own repetition as the echo and cut the answer that
    follows it -- scoring zero on exactly the items the model got right.
    """
    from scripts.retrieval_eval import LlamaCppBackend

    item = make_passkey_items(tokenizer, depths=(256,), per_depth=1, seed=1)[0]
    stdout = f"> {item.prompt}\n\n{item.prompt} 4821\n"

    def runner(command, **kwargs):
        if "--help" in command:
            return subprocess.CompletedProcess(command, 0, HELP_TEXT, "")
        return subprocess.CompletedProcess(command, 0, stdout, "")

    backend = LlamaCppBackend(gguf_path=tmp_path / "m.gguf",
                              binary=tmp_path / "llama-cli", runner=runner)

    assert backend.generate(item).endswith("4821")


def test_llama_cpp_backend_handles_a_truncated_prompt_echo(tmp_path, tokenizer):
    """Long prompts are echoed elided, so the full prompt never appears.

    The deep retrieval items are exactly the ones this hits, so without it the
    depth curve the gates read would be zero from 512 tokens up -- the shape a
    genuine long-context failure has.
    """
    from scripts.retrieval_eval import LlamaCppBackend

    item = make_passkey_items(tokenizer, depths=(256,), per_depth=1, seed=1)[0]
    stdout = (f"Loading model... |\b\n\navailable commands:\n  /exit\n\n"
              f"> {item.prompt[:40]} ... (truncated)\n\n4821\n\n"
              "[ Prompt: 613.2 t/s | Generation: 85.5 t/s ]\nExiting...")

    def runner(command, **kwargs):
        if "--help" in command:
            return subprocess.CompletedProcess(command, 0, HELP_TEXT, "")
        return subprocess.CompletedProcess(command, 0, stdout, "")

    backend = LlamaCppBackend(gguf_path=tmp_path / "m.gguf",
                              binary=tmp_path / "llama-cli", runner=runner)

    assert backend.generate(item) == "4821"


def test_llama_cpp_backend_refuses_to_score_leaked_banner_text(
        tmp_path, tokenizer):
    """An unrecognised layout fails the run instead of scoring the banner.

    This is the bug that motivated the whole extraction path: every item scored
    zero with "Loading model..." recorded as the model's answer, which reads as
    a model that cannot retrieve rather than a harness that never found the
    completion.
    """
    from scripts.retrieval_eval import LlamaCppBackend

    item = make_passkey_items(tokenizer, depths=(256,), per_depth=1, seed=1)[0]

    def runner(command, **kwargs):
        if "--help" in command:
            return subprocess.CompletedProcess(command, 0, HELP_TEXT, "")
        return subprocess.CompletedProcess(
            command, 0, "Loading model... |\b\n\nsome unrecognised layout\n", "")

    backend = LlamaCppBackend(gguf_path=tmp_path / "m.gguf",
                              binary=tmp_path / "llama-cli", runner=runner)

    with pytest.raises(RuntimeError, match="banner"):
        backend.generate(item)


def test_llama_cpp_backend_raises_on_a_failed_run(tmp_path, tokenizer):
    from scripts.retrieval_eval import LlamaCppBackend

    def runner(command, **kwargs):
        if "--help" in command:
            return subprocess.CompletedProcess(command, 0, HELP_TEXT, "")
        return subprocess.CompletedProcess(command, 1, "", "error: unknown argument")

    backend = LlamaCppBackend(gguf_path=tmp_path / "m.gguf",
                              binary=tmp_path / "llama-cli", runner=runner)
    item = make_passkey_items(tokenizer, depths=(256,), per_depth=1, seed=1)[0]

    with pytest.raises(RuntimeError, match="unknown argument"):
        backend.generate(item)


def test_llama_cpp_backend_bounds_every_call_with_a_timeout(tmp_path, tokenizer):
    from scripts.retrieval_eval import LlamaCppBackend

    calls = []
    backend = LlamaCppBackend(gguf_path=tmp_path / "m.gguf",
                              binary=tmp_path / "llama-cli", timeout_s=123.0,
                              runner=_fake_runner(calls, stdout=HELP_TEXT))
    backend.generate(make_passkey_items(tokenizer, depths=(256,), per_depth=1,
                                        seed=1)[0])

    generation = [call for call in calls if "--help" not in call["command"]][0]
    assert generation["kwargs"]["timeout"] == 123.0


# ------------------------------------------------------------ torch backend ---

class ConstantModel:
    """Emits a fixed id sequence, so decoding logic is tested without a model."""

    def __init__(self, ids, vocab_size=32):
        self.ids = list(ids)
        self.vocab_size = vocab_size
        self.calls = 0
        self.training = False

    def eval(self):
        return self

    def train(self, mode=True):
        self.training = mode
        return self

    def __call__(self, input_ids, targets=None, return_logits=True):
        import torch
        step = min(self.calls, len(self.ids) - 1)
        logits = torch.full((1, input_ids.shape[1], self.vocab_size), -10.0)
        logits[0, -1, self.ids[step]] = 10.0
        self.calls += 1
        return logits, None, None


class IdTokenizer:
    """Maps ids to single characters, so generated text is predictable."""

    alphabet = "0123456789abcdefghijklmnopqrstu"

    def encode(self, text, add_special_tokens=False):
        return [self.alphabet.index(ch) for ch in text if ch in self.alphabet]

    def decode(self, ids):
        return "".join(self.alphabet[i] if i < len(self.alphabet) else "\n"
                       for i in ids)


def test_torch_backend_decodes_greedily_and_stops_at_the_budget():
    from scripts.retrieval_eval import TorchBackend
    from daedalus.retrieval import RetrievalItem

    model = ConstantModel([4, 8, 2, 1])
    backend = TorchBackend(model=model, tokenizer=IdTokenizer(), device="cpu",
                           max_new_tokens=3)
    item = RetrievalItem(id="x", task="passkey", depth=256, prompt="123",
                         answer="482", needle_text="482", needle_depth_frac=0.0,
                         prompt_tokens=3)

    assert backend.generate(item) == "482"


def test_torch_backend_stops_at_the_eos_token():
    from scripts.retrieval_eval import TorchBackend
    from daedalus.retrieval import RetrievalItem

    model = ConstantModel([4, 0, 9])
    backend = TorchBackend(model=model, tokenizer=IdTokenizer(), device="cpu",
                           max_new_tokens=8, eos_id=0)
    item = RetrievalItem(id="x", task="passkey", depth=256, prompt="123",
                         answer="4", needle_text="4", needle_depth_frac=0.0,
                         prompt_tokens=3)

    assert backend.generate(item) == "4"


# -------------------------------------------------------------- scorecards ---

def test_run_retrieval_writes_one_scorecard_per_task(tmp_path, tokenizer):
    from scripts.retrieval_eval import run_retrieval

    items = {
        "passkey": make_passkey_items(tokenizer, depths=(256,), per_depth=2, seed=1),
        "copy-control": make_copy_control_items(tokenizer, per_item=2, seed=1),
    }
    artifact = ArtifactRef(path="m.gguf", sha256="a" * 64, kind="gguf-q4_0")
    tokenizer_ref = ArtifactRef(path="tok.json", sha256="b" * 64, kind="tokenizer")

    written = run_retrieval(items, OracleBackend(), out_dir=tmp_path,
                            artifact=artifact, tokenizer_ref=tokenizer_ref,
                            seed=1, git_sha="deadbee",
                            runtime={"backend": "oracle"})

    assert set(written) == {"passkey", "copy-control"}
    card = load_scorecard(written["passkey"]["scorecard"])
    assert card.kind == "retrieval"
    assert card.name == "retrieval-passkey"
    assert card.metrics["exact_match"] == 1.0
    assert card.item_count == 2
    assert card.provenance.bpb_mode == "not-applicable"
    assert card.provenance.runtime["backend"] == "oracle"
    assert [record["id"] for record in card.items] == [
        "passkey-d256-0", "passkey-d256-1"]


def test_run_retrieval_records_the_exact_prompt_for_every_item(tmp_path, tokenizer):
    from scripts.retrieval_eval import run_retrieval

    items = {"passkey": make_passkey_items(tokenizer, depths=(256,), per_depth=2,
                                           seed=1)}
    written = run_retrieval(items, OracleBackend(), out_dir=tmp_path,
                            artifact=ArtifactRef(path="m.gguf", sha256="a" * 64,
                                                 kind="gguf-q4_0"),
                            tokenizer_ref=ArtifactRef(path="t.json",
                                                      sha256="b" * 64,
                                                      kind="tokenizer"),
                            seed=1, git_sha="deadbee")

    sidecar = json.loads(written["passkey"]["items"].read_text())
    assert sidecar["items"][0]["prompt"] == items["passkey"][0].prompt


def test_scored_records_are_pairable_across_two_backends(tokenizer):
    from daedalus.scorecard import Provenance, Scorecard, paired_outcomes

    items = make_passkey_items(tokenizer, depths=(256,), per_depth=4, seed=1)
    oracle = score_items(items, OracleBackend().generate_all(items))
    degraded = score_items(items, ["" for _ in items])

    provenance = Provenance(
        artifact=ArtifactRef(path="m.gguf", sha256="a" * 64, kind="gguf-f16"),
        tokenizer=ArtifactRef(path="t.json", sha256="b" * 64, kind="tokenizer"),
        seed=1, git_sha="deadbee")
    left = Scorecard(kind="retrieval", name="fp16", provenance=provenance,
                     metrics={"exact_match": 1.0}, created_at="2026-08-24T00:00:00Z",
                     items=[{"id": r["id"], "correct": r["correct"]} for r in oracle])
    right = Scorecard(kind="retrieval", name="q4", provenance=provenance,
                      metrics={"exact_match": 0.0},
                      created_at="2026-08-24T00:00:00Z",
                      items=[{"id": r["id"], "correct": r["correct"]}
                             for r in degraded])

    paired = paired_outcomes(left, right)

    assert paired["n"] == 4
    assert paired["left_only"] == 4
    assert paired["delta"] == pytest.approx(-1.0)


def test_parse_depths_accepts_a_comma_list():
    from scripts.retrieval_eval import parse_depths

    assert parse_depths("256,512") == (256, 512)
    with pytest.raises(ValueError):
        parse_depths("256,notanumber")

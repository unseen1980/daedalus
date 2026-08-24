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
    older_help = HELP_TEXT.replace("-no-cnv, --no-conversation", "")
    backend = LlamaCppBackend(gguf_path=tmp_path / "m.gguf",
                              binary=tmp_path / "llama-cli",
                              runner=_fake_runner(calls, stdout=older_help))
    backend.generate(make_passkey_items(tokenizer, depths=(256,), per_depth=1,
                                        seed=1)[0])

    command = [call for call in calls if "--help" not in call["command"]][0]["command"]
    assert "-no-cnv" not in command
    assert "--no-display-prompt" in command


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

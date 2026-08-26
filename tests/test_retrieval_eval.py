"""Tests for the retrieval evaluation backends and their scorecard output.

Both backends have to answer the *same* items so a paired comparison between
PyTorch FP16 and a stock llama.cpp GGUF is valid, and neither may quietly
succeed when the model, the binary, or a flag is wrong. These tests pin the
argv, the parsing, the determinism settings, and the control gate.
"""

import json
import os
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


SINGLE_TURN_ONLY_HELP = HELP_TEXT.replace("-no-cnv, --no-conversation",
                                          "-st,   --single-turn")


PASSTHROUGH_HELP = SINGLE_TURN_ONLY_HELP.replace(
    "         --simple-io",
    "         --jinja\n         --chat-template-file FNAME\n         --simple-io")


# --------------------------------------------------------- binary capability ---

def test_probe_reports_a_raw_completion_build_as_usable(tmp_path):
    """The probe answers the question the scoring guards can only refuse.

    `supported_flags` raises on a build that cannot measure a base model, which
    is right for a run and useless for an operator asking what the build *does*
    support. There is no approved way to run `llama-cli --help` directly, so
    without this the only reading of a build's capabilities is a traceback.
    """
    from scripts.retrieval_eval import probe_binary

    binary = tmp_path / "llama-cli"
    binary.write_text("#!/bin/sh\n")

    report = probe_binary(binary, runner=_fake_runner([], stdout=HELP_TEXT))

    assert report["exists"] is True
    assert report["required_missing"] == []
    assert report["raw_completion_flags"] == ["-no-cnv", "--no-conversation"]
    assert report["base_model_route"] == "raw-completion"


def test_probe_names_the_passthrough_route_on_a_templated_only_build(tmp_path):
    """A templated-only build that offers `--jinja` is not actually blocked.

    A pass-through Jinja template renders the messages verbatim, so the
    templated turn feeds the model the prompt as written. That is the
    difference between "this box needs llama.cpp rebuilt" and "this box needs a
    template file", and only the binary's own help can tell them apart.
    """
    from scripts.retrieval_eval import probe_binary

    binary = tmp_path / "llama-cli"
    binary.write_text("#!/bin/sh\n")

    report = probe_binary(binary, runner=_fake_runner([], stdout=PASSTHROUGH_HELP))

    assert report["raw_completion_flags"] == []
    assert report["single_turn_flags"] == ["-st", "--single-turn"]
    assert report["passthrough_flags"] == ["--jinja", "--chat-template-file"]
    assert report["base_model_route"] == "passthrough-template"


def test_probe_reports_no_route_when_the_build_offers_neither(tmp_path):
    """The honest answer stays available: a rebuild really is required."""
    from scripts.retrieval_eval import probe_binary

    binary = tmp_path / "llama-cli"
    binary.write_text("#!/bin/sh\n")

    report = probe_binary(binary, runner=_fake_runner([],
                                                      stdout=SINGLE_TURN_ONLY_HELP))

    assert report["base_model_route"] == "none"
    assert "single-turn" in report["conversation_help"]


def test_probe_lists_sibling_binaries_that_could_complete_a_prompt(tmp_path):
    """`llama-completion` existing beside `llama-cli` is the cheapest remedy.

    The blocker was recorded as "rebuild llama.cpp" partly because nothing could
    answer whether the build already shipped a non-UI binary. The directory can.
    """
    from scripts.retrieval_eval import probe_binary

    binary = tmp_path / "llama-cli"
    binary.write_text("#!/bin/sh\n")
    (tmp_path / "llama-quantize").write_text("#!/bin/sh\n")
    (tmp_path / "llama-server").write_text("#!/bin/sh\n")

    report = probe_binary(binary, runner=_fake_runner([], stdout=HELP_TEXT))

    assert report["siblings"] == ["llama-cli", "llama-quantize", "llama-server"]


def test_probe_reports_a_missing_or_unrunnable_binary_without_raising(tmp_path):
    """A diagnostic that raises is the failure it exists to explain."""
    from scripts.retrieval_eval import probe_binary

    absent = probe_binary(tmp_path / "nope")
    assert absent["exists"] is False and absent["error"]

    binary = tmp_path / "llama-cli"
    binary.write_text("#!/bin/sh\n")

    def explode(command, **kwargs):
        raise OSError("Exec format error")

    broken = probe_binary(binary, runner=explode)
    assert "OSError" in broken["error"]


def test_probe_cli_prints_json_without_touching_the_tokenizer(tmp_path, capsys,
                                                              monkeypatch):
    """`--probe` must work on a box where the scoring path cannot even start.

    Building the items downloads a tokenizer; a probe asks about a binary. If
    the probe went through the tokenizer first, the one situation it exists for
    -- an environment that is not working -- would fail for an unrelated reason.
    """
    import scripts.retrieval_eval as module

    binary = tmp_path / "llama-cli"
    binary.write_text("#!/bin/sh\n")

    def refuse_tokenizer():
        raise AssertionError("--probe must not build items")

    monkeypatch.setattr("daedalus.data.get_tokenizer", refuse_tokenizer)
    monkeypatch.setattr(module, "probe_binary",
                        lambda path, **kw: {"binary": str(path), "probe": "ran"})

    assert module.main(["--backend", "llama-cpp", "--probe",
                        "--llama-cli", str(binary)]) == 0
    assert json.loads(capsys.readouterr().out)["probe"] == "ran"


# ------------------------------------------------------ pass-through template ---

def test_a_templated_only_build_with_jinja_is_driven_raw(tmp_path, tokenizer):
    """The pinned UI build can measure a base model after all.

    It advertises `-st`, no `-no-cnv`, *and* `--jinja`/`--chat-template-file`.
    A pass-through template renders the messages verbatim, so the templated turn
    hands the model the prompt as written -- on a stock binary, with no rebuild.
    Refusing here would leave the retrieval column unmeasurable for an
    environment reason that is not actually true.
    """
    from scripts.retrieval_eval import PASSTHROUGH_TEMPLATE, LlamaCppBackend

    calls = []
    backend = LlamaCppBackend(gguf_path=tmp_path / "m.gguf",
                              binary=tmp_path / "llama-cli",
                              runner=_fake_runner(calls, stdout=PASSTHROUGH_HELP))
    item = make_passkey_items(tokenizer, depths=(256,), per_depth=1, seed=1)[0]

    backend.generate(item)

    command = [call for call in calls if "--help" not in call["command"]][0]["command"]
    assert "-st" in command
    assert "--jinja" in command
    template = command[command.index("--chat-template-file") + 1]
    assert template.endswith(".jinja")
    # Written for the call and removed with the prompt file: nothing about this
    # measurement may depend on a file left behind by an earlier run.
    assert not os.path.exists(template)
    assert "role" not in PASSTHROUGH_TEMPLATE


def test_a_passthrough_turn_is_recorded_as_raw_completion_and_how(tmp_path):
    """`raw-completion` for the gate, `raw_completion_via` for the reader.

    The report gate matches `template_mode` exactly, so a pass-through card has
    to report the mode it actually measured or the column stays unmeasured for
    no reason. But "raw completion" reached two different ways is two different
    argv, and a card that cannot say which is not reproducible.
    """
    from scripts.retrieval_eval import LlamaCppBackend

    passthrough = LlamaCppBackend(gguf_path=tmp_path / "m.gguf",
                                  binary=tmp_path / "llama-cli",
                                  runner=_fake_runner([], stdout=PASSTHROUGH_HELP))
    assert passthrough.template_mode() == "raw-completion"
    assert passthrough.raw_completion_via() == "passthrough-template"
    assert passthrough.uses_passthrough_template() is True


def test_a_no_conversation_build_ignores_the_passthrough_route(tmp_path,
                                                               tokenizer):
    """`-no-cnv` stays preferred when the build has it.

    Leaving the chat machinery out entirely is fewer moving parts than
    neutralising it, and it keeps a scorecard's meaning independent of upstream
    Jinja behaviour. A build offering both must not quietly switch routes.
    """
    from scripts.retrieval_eval import LlamaCppBackend

    calls = []
    both = HELP_TEXT.replace(
        "         --simple-io",
        "         --jinja\n         --chat-template-file FNAME\n         --simple-io")
    backend = LlamaCppBackend(gguf_path=tmp_path / "m.gguf",
                              binary=tmp_path / "llama-cli",
                              runner=_fake_runner(calls, stdout=both))

    backend.generate(make_passkey_items(tokenizer, depths=(256,), per_depth=1,
                                        seed=1)[0])

    command = [call for call in calls if "--help" not in call["command"]][0]["command"]
    assert "-no-cnv" in command
    assert "--chat-template-file" not in command
    assert backend.uses_passthrough_template() is False
    assert backend.raw_completion_via() == "no-conversation-flag"


def test_passthrough_needs_both_flags_not_either(tmp_path, tokenizer):
    """`--chat-template-file` without a Jinja engine is not applied.

    Half the route is not a route, and accepting it would put the templated
    zeros back with a scorecard claiming raw completion -- strictly worse than
    the refusal it replaced.
    """
    from scripts.retrieval_eval import LlamaCppBackend

    jinja_only = SINGLE_TURN_ONLY_HELP.replace("         --simple-io",
                                               "         --jinja\n         --simple-io")
    backend = LlamaCppBackend(gguf_path=tmp_path / "m.gguf",
                              binary=tmp_path / "llama-cli",
                              runner=_fake_runner([], stdout=jinja_only))

    with pytest.raises(RuntimeError, match="chat template"):
        backend.generate(make_passkey_items(tokenizer, depths=(256,),
                                            per_depth=1, seed=1)[0])


def test_llama_cpp_backend_refuses_a_templated_single_turn_by_default(
        tmp_path, tokenizer):
    """`-st` exits, but it still templates -- so it is not a substitute.

    The pinned build advertises `-st` and no `-no-cnv`, and the backend used to
    accept it on the grounds that it could drive the binary. It could; it was
    measuring the chat template. On the released base weights the copy-control
    scores 1.0 through torch and 0.0 through this path, emitting the template's
    own `assistant` marker, and phase 6 stage A recorded 0.0 for every arm at
    every depth as a result -- a number indistinguishable from "these proxies
    cannot retrieve".

    So the fallback is refused, and the message has to name the remedy: the
    next question after "it stopped working" is always "then what do I run?".
    """
    from scripts.retrieval_eval import LlamaCppBackend

    backend = LlamaCppBackend(gguf_path=tmp_path / "m.gguf",
                              binary=tmp_path / "llama-cli",
                              runner=_fake_runner([],
                                                  stdout=SINGLE_TURN_ONLY_HELP))

    with pytest.raises(RuntimeError, match="chat template") as raised:
        backend.generate(make_passkey_items(tokenizer, depths=(256,),
                                            per_depth=1, seed=1)[0])
    assert "LLAMA_BUILD_UI=OFF" in str(raised.value)
    assert "--allow-chat-template" in str(raised.value)


def test_templated_single_turn_is_available_when_asked_for(tmp_path, tokenizer):
    """The refusal is a default, not a wall.

    An instruct checkpoint is scored through its chat template anyway, so on
    that model the templated turn is the correct mode rather than a defect. The
    caller says so explicitly, and `template_mode` records which one ran.
    """
    from scripts.retrieval_eval import LlamaCppBackend

    calls = []
    backend = LlamaCppBackend(gguf_path=tmp_path / "m.gguf",
                              binary=tmp_path / "llama-cli",
                              allow_chat_template=True,
                              runner=_fake_runner(calls,
                                                  stdout=SINGLE_TURN_ONLY_HELP))

    backend.generate(make_passkey_items(tokenizer, depths=(256,), per_depth=1,
                                        seed=1)[0])

    command = [call for call in calls if "--help" not in call["command"]][0]["command"]
    assert "-st" in command
    assert "-no-cnv" not in command
    assert backend.template_mode() == "chat-single-turn"


def test_raw_completion_is_preferred_over_a_templated_turn(tmp_path, tokenizer):
    """A binary offering both must be driven raw, opt-in or not.

    `_BARE_FLAGS` resolved the conversation switch as "first supported alias
    wins", which made the choice depend on the order of a tuple rather than on
    which mode the harness needs. A build advertising both would then be driven
    templated the moment somebody reordered it.
    """
    from scripts.retrieval_eval import LlamaCppBackend

    calls = []
    both = HELP_TEXT.replace("-no-cnv, --no-conversation",
                             "-no-cnv, --no-conversation\n  -st,   --single-turn")
    backend = LlamaCppBackend(gguf_path=tmp_path / "m.gguf",
                              binary=tmp_path / "llama-cli",
                              allow_chat_template=True,
                              runner=_fake_runner(calls, stdout=both))

    backend.generate(make_passkey_items(tokenizer, depths=(256,), per_depth=1,
                                        seed=1)[0])

    command = [call for call in calls if "--help" not in call["command"]][0]["command"]
    assert "-no-cnv" in command
    assert "-st" not in command
    assert backend.template_mode() == "raw-completion"


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

"""Score the retrieval tasks through PyTorch FP16 and stock llama.cpp GGUF.

Both paths answer the *same* generated items, so the FP16-vs-Q4_0 comparison is
paired at the item level rather than a difference of two aggregates. That
matters here more than usual: retrieval accuracy at 150M is low and lumpy, and
an unpaired difference of two ~10% scores has an error bar wide enough to hide
the entire effect a quantization gate is trying to detect.

Two design choices are worth stating outright.

**The control gate runs before the model.** `verify_items` answers every
generated item with `OracleBackend`, which reads the answer out of the prompt
text. If that is not 100%, the items are malformed and no model run can be
interpreted, so the evaluation aborts instead of producing a plausible-looking
zero. This is the plan's "synthetic controls score 100% where expected" check,
enforced at the point where it is cheap.

**llama.cpp flags are probed, not assumed.** The binary is built from a pinned
upstream commit that this code does not control, and llama-cli's flag set moves
(`-no-cnv` arrived in 2025; `--no-display-prompt` has been renamed before). A
wrong flag is not a soft failure -- llama-cli exits non-zero and an unattended
sweep loses its arm. So the backend reads `--help` once and includes only the
optional flags that binary advertises, while the flags it cannot run without
(`-m`, `-f`, `-n`, `-c`, `--temp`) are required and their absence is an error.
Nothing about the model or the GGUF is modified: this is stock llama.cpp,
invoked as a user would.

**Probing a flag is not the same as probing a mode, which cost this phase a
column.** `-st`/`--single-turn` and `-no-cnv`/`--no-conversation` both make
llama-cli exit instead of waiting at its prompt, so a backend that only asks
"can I drive this binary" accepts either. They are not interchangeable: `-st`
still applies the model's chat template, so a base model asked to continue
`The phrase is:` is handed a user/assistant transcript instead. Measured on the
released base weights, that is exact_match 1.0 through torch against 0.0 here,
with the template's own `assistant` marker in the output. Phase 6 stage A read
0.0 for every arm at every depth on the strength of it. `supported_flags`
therefore refuses a build that offers only the templated turn unless the caller
opts in for a model that is chat-templated anyway, and `template_mode` records
which mode produced a card.

**A templated-only build is not necessarily a blocked one.** This box's pinned
`llama-cli` is a UI build: it advertises `-st` and no `-no-cnv`, and the recorded
remedy was to rebuild llama.cpp with `-DLLAMA_BUILD_UI=OFF`. `--probe` says
otherwise -- it also advertises `--jinja` and `--chat-template-file`, and a
pass-through template renders the messages verbatim, so the templated turn feeds
the model exactly the prompt as written. That is raw completion on a stock,
unmodified binary, which is the constraint that mattered; no rebuild, and no
custom build to caveat the results with. The backend takes that route by itself
when it is the only honest one available, reports `raw-completion`, and records
`raw_completion_via` plus the template text so the mechanism stays auditable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from daedalus.retrieval import (  # noqa: E402
    DEPTHS,
    OracleBackend,
    RetrievalItem,
    make_all_items,
    score_items,
    summarize,
)
from daedalus.scorecard import (  # noqa: E402
    ArtifactRef,
    Provenance,
    Scorecard,
    sha256_file,
    write_scorecard,
)


# One generation is a few dozen tokens on a 150M model; a minute is ~50x the
# measured cost and exists only to turn a wedged binary into a traceback, the
# same reasoning as export.py's bounds.
GENERATION_TIMEOUT_S = 60.0
HELP_TIMEOUT_S = 30.0

# Flags the backend cannot work without. Their absence means the binary is not
# llama-cli, and guessing further would waste an unattended run.
_REQUIRED_FLAGS = ("-m", "-f", "-n", "-c", "--temp")

# Flags that improve determinism or output hygiene but whose spelling has
# changed across upstream releases. Included only when `--help` advertises them.
#
# Both spellings of the conversation switch are listed. Upstream has shipped it
# as `-no-cnv` and as `--no-conversation`, and the pinned build advertises only
# the long form -- so a probe that knew the short form alone found nothing, left
# conversation mode on, and every item died waiting for a person to type.
_OPTIONAL_FLAGS = ("--top-k", "-s", "-t", "-ngl", "--no-warmup", "-no-cnv",
                   "--no-conversation", "-st", "--single-turn",
                   "--no-display-prompt")

# Disabling chat outright: the prompt reaches the model as written, which is the
# only mode a *base*-model completion harness may score through. Passing two
# aliases of one switch would be a duplicate argument, so the first supported
# entry wins.
_NO_CHAT_FLAGS = ("-no-cnv", "--no-conversation")

# Exiting after one turn -- but a *templated* one. Not a substitute for the
# above, and treating it as one is the defect this split exists to prevent: see
# `supported_flags`.
_SINGLE_TURN_FLAGS = ("-st", "--single-turn")

# Every way out of an interactive prompt, for the probe and for provenance.
_CONVERSATION_FLAGS = _NO_CHAT_FLAGS + _SINGLE_TURN_FLAGS

# A pass-through Jinja template renders the messages verbatim, so on a build
# that offers these two the *templated* turn feeds the model the prompt exactly
# as written -- raw completion by another route, on a stock binary, with no
# rebuild. Reported by `probe_binary` because whether this box has that route is
# the question the retrieval blocker turns on, and it is answerable from
# `--help` rather than from a guess.
_PASSTHROUGH_FLAGS = ("--jinja", "--chat-template-file")

# Binaries a llama.cpp build may ship that complete a prompt without a chat UI.
# The probe reports which of them exist beside `llama-cli`, so "is there another
# way to run this build" is answered by the directory rather than assumed.
_COMPLETION_BINARIES = ("llama-completion", "llama-simple", "llama-simple-chat",
                        "llama-server", "llama-perplexity", "llama-cli",
                        "llama-quantize", "llama-bench")

# The pass-through template itself. It emits the concatenated message contents
# and nothing else -- no role markers, and `add_generation_prompt` is ignored
# rather than answered with an `assistant` header, which is the exact token that
# gave phase 6 its zeros. With one user message carrying the prompt, the
# rendered string *is* the prompt, so the templated turn and raw completion
# become the same bytes.
#
# Whitespace control on every delimiter is load-bearing: Jinja emits the literal
# newlines around a bare `{% for %}` otherwise, and a leading newline before a
# passkey prompt is a different prompt.
PASSTHROUGH_TEMPLATE = (
    "{%- for message in messages -%}{{ message['content'] }}{%- endfor -%}"
)

# Bare switches, in the order they are appended. The conversation entry is
# resolved by `_conversation_flag`, which is where the raw/templated choice is
# made rather than by "first supported wins".
_BARE_FLAGS = ("--no-warmup", _CONVERSATION_FLAGS, "--no-display-prompt")


class ControlFailure(RuntimeError):
    """Raised when the synthetic controls do not score 100%."""


_CONVERSATION_HELP = re.compile(
    r"^.*(conversation|chat|interactive|single-turn).*$",
    flags=re.MULTILINE | re.IGNORECASE)


def _conversation_help_lines(help_text: str, limit: int = 12) -> str:
    """The binary's own help lines about chat mode, for a rename we did not see.

    Upstream has renamed this switch more than once. Quoting what this binary
    actually says turns "then what does it call it?" into something the error
    message already answers, instead of a guess per rebuild.
    """

    lines = [match.group(0).strip()
             for match in _CONVERSATION_HELP.finditer(help_text)]
    return "\n".join(f"  {line}" for line in lines[:limit]) if lines else "  (none)"


def _advertises(help_text: str, flag: str) -> bool:
    """Does this `--help` offer `flag`, as a whole word rather than a substring?

    `-s` is inside `--seed`, `-t` is inside `--temp`; a plain `in` test says yes
    to both and the binary then rejects the argv. Shared by the backend and the
    probe so the two cannot answer the same question differently.
    """

    return bool(re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", help_text))


def probe_binary(binary, *, runner: Callable[..., subprocess.CompletedProcess]
                 = subprocess.run) -> dict:
    """What this llama.cpp build can actually be driven as, as data.

    Deliberately non-raising, which is the whole point of having it separate
    from `supported_flags`. That guard's job is to stop a run that would measure
    the wrong thing, so on this box's build it raises -- and an operator asking
    "then what *does* this binary support?" gets an exception instead of an
    answer, has no approved way to run `llama-cli --help` directly, and is left
    inferring the build's capabilities from a traceback.

    Phase 6 spent a column on exactly that gap: `-st` and `-no-cnv` were treated
    as interchangeable for months because nothing recorded, per binary, which
    modes were on offer. This returns the flag families separately, reports the
    pass-through-template route that would make a templated-only build usable
    without a rebuild, and lists the sibling binaries that might complete a
    prompt without a chat UI at all.
    """

    path = Path(binary)
    report: dict = {"binary": str(path), "exists": path.exists()}

    siblings = []
    if path.parent.is_dir():
        siblings = sorted(name for name in _COMPLETION_BINARIES
                          if (path.parent / name).exists())
    report["siblings"] = siblings

    if not path.exists():
        report["error"] = "binary does not exist"
        return report

    try:
        result = runner([str(path), "--help"], capture_output=True, text=True,
                        timeout=HELP_TIMEOUT_S, stdin=subprocess.DEVNULL)
    except Exception as error:  # noqa: BLE001 - a diagnostic reports, never bites
        report["error"] = f"{type(error).__name__}: {error}"
        return report

    help_text = (result.stdout or "") + (result.stderr or "")
    report["help_bytes"] = len(help_text)
    report["required_missing"] = [flag for flag in _REQUIRED_FLAGS
                                  if not _advertises(help_text, flag)]
    report["raw_completion_flags"] = [flag for flag in _NO_CHAT_FLAGS
                                      if _advertises(help_text, flag)]
    report["single_turn_flags"] = [flag for flag in _SINGLE_TURN_FLAGS
                                   if _advertises(help_text, flag)]
    report["passthrough_flags"] = [flag for flag in _PASSTHROUGH_FLAGS
                                   if _advertises(help_text, flag)]
    report["optional_flags"] = [flag for flag in _OPTIONAL_FLAGS
                                if _advertises(help_text, flag)]
    # The one line a reader wants: can a *base* model be scored through this
    # binary as written, and if not, is there a route that needs no rebuild.
    if report["raw_completion_flags"]:
        report["base_model_route"] = "raw-completion"
    elif len(report["passthrough_flags"]) == len(_PASSTHROUGH_FLAGS) \
            and report["single_turn_flags"]:
        report["base_model_route"] = "passthrough-template"
    else:
        report["base_model_route"] = "none"
    report["conversation_help"] = _conversation_help_lines(help_text)
    return report


def _tail(stream, limit: int = 800) -> str:
    """The last of whatever a killed child managed to write, as text."""

    if stream is None:
        return ""
    if isinstance(stream, bytes):
        stream = stream.decode("utf-8", "replace")
    return stream.strip()[-limit:]


def parse_depths(value: str):
    return tuple(int(part) for part in value.split(",") if part.strip())


def verify_items(items: Sequence[RetrievalItem]) -> Dict[str, float]:
    """Answer every item from its own prompt; refuse to continue below 100%.

    A failure here is a formatter bug -- a dropped needle, a query whose binding
    is absent, an answer that cannot be extracted -- not a model result.
    """

    records = score_items(items, OracleBackend().generate_all(items))
    metrics = summarize(items, records)
    if metrics["exact_match"] < 1.0:
        broken = [record["id"] for record in records if not record["correct"]]
        raise ControlFailure(
            f"synthetic control scored {metrics['exact_match']:.3f}, expected "
            f"1.0; {len(broken)} malformed item(s), first: {broken[:5]}")
    return metrics


# -------------------------------------------------------- llama.cpp backend ---

class LlamaCppBackend:
    """Greedy generation through an unmodified `llama-cli`."""

    def __init__(self, gguf_path, binary, *, threads: int = 8, n_ctx: int = 4096,
                 max_new_tokens: int = 24, seed: int = 20260824,
                 timeout_s: float = GENERATION_TIMEOUT_S,
                 require_non_interactive: bool = True,
                 allow_chat_template: bool = False,
                 show_prompt: bool = False,
                 runner: Callable[..., subprocess.CompletedProcess] = subprocess.run):
        self.gguf_path = Path(gguf_path)
        self.binary = Path(binary)
        self.threads = threads
        self.n_ctx = n_ctx
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.timeout_s = timeout_s
        self.require_non_interactive = require_non_interactive
        self.allow_chat_template = allow_chat_template
        self.show_prompt = show_prompt
        self.runner = runner
        self._supported: Optional[set] = None

    # -- flag probing --------------------------------------------------------
    def supported_flags(self) -> set:
        if self._supported is not None:
            return self._supported
        result = self.runner([str(self.binary), "--help"], capture_output=True,
                             text=True, timeout=HELP_TIMEOUT_S,
                             stdin=subprocess.DEVNULL)
        help_text = (result.stdout or "") + (result.stderr or "")
        missing = [flag for flag in _REQUIRED_FLAGS
                   if not _advertises(help_text, flag)]
        if missing:
            raise RuntimeError(
                f"{self.binary} does not advertise required flags {missing}; "
                "this does not look like llama-cli")
        # `_PASSTHROUGH_FLAGS` are probed but deliberately not in
        # `_OPTIONAL_FLAGS`: `_command` appends them itself, as a pair with the
        # template path, so listing them among the bare flags would emit a
        # dangling `--chat-template-file`. `resolved_flags` reads
        # `_OPTIONAL_FLAGS` for the same reason -- the pass-through route is
        # reported by `raw_completion_via`, not as a loose switch.
        supported = {flag for flag in
                     _REQUIRED_FLAGS + _OPTIONAL_FLAGS + _PASSTHROUGH_FLAGS
                     if _advertises(help_text, flag)}
        # Modern `llama-cli` starts a chat when the model carries a template,
        # and then waits at its `> ` prompt. With no switch to turn that off,
        # every item burns the full generation timeout and the run learns
        # nothing, so refuse now and name what the binary does advertise --
        # upstream has spelled this switch more than one way.
        if self.require_non_interactive and \
                not any(flag in supported for flag in _CONVERSATION_FLAGS):
            raise RuntimeError(
                f"{self.binary} advertises no way to leave conversation mode "
                f"(looked for {list(_CONVERSATION_FLAGS)}); it would wait at an "
                "interactive prompt for every item. Optional flags it does "
                f"advertise: {sorted(supported - set(_REQUIRED_FLAGS))}.\n"
                "Help lines mentioning conversation/chat/interactive:\n"
                f"{_conversation_help_lines(help_text)}\n"
                "Pass require_non_interactive=False only for a binary old "
                "enough to predate conversation mode.")

        # Exiting is not the same as not templating, and this is where that
        # distinction is enforced. `-st` runs one turn and exits, so a run
        # driven by it *completes* -- it simply measures the wrong thing: the
        # model is handed its chat template, and a base model asked to continue
        # `The phrase is:` instead sees a user/assistant transcript.
        #
        # Measured on this box's pinned build, which advertises `-st` and no
        # `-no-cnv`: the released base model scores exact_match 1.0 on the
        # copy-control through the torch backend and 0.0 through this one, on
        # the same weights, emitting `cannibalassistant` -- the template's own
        # role marker. Phase 6 stage A then read 0.0 for every arm at every
        # depth and every cell landed `no-power`, which is indistinguishable
        # from "these proxies cannot retrieve" and is not what happened.
        #
        # So a templated turn is refused rather than silently substituted. The
        # opt-in exists because the mode is correct for an *instruct* model,
        # whose scoring is chat-templated anyway.
        if self.require_non_interactive and not self.allow_chat_template and \
                not any(flag in supported for flag in _NO_CHAT_FLAGS) and \
                not self._passthrough_available(supported):
            raise RuntimeError(
                f"{self.binary} advertises only a templated single turn "
                f"({[f for f in _SINGLE_TURN_FLAGS if f in supported]}), none "
                f"of {list(_NO_CHAT_FLAGS)}, and not both of "
                f"{list(_PASSTHROUGH_FLAGS)}, so every prompt would reach "
                "the model wrapped in its chat template. That is not a "
                "base-model completion measurement: on the released base "
                "weights it scores the copy-control 0.0 here against 1.0 "
                "through the torch backend.\n"
                "Remedy: build `llama-cli` with -DLLAMA_BUILD_UI=OFF (the UI "
                "build is what turned conversation on by default and dropped "
                "the switch), or point --llama-cli at a `llama-completion` "
                "binary. Run `--probe` first: a build offering --jinja and "
                "--chat-template-file needs neither, because a pass-through "
                "template renders the prompt verbatim.\n"
                "Pass --allow-chat-template only to score a model whose "
                "prompts are chat-templated by design, such as an instruct "
                "checkpoint.\n"
                "Help lines mentioning conversation/chat/interactive:\n"
                f"{_conversation_help_lines(help_text)}")
        self._supported = supported
        return self._supported

    @staticmethod
    def _passthrough_available(supported) -> bool:
        """Can a templated-only build still be driven raw, via its own template?

        Both flags, not either: `--chat-template-file` without a Jinja engine is
        not applied, and `--jinja` alone leaves the model's own template in
        place. This box's pinned build advertises both (`--probe`), which is why
        the retrieval column does not in fact need llama.cpp rebuilt.
        """

        return all(flag in supported for flag in _PASSTHROUGH_FLAGS)

    def uses_passthrough_template(self) -> bool:
        """True when the raw prompt reaches the model *through* the chat path.

        Only when there is no honest alternative. A build that advertises
        `-no-cnv` is driven with it, because leaving the chat machinery out
        entirely is fewer moving parts than neutralising it -- and because an
        upstream change to Jinja handling must not silently alter what a
        scorecard measured.
        """

        supported = self.supported_flags()
        if any(flag in supported for flag in _NO_CHAT_FLAGS):
            return False
        if not self._passthrough_available(supported):
            return False
        return any(flag in supported for flag in _SINGLE_TURN_FLAGS)

    def _conversation_flag(self) -> Optional[str]:
        """The switch that leaves conversation mode, raw completion preferred.

        A single place to make the choice, so `_command` cannot drift from the
        guard above by re-deriving it as "first supported wins" -- which is
        exactly how a templated turn became the default.

        With a pass-through template in play the single-turn flag is back in the
        candidate list *without* `allow_chat_template`: `-st` then means "one
        turn, then exit", and the turn it runs is no longer templated.
        """

        supported = self.supported_flags()
        candidates = _NO_CHAT_FLAGS + (
            _SINGLE_TURN_FLAGS
            if self.allow_chat_template or self.uses_passthrough_template()
            else ())
        return next((flag for flag in candidates if flag in supported), None)

    def template_mode(self) -> str:
        """Whether this binary will be driven raw or through a chat template.

        Recorded in provenance: two scorecards that differ in this differ in
        what was measured, and no other field would say so. A pass-through turn
        reports `raw-completion` because that is what the model is fed --
        `raw_completion_via` carries *how*, so the mechanism stays auditable
        without the gate having to enumerate mechanisms.
        """

        if self.uses_passthrough_template():
            return "raw-completion"
        flag = self._conversation_flag()
        if flag in _NO_CHAT_FLAGS:
            return "raw-completion"
        if flag in _SINGLE_TURN_FLAGS:
            return "chat-single-turn"
        return "unknown"

    def raw_completion_via(self) -> Optional[str]:
        """Which mechanism produced a raw completion, or None if none did."""

        if self.uses_passthrough_template():
            return "passthrough-template"
        if self._conversation_flag() in _NO_CHAT_FLAGS:
            return "no-conversation-flag"
        return None

    def _command(self, prompt_file: str,
                 template_file: Optional[str] = None) -> List[str]:
        supported = self.supported_flags()
        command = [str(self.binary), "-m", str(self.gguf_path),
                   "-f", prompt_file,
                   "-n", str(self.max_new_tokens),
                   "-c", str(self.n_ctx),
                   # Temperature zero is the greedy path in llama.cpp; top-k 1
                   # pins it even on builds whose temp-0 handling changed.
                   "--temp", "0"]
        for flag, value in (("--top-k", "1"), ("-s", str(self.seed)),
                            ("-t", str(self.threads)), ("-ngl", "0")):
            if flag in supported:
                command += [flag, value]
        for entry in _BARE_FLAGS:
            # `--no-display-prompt` is what makes stdout the completion alone;
            # dropping it is how an operator sees the text the binary actually
            # fed the model, chat template and all.
            if entry == "--no-display-prompt" and self.show_prompt:
                continue
            if entry is _CONVERSATION_FLAGS:
                match = self._conversation_flag()
            else:
                aliases = (entry,) if isinstance(entry, str) else entry
                match = next((flag for flag in aliases if flag in supported),
                             None)
            if match is not None:
                command.append(match)
        if template_file is not None:
            # Appended after the bare flags so the template wins over anything
            # the model's own metadata would have supplied.
            command += ["--jinja", "--chat-template-file", str(template_file)]
        return command

    def resolved_flags(self) -> List[str]:
        """The optional flags this binary actually accepted, for provenance.

        Two runs of "stock llama.cpp" that differ only in which switches the
        binary advertised are not the same measurement, and no other field in
        the scorecard would record the difference.
        """

        supported = self.supported_flags()
        return [flag for flag in _OPTIONAL_FLAGS if flag in supported]

    # -- generation ----------------------------------------------------------
    def generate(self, item: RetrievalItem) -> str:
        # The prompt goes in a file, never in argv: these prompts run to 2048
        # tokens and contain newlines, and an argv round trip through a shell
        # would be one quoting bug away from silently scoring a truncated
        # context.
        handle, prompt_file = tempfile.mkstemp(prefix="daedalus-retrieval-",
                                               suffix=".txt")
        template_file = None
        try:
            with os.fdopen(handle, "w") as stream:
                stream.write(item.prompt)
            if self.uses_passthrough_template():
                template_handle, template_file = tempfile.mkstemp(
                    prefix="daedalus-passthrough-", suffix=".jinja")
                with os.fdopen(template_handle, "w") as stream:
                    stream.write(PASSTHROUGH_TEMPLATE)
            # A closed stdin, not just the conversation flag. The flag's
            # spelling is upstream's to change; EOF on stdin is not, and it
            # turns "wait for a person who is not there" into a clean exit.
            try:
                result = self.runner(self._command(prompt_file, template_file),
                                     capture_output=True, text=True,
                                     timeout=self.timeout_s,
                                     stdin=subprocess.DEVNULL)
            except subprocess.TimeoutExpired as expired:
                # `subprocess` kills the child and carries whatever it had
                # already written. Discarding that leaves "timed out after 60s"
                # as the entire diagnosis, which cannot distinguish a binary
                # stuck before the model loaded from one still decoding -- a
                # distinction worth a lot at 60 s an item.
                raise RuntimeError(
                    f"llama-cli did not finish item {item.id} within "
                    f"{self.timeout_s:.0f}s. Last output:\n"
                    f"{_tail(expired.stderr)}\n{_tail(expired.stdout)}"
                ) from expired
        finally:
            os.unlink(prompt_file)
            if template_file is not None:
                os.unlink(template_file)
        if result.returncode != 0:
            raise RuntimeError(
                f"llama-cli exited {result.returncode} for item {item.id}: "
                f"{(result.stderr or '').strip()[-500:]}")
        return _clean_completion(result.stdout or "", item.prompt)

    def generate_all(self, items: Sequence[RetrievalItem]) -> List[str]:
        return [self.generate(item) for item in items]


_NOISE = re.compile(
    r"\[end of text\]|^llama_perf.*$|^llama_print_timings.*$|^main:.*$",
    flags=re.MULTILINE)

# The chat UI's own footer. Builds that run this harness in conversation mode
# close with a throughput line and a goodbye, both on stdout.
_CHAT_FOOTER = re.compile(r"\[ *Prompt:[^\]]*\]|^Exiting\.\.\..*$",
                          flags=re.MULTILINE)

# The chat UI elides a long echoed prompt rather than printing it whole, so for
# the deeper retrieval items there is no full prompt in stdout to anchor on.
_ECHO_TRUNCATION = " ... (truncated)"

# Text that must never reach a scorecard. If any of it survives extraction the
# layout has changed and the completion was not found, which previously scored
# as a confident zero.
_BANNER_SIGNATURES = ("Loading model...", "available commands:")


def _clean_completion(text: str, prompt: Optional[str] = None) -> str:
    """The model's completion alone, from whatever the binary put on stdout.

    Stripping known noise markers is not enough on a build that runs the chat
    UI: it prints a loading spinner, an ASCII-art banner, a build/model block
    and a command list to *stdout*, ignores `--no-display-prompt`, and echoes
    the prompt. Measured against the pinned binary, that made every recorded
    completion the string "Loading model..." -- a silently zero score that
    looks exactly like a model that cannot retrieve.

    So the completion is located rather than filtered: everything the binary
    printed before the prompt it echoed is preamble by construction, whatever
    that preamble happens to contain. The *first* occurrence is the echo -- a
    model that repeats its prompt back (these do) would otherwise have its own
    repetition mistaken for the echo and its answer cut away.

    A binary that honours `--no-display-prompt` echoes nothing, the prompt is
    not found, and the older noise-stripping path is used unchanged.

    Whatever survives is checked for banner text before it is returned. Every
    anchor here describes one binary's UI, which upstream is free to change; a
    changed UI must surface as a failed run rather than as a model that
    suddenly cannot retrieve.
    """

    if prompt:
        marker = prompt.strip()
        index = text.find(marker)
        if index >= 0:
            text = text[index + len(marker):]
        else:
            # Long prompts are echoed truncated ("... (truncated)"), so the
            # full prompt never appears and there is nothing to match on.
            cut = text.find(_ECHO_TRUNCATION)
            if cut >= 0:
                text = text[cut + len(_ECHO_TRUNCATION):]
    cleaned = _CHAT_FOOTER.sub("", _NOISE.sub("", text)).strip()
    leaked = [signature for signature in _BANNER_SIGNATURES if signature in cleaned]
    if leaked:
        raise RuntimeError(
            f"llama-cli's banner survived completion extraction ({leaked}); the "
            "binary's output layout is not the one this backend knows, and "
            "scoring it would record banner text as the model's answer. "
            f"Got: {cleaned[:300]!r}")
    return cleaned


# ------------------------------------------------------------ torch backend ---

class TorchBackend:
    """Greedy decoding from a PyTorch model, matching the GGUF path's sampling."""

    def __init__(self, model, tokenizer, *, device: str = "cpu",
                 max_new_tokens: int = 24, eos_id: Optional[int] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.eos_id = eos_id

    def generate(self, item: RetrievalItem) -> str:
        import torch

        try:
            ids = self.tokenizer.encode(item.prompt, add_special_tokens=False)
        except TypeError:
            ids = self.tokenizer.encode(item.prompt)
        was_training = getattr(self.model, "training", False)
        self.model.eval()
        generated: List[int] = []
        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                window = torch.tensor([ids], device=self.device)
                logits, _, _ = self.model(window, targets=None, return_logits=True)
                next_id = int(torch.argmax(logits[0, -1]).item())
                if self.eos_id is not None and next_id == self.eos_id:
                    break
                generated.append(next_id)
                ids.append(next_id)
        self.model.train(was_training)
        return self.tokenizer.decode(generated).strip()

    def generate_all(self, items: Sequence[RetrievalItem]) -> List[str]:
        return [self.generate(item) for item in items]


# -------------------------------------------------------------- scorecards ---

def run_retrieval(items_by_task: Dict[str, List[RetrievalItem]], backend, *,
                  out_dir, artifact: ArtifactRef, tokenizer_ref: ArtifactRef,
                  seed: int, git_sha: str,
                  runtime: Optional[dict] = None,
                  verify: bool = True) -> Dict[str, Dict[str, Path]]:
    """Score every task and write one scorecard (plus item sidecar) per task."""

    out_dir = Path(out_dir)
    if verify:
        for items in items_by_task.values():
            verify_items(items)

    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    written: Dict[str, Dict[str, Path]] = {}
    for task, items in items_by_task.items():
        records = score_items(items, backend.generate_all(items))
        # The exact prompt rides with each outcome: a retrieval number is only
        # checkable if the text that produced it can be read back.
        for item, record in zip(items, records):
            record["prompt"] = item.prompt
        card = Scorecard(
            kind="retrieval",
            name=f"retrieval-{task}",
            provenance=Provenance(
                artifact=artifact, tokenizer=tokenizer_ref, seed=seed,
                git_sha=git_sha, bpb_mode="not-applicable",
                runtime=dict(runtime or {}),
            ),
            metrics=summarize(items, records),
            created_at=created_at,
            items=records,
        )
        written[task] = write_scorecard(out_dir / f"retrieval-{task}.json", card)
    return written


# --------------------------------------------------------------------- cli ---

def _git_short_sha() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True)
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _artifact_ref(path: Optional[str], kind: str) -> ArtifactRef:
    if path is None:
        return ArtifactRef(path="<none>", sha256="0" * 64, kind=kind)
    return ArtifactRef(path=path, sha256=sha256_file(path), kind=kind)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("torch", "llama-cpp", "oracle"),
                        required=True)
    parser.add_argument("--gguf", help="GGUF artifact for the llama-cpp backend")
    parser.add_argument("--llama-cli", default="/opt/llama.cpp/build/bin/llama-cli")
    parser.add_argument("--checkpoint", help="Daedalus .pt for the torch backend")
    parser.add_argument("--config", default="daedalus-150m")
    parser.add_argument("--tokenizer", default=None,
                        help="tokenizer directory or file recorded in provenance")
    parser.add_argument("--depths", default=",".join(str(d) for d in DEPTHS))
    parser.add_argument("--per-depth", type=int, default=10)
    parser.add_argument("--n-queries", type=int, default=4)
    parser.add_argument("--control-items", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="runs/eval/retrieval")
    parser.add_argument("--allow-chat-template", action="store_true",
                        help="permit a binary that only offers a templated "
                             "single turn (-st). Correct for an instruct "
                             "checkpoint, whose prompts are chat-templated by "
                             "design; invalid for base-model completion, where "
                             "it scores the template rather than the model.")
    parser.add_argument("--probe", action="store_true",
                        help="report what --llama-cli can be driven as, as "
                             "JSON, and exit without scoring anything. The "
                             "scoring guards raise on a build that cannot "
                             "measure a base model, so this is the only way to "
                             "read that build's actual capabilities.")
    parser.add_argument("--show-prompt", action="store_true",
                        help="keep llama-cli's prompt echo, so the text the "
                             "binary actually fed the model -- chat template "
                             "included -- can be read back. Diagnostic only: "
                             "the echo lands in the recorded completion.")
    args = parser.parse_args(argv)

    if args.probe:
        # Before the tokenizer and before the items: a probe asks about the
        # binary, and a build that cannot be probed is exactly the situation
        # where downloading a tokenizer first would fail for an unrelated reason.
        print(json.dumps(probe_binary(args.llama_cli), indent=2, sort_keys=True))
        return 0

    from daedalus.data import get_tokenizer

    tokenizer = get_tokenizer()
    items = make_all_items(tokenizer, depths=parse_depths(args.depths),
                           per_depth=args.per_depth, seed=args.seed,
                           n_queries=args.n_queries,
                           control_items=args.control_items)

    runtime = {"backend": args.backend, "device": args.device,
               "max_new_tokens": args.max_new_tokens}
    if args.backend == "llama-cpp":
        if not args.gguf:
            parser.error("--gguf is required for the llama-cpp backend")
        backend = LlamaCppBackend(args.gguf, args.llama_cli, threads=args.threads,
                                  n_ctx=args.n_ctx,
                                  max_new_tokens=args.max_new_tokens,
                                  seed=args.seed, show_prompt=args.show_prompt,
                                  allow_chat_template=args.allow_chat_template)
        artifact = _artifact_ref(args.gguf,
                                 "gguf-q4_0" if "q4" in Path(args.gguf).name.lower()
                                 else "gguf-f16")
        runtime["llama_cli"] = str(args.llama_cli)
        runtime["threads"] = args.threads
        runtime["llama_cli_flags"] = backend.resolved_flags()
        # Which of the two modes produced this card. A scorecard that does not
        # say cannot be told from one measured the other way.
        runtime["template_mode"] = backend.template_mode()
        runtime["raw_completion_via"] = backend.raw_completion_via()
        if backend.uses_passthrough_template():
            # The template is the measurement here: a card scored through a
            # different pass-through is a different card, and the only way to
            # check that later is to have kept the text.
            runtime["chat_template"] = PASSTHROUGH_TEMPLATE
    elif args.backend == "torch":
        if not args.checkpoint:
            parser.error("--checkpoint is required for the torch backend")
        import torch
        from daedalus.config import PRESETS
        from daedalus.model import Daedalus
        from train import load_checkpoint

        model = Daedalus(PRESETS[args.config]).to(args.device)
        load_checkpoint(args.checkpoint, model, map_location=args.device)
        if args.device.startswith("cuda"):
            model = model.half()
        backend = TorchBackend(model=model, tokenizer=tokenizer,
                               device=args.device,
                               max_new_tokens=args.max_new_tokens, eos_id=0)
        artifact = _artifact_ref(args.checkpoint, "checkpoint")
        artifact.config = args.config
        runtime["torch"] = torch.__version__
    else:
        backend = OracleBackend()
        artifact = _artifact_ref(None, "checkpoint")

    written = run_retrieval(
        items, backend, out_dir=args.out_dir, artifact=artifact,
        tokenizer_ref=_artifact_ref(args.tokenizer, "tokenizer"),
        seed=args.seed, git_sha=_git_short_sha(), runtime=runtime)

    for task, paths in sorted(written.items()):
        print(f"wrote {paths['scorecard']}")
        print(json.dumps(json.loads(Path(paths["scorecard"]).read_text())["metrics"],
                         indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

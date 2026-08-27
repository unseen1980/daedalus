"""Tests for `scripts/code_sft.py probe`.

The module's own rules are tested in `test_code_sft.py`; what is left here is
what only the CLI can get wrong -- a source key nobody has, an index path that
drifted from the one the corpus build filters against, and a verdict that says
"every source resolved" while a problem list is non-empty.
"""

import json

import pytest

from daedalus.code_sft import SFT_SOURCES, SFTSource, probe_sources
from scripts import code_sft as cli


def test_the_eval_index_default_matches_the_corpus_build_s():
    """Two literals of one path is the drift this pins. A probe filtering
    against a different index than the corpus build measures a yield no build
    will ever see, and nothing in either output would say so."""

    from scripts import codeprep

    assert cli.DEFAULT_EVAL_INDEX_PATH == codeprep.DEFAULT_EVAL_INDEX_PATH


def test_an_unknown_source_key_is_refused_before_any_row_is_read(capsys):
    code = cli._cli(["probe", "--source", "not-a-source", "--no-decontam",
                     "--no-tokenizer"])
    assert code == 2
    assert "unknown source key" in capsys.readouterr().err


def test_a_probe_writes_its_record_before_returning_a_failing_verdict(
        tmp_path, monkeypatch, capsys):
    """A failing probe's whole value is the record of what was in the rows, so
    it is written before the verdict rather than after it."""

    out = tmp_path / "probe.json"
    # A real record over an injected stream rather than a hand-typed dict: the
    # reporter reads a dozen keys and a stub that misses one fails as a
    # KeyError about the test rather than about the code.
    one_half = SFTSource(key="code-only", half="code", dataset="d",
                         declared_license="apache-2.0", note="")
    monkeypatch.setattr(
        cli, "probe_sources",
        lambda sources, **kwargs: probe_sources(
            [one_half], rows=5,
            stream=lambda source: [{"messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "```python\ndef f(:\n```"}]}]))
    code = cli._cli(["probe", "--no-decontam", "--no-tokenizer",
                     "--json-out", str(out)])
    assert code == 3
    written = json.loads(out.read_text())
    assert written["problems"], "the record must carry the problems it exited on"
    err = capsys.readouterr().err
    assert "the admission gate refused every one" in err
    assert "the general half kept no rows" in err


def test_the_probe_defaults_to_every_source_in_the_table(monkeypatch):
    seen = {}

    def fake(sources, **kwargs):
        seen["keys"] = [s.key for s in sources]
        return {"schema": 1, "sources": [], "alternatives": []}

    monkeypatch.setattr(cli, "probe_sources", fake)
    # No sources means no rows kept for either half, so the verdict is 3; the
    # selection is what this asserts.
    cli._cli(["probe", "--no-decontam", "--no-tokenizer"])
    assert seen["keys"] == [s.key for s in SFT_SOURCES]


@pytest.mark.parametrize("flag,key", [("--no-decontam", "indexes"),
                                      ("--no-tokenizer", "tokenizer")])
def test_the_opt_outs_reach_the_probe(monkeypatch, flag, key):
    seen = {}
    monkeypatch.setattr(cli, "probe_sources",
                        lambda sources, **kwargs: seen.update(kwargs) or
                        {"schema": 1, "sources": [], "alternatives": []})
    cli._cli(["probe", "--no-decontam", "--no-tokenizer"])
    assert not seen[key]

"""Permission rules only apply to a trusted workspace."""

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "daedalus_trust_workspace", ROOT / "ops" / "vast" / "trust_workspace.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_untrusted_workspace_becomes_trusted(tmp_path):
    module = _module()
    config = tmp_path / "claude.json"
    config.write_text(json.dumps({"projects": {"/workspace/daedalus": {}}}))

    assert module.is_trusted(config, "/workspace/daedalus") is False
    assert module.trust_workspace(config, "/workspace/daedalus") is True
    assert module.is_trusted(config, "/workspace/daedalus") is True


def test_trusting_twice_changes_nothing(tmp_path):
    module = _module()
    config = tmp_path / "claude.json"
    config.write_text(json.dumps({"projects": {}}))

    assert module.trust_workspace(config, "/workspace/daedalus") is True
    assert module.trust_workspace(config, "/workspace/daedalus") is False


def test_unrelated_account_state_survives(tmp_path):
    module = _module()
    config = tmp_path / "claude.json"
    config.write_text(
        json.dumps({"userID": "keep-me", "projects": {"/other": {"a": 1}}})
    )

    module.trust_workspace(config, "/workspace/daedalus")
    written = json.loads(config.read_text())

    assert written["userID"] == "keep-me"
    assert written["projects"]["/other"] == {"a": 1}


def test_a_missing_config_is_created_private(tmp_path):
    module = _module()
    config = tmp_path / "claude.json"

    module.trust_workspace(config, "/workspace/daedalus")

    assert module.is_trusted(config, "/workspace/daedalus") is True
    assert oct(config.stat().st_mode)[-3:] == "600"


def test_an_unparsable_config_is_never_rewritten(tmp_path):
    module = _module()
    config = tmp_path / "claude.json"
    config.write_text("{not json")

    with pytest.raises(SystemExit):
        module.trust_workspace(config, "/workspace/daedalus")

    assert config.read_text() == "{not json"


def test_the_installer_trusts_the_workspace_it_installs():
    installer = (ROOT / "ops/vast/install_supervisor.sh").read_text()

    assert "trust_workspace.py" in installer

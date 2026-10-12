from __future__ import annotations

import json

import pytest

from subs_pool import agent_cli, cli
from subs_pool.modules.codex import cli as codex_cli
from subs_pool.modules.codex.accounts import AccountStore
from fakes import write_auth_fixture


def test_human_account_commands_use_singular_grammar(capsys, tmp_path):
    auth = tmp_path / "auth.json"
    write_auth_fixture(auth)
    assert cli.main(["codex", "account", "import", "work", "--path", str(auth), "--weight", "2"]) == 0
    assert cli.main(["codex", "account", "list"]) == 0
    output = capsys.readouterr().out
    assert "imported work" in output
    assert "work" in output
    assert "quota" not in output


def test_human_invalid_command_is_human_error(capsys):
    with pytest.raises(SystemExit):
        codex_cli.main(["bogus-command"])
    assert "usage:" in capsys.readouterr().err


def test_agent_machine_surface_emits_one_envelope_and_never_human_text(capsys):
    assert agent_cli.main(["codex", "account", "login", "work"]) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["ok"] is False
    assert payload["error"]["code"] == "prompt_required"


def test_agent_unknown_and_serve_are_stable_json_errors(capsys):
    assert agent_cli.main(["serve"]) == 2
    serve = json.loads(capsys.readouterr().out)
    assert serve["error"]["code"] == "unknown_module"
    assert agent_cli.main(["codex", "serve"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "unsupported_command"


def test_serve_rejects_non_loopback_and_keeps_default_port(capsys, monkeypatch):
    assert codex_cli.main(["serve", "--listen", "0.0.0.0:9000", "--api-key", "x"]) == 4
    assert "loopback" in capsys.readouterr().err
    monkeypatch.setattr("subs_pool.modules.codex.cli._store", lambda: AccountStore())
    assert codex_cli.build_parser().parse_args(["serve"]).listen == "127.0.0.1:8765"

"""Frozen: every command of the CLI table points to an importable module with main()."""

import importlib

import pytest

from kpr.cli import COMMANDS, main
from kpr.schemas import SESSIONS


@pytest.mark.parametrize("key", sorted(COMMANDS))
def test_command_target_exists(key):
    module_name, owner, _task = COMMANDS[key]
    assert owner in SESSIONS
    assert callable(importlib.import_module(module_name).main)


def test_list_prints_usage(capsys):
    assert main(["list"]) == 0
    assert "collect" in capsys.readouterr().out


def test_unknown_command_fails(capsys):
    assert main(["nope", "nothing"]) == 2

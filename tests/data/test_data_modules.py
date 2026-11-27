"""Owner: data session. Smoke test: every module of the data session imports."""

import importlib

import pytest

from kpr.cli import COMMANDS

MODULES = sorted({m for m, owner, _t in COMMANDS.values() if owner == "data"})


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)

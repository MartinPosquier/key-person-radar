"""Owner: quant session. Smoke test: every module of the quant session imports."""

import importlib

import pytest

from kpr.cli import COMMANDS

MODULES = sorted({m for m, owner, _t in COMMANDS.values() if owner == "quant"})


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)


def test_calendar_imports():
    importlib.import_module("kpr.utils.calendar")

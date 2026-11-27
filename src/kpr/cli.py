"""Frozen command table: `python -m kpr.cli <group> <command> [args...]`.

Each command points to a module whose `main(argv)` does the work. Sessions implement the
`main` of the modules they own; they never edit this file (PARALLEL_SESSIONS.md §3).
Every module can also be run directly: `python -m kpr.collect.wiki`.
"""

from __future__ import annotations

import importlib
import sys

# (group, command) -> (module, owner, task id in PARALLEL_SESSIONS.md §7)
COMMANDS: dict[tuple[str, str], tuple[str, str, str]] = {
    ("collect", "wikidata"): ("kpr.collect.wikidata", "data", "D2"),
    ("collect", "prices"): ("kpr.collect.prices", "data", "D3"),
    ("collect", "wiki"): ("kpr.collect.wiki", "data", "D4"),
    ("collect", "gdelt"): ("kpr.collect.gdelt", "data", "D5, D8"),
    ("collect", "edgar"): ("kpr.collect.edgar", "data", "D6"),
    ("collect", "trends"): ("kpr.collect.trends", "data", "D7"),
    ("collect", "fnspid"): ("kpr.collect.fnspid", "data", "D8"),
    ("collect", "earnings"): ("kpr.collect.earnings", "data", "D9"),
    ("collect", "headlines"): ("kpr.collect.headlines", "data", "D12"),
    ("build", "clean"): ("kpr.build.clean", "data", "D10"),
    ("build", "panel"): ("kpr.build.panel", "data", "D11"),
    ("build", "returns"): ("kpr.build.returns", "quant", "Q2"),
    ("build", "events"): ("kpr.build.events", "quant", "Q4, Q5"),
    ("build", "splits"): ("kpr.build.splits", "quant", "Q6"),
    ("build", "tone"): ("kpr.build.tone", "ml", "M5"),
    ("build", "features"): ("kpr.build.features", "ml", "M2"),
    ("models", "baselines"): ("kpr.models.baselines", "ml", "M3"),
    ("models", "evaluate"): ("kpr.models.evaluate", "ml", "M4"),
    ("models", "train"): ("kpr.models.train", "ml", "M7"),
    ("models", "tune"): ("kpr.models.tune", "ml", "M7"),
    ("models", "mixed"): ("kpr.models.mixed", "ml", "M8"),
    ("models", "select"): ("kpr.models.select", "ml", "M9"),
    ("models", "explain"): ("kpr.models.explain", "ml", "M11"),
    ("scores", "sensitivity"): ("kpr.scores.sensitivity", "ml", "M10"),
    ("scores", "value"): ("kpr.scores.value", "ml", "M10"),
    ("scores", "smoothing"): ("kpr.scores.smoothing", "ml", "M10"),
    ("scores", "export-app"): ("kpr.scores.export_app", "ml", "M13"),
    ("validate", "placebo"): ("kpr.validation.placebo", "quant", "Q8"),
    ("validate", "robustness"): ("kpr.validation.robustness", "quant", "Q9"),
    ("validate", "natural-experiments"): ("kpr.validation.natural_experiments", "quant", "Q10"),
    ("validate", "hypotheses"): ("kpr.validation.hypotheses", "quant", "Q11"),
    ("validate", "case-nvidia"): ("kpr.validation.case_nvidia", "quant", "Q12"),
}


def _usage() -> str:
    lines = ["usage: python -m kpr.cli <group> <command> [args...]", "", "commands:"]
    for (group, cmd), (module, owner, task) in COMMANDS.items():
        lines.append(f"  {group:<9} {cmd:<20} {module:<36} [{owner} {task}]")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2 or (argv[0], argv[1]) not in COMMANDS:
        print(_usage())
        return 0 if not argv or argv[0] in ("-h", "--help", "list") else 2
    module_name = COMMANDS[(argv[0], argv[1])][0]
    module = importlib.import_module(module_name)
    result = module.main(argv[2:])
    return int(result or 0)


if __name__ == "__main__":
    sys.exit(main())

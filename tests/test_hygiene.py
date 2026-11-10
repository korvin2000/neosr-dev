"""Banned constructs under neosr/. A new offender is fixed, not allowlisted: ALLOW names only the one
module that owns a construct by design (importing by name belongs to the registries, design T7)."""

import re
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "neosr"

PATTERNS = {
    "net_opt": r"\bnet_opt\b",
    "parse_options": r"\bparse_options\b",
    "set_default_device": r"\bset_default_device\b",
    "sys.exit": r"\bsys\.exit\b",
    "sys.argv": r"\bsys\.argv\b",
    "bare except": r"^\s*except\s*:",
    "import by name": r"\b(importlib|pkgutil|__import__)\b",
}


# pattern -> files (relative to neosr/, posix) that may contain it
ALLOW: dict[str, set[str]] = {p: set() for p in PATTERNS} | {
    "import by name": {"utils/registry.py"}
}


def offenders(pattern: str) -> set[str]:
    regex = re.compile(PATTERNS[pattern], re.MULTILINE)
    return {
        f.relative_to(PKG).as_posix()
        for f in PKG.rglob("*.py")
        if regex.search(f.read_text("utf-8"))
    }


@pytest.mark.parametrize("pattern", PATTERNS)
def test_no_new_offenders(pattern: str) -> None:
    assert sorted(offenders(pattern) - ALLOW[pattern]) == []


@pytest.mark.parametrize("pattern", PATTERNS)
def test_allowlist_is_current(pattern: str) -> None:
    """A migrated or deleted file must leave the allowlist, so the list only shrinks."""
    assert sorted(ALLOW[pattern] - offenders(pattern)) == []

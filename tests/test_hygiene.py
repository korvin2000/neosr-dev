"""Banned constructs under neosr/. ALLOW must stay empty; a new offender is fixed, not allowlisted."""

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
    "registry": r"_REGISTRY\b",
}


# pattern -> files (relative to neosr/) that may still contain it; empty since T3-Z
ALLOW: dict[str, set[str]] = {p: set() for p in PATTERNS}


def offenders(pattern: str) -> set[str]:
    regex = re.compile(PATTERNS[pattern], re.MULTILINE)
    return {
        str(f.relative_to(PKG))
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

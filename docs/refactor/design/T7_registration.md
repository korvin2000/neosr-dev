# T7: the five registries, lazy

Status: **decided and implemented** (2026-11-10, session 7, Windows machine; Codex-reviewed, section 8; the
owner's choices in section 9).
Decided already (`DECISIONS.md`, 2026-11-10): `ARCH_`, `LOSS_`, `METRIC_`, `DATASET_`, `MODEL_REGISTRY` return
with `@X_REGISTRY.register()`; no import of every module, no work at import time. This note proposes the
mechanism. It supersedes T3 D3 in form, not in its goals.

## 1. The idea in one paragraph
A registry needs two things: `name -> module` (to import one module) and `name -> object`. The second is free
once the module is imported (`getattr(module, name)`: the type name is the Python name, as today). The first is
written in the source already, as the decorator line above the definition. So the registry **reads the
decorators from the source text** of its package on first use, without importing anything, and is then a lazy
read-only mapping `name -> Row(module, scale, meta)`: the same shape as today's tables, produced from the
decorators instead of by hand. The tables in the five `__init__.py` files are deleted.

```python
# neosr/archs/myarch_arch.py: the only file touched to add an arch
from neosr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register(scale="upscale")      # the ctor kwarg that receives the run scale
class myarch(nn.Module): ...

@ARCH_REGISTRY.register(scale="upscale")      # factories are rows too
def myarch_light(**kwargs): return myarch(dim=32, **kwargs)

@ARCH_REGISTRY.register(scale=None)           # a network without a scale (discriminator)
class mydisc(nn.Module): ...
```

## 2. Mechanisms compared
Measured on this machine (Ryzen 5 5500U, Python 3.14) over `neosr/archs` (56 files, 947 KiB, 105 rows);
`import torch` is 3.7 s here.

| | mechanism | adding an arch | cost per process | state | verdict |
|---|---|---|---|---|---|
| A1 | **decorator is a declaration, the scan is the registry.** `register()` returns the object unchanged; rows come from the scan; the object from `getattr` after importing its one module | one file | read 4.6 ms + regex 6.9 ms + 1 ms for the keyword literals (16 ms cold, measured end to end), once, on first lookup; nothing at import | one cached dict of rows | **proposed** |
| A2 | as A1, but `register()` also stores the object in a dict at import; the lookup returns that object and cross-checks the scan against it | one file | as A1 + one dict insert per decorated definition at import | rows + objects, two sources that must agree | catches one thing A1 cannot at run time: a registered name rebound later in its module (`Net = other`), where A1 returns the final binding. A1 covers it in the test suite instead (section 7, the `ast` oracle: a registered name is bound once) |
| A3 | A1 with full `ast.parse` instead of the regex | one file | 233 ms | as A1 | 15-20x slower for accepting forms nobody needs |
| B | decorator + a generated, committed index (the present tables written by a script, stale check in a test) | one file **and** run a script | none | index file | the extra step is the inconvenience the owner wants gone; 16 ms does not pay for it |
| C | decorator validates, the table stays the lookup | two files | none | table | ruled out by the owner's decision |
| D | entry points / plugin folders | n/a | n/a | | not now (section 6, Q8) |

Why A1 and not the old registry: the old one learned `name -> object` by *executing* every module. The names
are static text; executing 54 modules to read 105 names was the whole cost.

## 3. Shape
`neosr/utils/registry.py`, standard library only. `scripts/studio_surface.py` loads it **by file path**
(importing `neosr.utils` would pull in torch), so the Studio surface stays torch-free.

```python
class Row(NamedTuple):
    module: str              # file stem under the package
    scale: str | None        # ctor kwarg that takes the run scale
    meta: dict[str, Any]     # the other decorator keywords

class Registry(Mapping[str, Row]):
    def __init__(self, marker: str, package: str, *, required=(), optional=(), path=None): ...
    def register(self, **meta) -> Callable[[T], T]:   # identity: nothing happens at import
        return lambda obj: obj
    def load(self, name: str) -> Any:                 # imports the row's module, nothing else
        return getattr(import_module(f"{self.package}.{self[name].module}"), name)
    # Mapping protocol over the cached scan: registry[name], name in registry, iteration, len

ARCH_REGISTRY    = Registry("ARCH_REGISTRY", "neosr.archs", required=["scale"])
LOSS_REGISTRY    = Registry("LOSS_REGISTRY", "neosr.losses")
METRIC_REGISTRY  = Registry("METRIC_REGISTRY", "neosr.metrics", optional=["choices"])
DATASET_REGISTRY = Registry("DATASET_REGISTRY", "neosr.data")
MODEL_REGISTRY   = Registry("MODEL_REGISTRY", "neosr.models")
```

`neosr/utils/factory.py` keeps `resolve`, `inject_scale`, `construct`, `build` and their behaviour (unknown type
-> `ConfigError` with a did-you-mean, scale injection, kwarg errors); they take the registry instead of
`(table, package)`. The builders (`build_network`, `build_loss`, `build_metric`, `build_dataset`, `build_model`)
keep their signatures and call shapes. `OPTIMIZERS` and `SCHEDULERS` stay tables: they hold torch's own classes,
which cannot carry a decorator (assumption, to confirm).

## 4. The scan, exactly
For every `*.py` directly in the package folder except `__init__.py`:

1. read it as text (`utf-8-sig`, universal newlines: CRLF and a BOM are the reader's business) and find
   `^@MARKER\.register\((.*?)\)[ \t]*(#[^\n]*)?\n(class|def) (\w+)` (multi-line arguments and a trailing comment
   are fine; `async def` is not a type);
2. the arguments must be keyword arguments with literal values (`ast` on that snippet alone): positional
   arguments, `**`, a repeated keyword, a key outside `required + optional`, a missing required key or a `scale`
   that is not a string or `None` are errors;
3. **self-check:** in a module of the package the name `MARKER` may appear only in its import and in these
   decorators: `count(MARKER) == decorators + 1`. Every other spelling (`MARKER.register (..)`, an alias
   `REG = MARKER`, a call `MARKER.register(obj)`, an indented or commented-out decorator, another decorator or a
   blank line between it and the definition) breaks the count and fails with the file name, before anything is
   imported;
4. a name registered twice (one file or two) is an error naming both files.

Accepted form, the only one: the registry decorator at column 0, directly above a module-level `class` or `def`
(other decorators go above it). The regex is a restricted convention, not a parser: its independent check is a
test (section 7) that reads the same files with `ast` and must get the same rows, which catches what a text scan
cannot know (a decorator-shaped example inside a string, a registered name rebound later in its module).
Scan errors are bugs in the source tree: `RuntimeError` with the file name, not `ConfigError`. The source must be
readable as files (a checkout or an editable/unpacked install; zip and pyc-only installs are not supported:
assumption, matches how the project is run).

## 5. What the decorator carries
| registry | keywords | replaces |
|---|---|---|
| `ARCH_REGISTRY` | `scale=` **required**: the kwarg name, or `None` | column 2 of `ARCHS`. A forgotten scale cannot happen; a *wrong* `scale=None` is caught by a test that needs no list: a scale-free network (and the classes a scale-free factory calls) must have no parameter named like any registered scale kwarg. One reviewed exception today: `resgan`'s bool `upscale`. This replaces the hand-kept `NON_SR_ARCHS` |
| `LOSS_`, `DATASET_`, `MODEL_REGISTRY` | none | `LOSSES`, `DATASETS`, `MODELS` |
| `METRIC_REGISTRY` | `choices=` optional, on the numpy reference `calculate_<x>`; the `_pt` twin rule is unchanged | `METRICS` and `CHOICES`: registration, validation and the Studio surface read one place |
| export gaps (follow-up S.11) | **not in the decorator** | `NO_ONNX` / `STATIC_ONLY` stay in `convert.py`: they record what one exporter at one opset cannot do (the table's own comment: re-swept, opset 17, TorchScript exporter), not a property of the arch; a bare `no_onnx` on the class would go stale with the next exporter. The mechanism allows the move later |

The scale is declared, not inferred: inference needs the import (the static Studio surface has none) and fails
on factories that forward `**kwargs`. `tests/test_registry.py` keeps checking that the declared kwarg really
reaches the constructor.

## 6. Answers to the questions of `next_tasks.md` 2.6
| # | answer |
|---|---|
| 1 | `register(scale="upscale")` / `register(scale=None)`, required for archs. ONNX gaps stay in `convert.py` (section 5); 1x-only (`nafnet`) stays a constructor check |
| 2 | classes and functions alike: the pattern accepts `class` and `def` |
| 3 | table in section 5; datasets keep the one positional dict, models `(opt, runtime)`; `OPTIMIZERS` / `SCHEDULERS` stay tables |
| 4 | section 4 |
| 5 | duplicates fail at scan time; the type name is the Python name; **no aliases** (type names, `state_dict` keys and export metadata must not change, and an alias is a second name for nothing); did-you-mean runs over the scanned names |
| 6 | `registry.py` imports the standard library only; the `_REGISTRY` ban in `tests/test_hygiene.py` becomes: importing by name (`importlib`, `pkgutil`, `__import__`) is allowed under `neosr/` only in `utils/registry.py`, and a test holds `registry.py` to the standard library |
| 7 | every process that resolves a type scans once (16 ms for archs, 1-3 ms for the others). Loader workers resolve nothing. Passing the map down would add state for no measurable gain |
| 8 | user folders: **later**, not now. It needs more than a second folder in the scan (`load` imports under the registry's package), so it is its own small design |
| 9 | a script writes the decorators from the present tables (105 + 16 + 5 + 3 + 2 = 131 rows, line endings kept, `*_arch.py` is outside ruff), then the tables are deleted. Order: registry + scan + tests, archs, losses, metrics, data, models, docs |
| 10 | the subprocess probe of `tests/test_imports.py` (a package import loads no row module **and runs no scan**), a test that one lookup imports the requested module and not its siblings, `register()` returns its argument untouched |

## 7. Proof and tests
- `studio/shared/catalog/neosr-surface.json` must come out **identical** after the migration: regenerate it,
  `git diff` shows nothing. The script reads the registries through the same scan.
- `tests/test_registry.py`: every row resolves and its scale kwarg reaches the constructor (unchanged checks,
  rows from the registries); **the `ast` oracle** (the scan equals an independent `ast` reading of the five
  packages; registered names are module-level and bound once); the scale-free check of section 5; scan unit
  tests on a temporary package: the accepted forms, every rejected form of section 4, duplicates, bad keywords,
  one lookup imports one module and not its sibling.
- `tests/test_imports.py` (`TABLES` renamed, plus "no scan at import"), `tests/test_hygiene.py` (rule of Q6),
  `tests/test_config.py`, `tests/test_studio_surface.py` on Windows; the per-arch tests and the full suite on
  the Linux machine.
- Callers: `neosr/models/checkpoint.py` and the tests that read the tables (`tests/test_archs*.py`,
  `tests/test_s2_cnn.py`, `tests/test_losses_pipeline.py`, `tests/test_s3_losses.py`, `tests/test_losses.py`) get
  the new name and, where they unpack `(stem, kw)`, the row's attributes. The Linux-only files are checked here by
  `pytest --collect-only` (their table code runs at collection).
- **Dry run on a copy of the tree (2026-11-10):** the scripted migration wrote 131 decorators, the scan equals
  all five tables, the draft `tests/test_registry.py` passes (164 cases, 10 s).

## 8. Codex review of draft 1 (gpt-6.1-sol, xhigh, read-only, 2026-11-10)
| finding | disposition |
|---|---|
| the first self-check (`count("MARKER.register(")`) missed `MARKER.register (` and an alias | fixed: the count is over the bare name (section 4.3) |
| a text scan cannot see a decorator-shaped string or a rebinding; A2 would catch the rebinding at run time | accepted as A1's limit, covered by the `ast` oracle test; stated in section 2 so the owner can weigh A2 |
| importing `neosr.utils.registry` loads torch through `neosr/utils/__init__.py` | the script loads the file by path (section 3) |
| dropping `NON_SR_ARCHS` leaves a wrong `scale=None` unchecked | replaced by the list-free check of section 5 |
| grammar not exact; BOM; repeated keywords; value types | section 4 rewritten from the implemented pattern; `ast.parse` accepts a repeated keyword, so the scan rejects it itself |
| a three-field `Row` is not "a rename" for tests that unpack two | wording fixed (section 7); `Row[1]` stays the scale, so index access keeps its meaning |
| "one lookup imports exactly one module" is false (`spanplus_arch` imports `span_arch`); the probe cannot show that the scan is deferred | wording fixed (Q10); the probe asserts that no scan ran at package import |
| metric `CHOICES` belong next to the metric; ONNX gaps do not belong on the arch | both adopted (section 5); draft 1 proposed moving the ONNX tables |

## 9. The owner's choices (2026-11-10)
1. Mechanism: **A1**.
2. Decorator arguments: `scale=` required on archs, `choices=` on metrics.
3. ONNX gaps (S.11): stay in `convert.py`.
4. Not asked separately, implemented as proposed and open to revision: no aliases; user plugin folders later;
   `OPTIMIZERS` / `SCHEDULERS` stay tables.

# Critical audit of neosr training, architectures, and deployment

**Audit date:** 2026-10-22  
**Checkout:** `C:\work\neosr-dev`  
**Commit:** `1767f8d784dd82e3d62dbb10f30f9a1f5bcdf598`  
**Scope:** the supplied neosr repository, including training/inference entry points, architectures, losses, optimizers, data preparation, degradation, metrics, checkpointing, export, configuration, tests, and installation. This is not a separate audit of an external traiNNer checkout.  
**Changes made:** this report only. Existing changes to `AGENTS.md` and `.codex` were preserved.

## 1. Assessment

The refactored trainer has a substantially better foundation than the older issues recorded in this repository might suggest. Its atomic checkpoints, explicit optimizer phases, detached logging, gradient accumulation boundaries, adversarial gradient ownership, and bounded data/metric caches are valuable improvements. A small CPU training run with EMA, accumulation, validation, saving, and resume worked in this audit.

However, **the project should not yet be treated as uniformly correct across its advertised architectures and options**. The most serious remaining problems can silently change what is trained: the default schedule-free Adan recurrence is incorrect, ASID's default attention disconnects gradients, and MFHGMoE's router cannot learn. Optional attention and checkpoint switches are not consistently equivalent to their reference paths. Export can pass its own verification while producing incorrect pixels at another input size.

The main priority is to establish mathematical and gradient correctness for supported configurations before pursuing further kernel-level speedups. Some of the largest performance opportunities are also correctness problems: disconnected branches still consume compute, nominally sparse experts execute densely, and reduced-correlation attention constructs enormous dense position-bias intermediates.

No general claim of an unbounded training memory leak is supported by this audit. Several **large but bounded allocations**, disk-growth paths, and exception-time resource weaknesses are identified below. No new CUDA throughput, long-run convergence, or SOTA image-quality result is claimed.

### Priority overview

| ID | Priority | Finding | Evidence level | First action |
|---|---|---|---|---|
| F01 | P1 | Schedule-free Adan's first evaluation step moves uphill; first-step weight decay is lost | CPU reproduction + independent algebra review | Define and test the intended recurrence; version the behavior change |
| F02 | P1 | Default ASID attention cuts QKV gradients and ignores learned temperature | Independently reproduced by two reviewers | Restore autograd and attention semantics |
| F03 | P1 | MFHGMoE's routing and routing-feature branch receive no gradients | Independently reproduced; reduced full-model probe | Use differentiable routing weights; test expected parameter gradients |
| F04 | P1 when enabled | SwinIR/SwinIR-dense flash path omits positional bias and shift masks | CPU output/gradient comparison | Implement equivalent masked SDPA |
| F05 | P2 | SRFormer checkpointing raises an argument error | CPU reproduction | Pass spatial size and use non-reentrant checkpointing |
| F06 | P2 | HMA/MFHGMoE accept checkpointing flags without implementing it | Source verification | Implement or reject the option |
| F07 | P1 for dynamic deployment | DySample ONNX export freezes normalization dimensions and produces wrong pixels | ONNX Runtime comparison at multiple sizes | Preserve dynamic shape arithmetic |
| F08 | P2 | Transformer dynamic export can freeze padding/mask branches | Reduced SwinIR ONNX reproduction | Validate supported shapes and export constraints |
| F09 | P1 for affected HiT variants | Position-bias construction creates multi-GiB quadratic intermediates | Verified constructor geometry and tensor-size calculation | Chunk or algebraically preaggregate bias construction |
| F10 | P2 | MFHGMoE executes every expert and stacks all expert outputs | Source + operation/storage calculation | Correct routing, then redesign dispatch |
| F11 | P2 | DITN computes and discards learned-temperature attention | CPU gradient reproduction | Remove dead work and preserve temperature in SDPA |
| F12 | P2 | RCAN/KRGN MeanShift parameters are accidentally trainable | CPU parameter inspection | Freeze parameters explicitly, with compatibility review |
| F13 | P2 | Accepted dataset normalization conflicts with clamp, output, and metric domains | CPU reproduction | Define one consistent image-domain contract |
| F14 | P2 | Grayscale OTF data crashes in default JPEG degradation | Full Degrader CPU reproduction | Reject the combination or implement a consistent grayscale path |
| F15 | P2 | Hidden files break recursive dataset discovery | Filesystem reproduction | Recurse only into directories |
| F16 | P2 for distributed data | Unsorted paired paths can duplicate/omit samples across ranks | Two-order sampler reproduction | Sort stable relative paths |
| F17 | P2 | LMDB key extraction can silently merge distinct images | Two input images became one database entry | Use final-suffix removal and reject duplicate keys |
| F18 | P2 | LMDB creation retains the whole encoded dataset and hides worker failures | Source + corrupt-image worker reproduction | Stream bounded results and propagate errors |
| F19 | P2 | Resume silently overrides edited optimizer settings and misses objective changes | CPU optimizer-state reproduction | Compare effective saved/current training configuration |
| F20 | P2 | Invalid EMA decay is accepted and can extrapolate away from the model | Config + numerical reproduction | Validate finite ranges and cross-key invariants |
| F21 | P2 | Validation image names collide across directories/datasets | Source verification | Include dataset and relative image identity |
| F22 | P2 resource risk | TOPIQ materializes attention weights that it immediately discards | CPU profiler + size calculation | Set `need_weights=False`, verify score parity |
| F23 | P2 operational | Rank-zero validation failure can strand other distributed ranks | Control-flow verification; previously documented | Propagate validation status to all ranks |
| F24 | P2 tooling | Windows installation aliases retain an undefined installation variable | PowerShell source verification | Persist a literal, correctly quoted installation path |
| F25 | P2 assurance gap | Test portability, missing CI, and permissive gradient checks leave important paths unverified | Actual test results + test inspection | Establish portable CPU and CUDA capability suites |

**Severity:** P1 means materially wrong learning/deployment behavior or a major resource failure on an affected supported path; it does not mean every configuration fails. P2 means a significant conditional correctness, operational, performance, or assurance problem. Lower-priority limitations and research/tuning opportunities are separated later. No P0 incident was established.

## 2. Method, coverage, and limitations

### Repository inventory

| Item | Observed |
|---|---:|
| Python files under `neosr/` | 111 |
| Lines in those files | 38,147 |
| `tests/test*.py` files | 29 |
| Architecture lookup entries | 100 |
| Architecture source modules referenced by the lookup | 52 |
| Scale-taking architecture entries, including variants | 93 |
| Other entries: discriminator/perceptual models | 7 |
| Training TOMLs | 90 |
| Other/inference TOMLs | 43 |
| Optimizer sections selecting `adan_sf` | 168 |
| Optimizer sections selecting `adamw` | 6 |

Counts describe the current tree; aliases and variants are not 100 independent architectures. The README describes 45 generator families. Architecture families were inventoried and reviewed selectively in depth; not every variant was trained or exported.

Three parallel reviews used **GPT-6** for the trainer/optimizers and **GPT-6.1** for data/losses/metrics and architectures/export. The primary review checked configuration, installation, LMDB preparation, test failures, numerical/resource claims, and source references, then reconciled findings. Agreement between models was not treated as proof: the strongest findings were reproduced or derived directly from the implementation.

### Execution environment

The initially available Python was 3.10.9 without PyTorch or pytest. An isolated environment was created under the system temporary directory with Python 3.13.17, **PyTorch 2.9.1+cpu**, and torchvision 0.24.1+cpu. These match the locked PyTorch/torchvision release family, but use CPU wheels. Ancillary packages were installed separately, so this was **not a complete `uv.lock` reproduction**. No source changes were needed for the probes.

There was no usable CUDA test environment. Consequently, CUDA/NCCL behavior, compiler throughput, GPU peak memory, tensor-core dispatch, hardware BF16 support, and long-run image-quality results remain unverified here. Allocation estimates explicitly state their dtype and tensor shape. CPU autocast results are not evidence of CUDA AMP stability.

### Checks actually executed

| Check | Result | Interpretation |
|---|---|---|
| Core configuration, lookup, hygiene, checkpoint, phase/state, model, optimizer, EMA, stop, entry-point, runtime tests | **447 passed, 26 skipped, 20 failed** | Sixteen failures originated in Windows-invalid TOML generated by entry-point fixtures; four failed at Gloo device initialization |
| Data, OTF degradation, tiling, metrics; pretrained metric tests excluded | **115 passed, 6 skipped, 2 deselected** | No pretrained metric weights downloaded |
| Lightweight loss tests, excluding pretrained/backbone and historical comparisons | **96 passed, 51 deselected** | Exercises current lightweight loss behavior |
| Loss pipeline, excluding real heavyweight constructors | **121 passed, 1 deselected** | Builder, weighting, target, and pipeline contracts |
| Architecture A1 + A2 tests | **37 passed, 41 skipped** | Many CUDA-specific paths skipped |
| Aggregate of the scoped pytest invocations above | **816 passed, 73 skipped, 20 failed, 54 deselected** | This is not a full-suite green result |
| Eleven lightweight losses, random 64×64 input, FP32 / CPU FP16 autocast / CPU BF16 autocast | **33/33 finite forward/backward probes** | Small numerical smoke test only |
| Separate correctly quoted Windows TOML training probe | Train to update 4: **exit 0**; resume to update 6: **exit 0** | Compact G, AdamW, L1, EMA, accumulation 2, validation and checkpoint/export paths exercised |
| Python syntax parsing | All 111 package files parsed under Python 3.13 | Does not establish runtime correctness |
| Ruff isolated `F,E9` scan including architecture files | **10 diagnostics**, including 2 undefined-name uses in an unused OmniSR helper | Architecture files are excluded from the normal configured Ruff scan |
| Focused bug probes | Confirmed findings described below | Includes ONNX Runtime, gradient ownership, filesystem, normalization, EMA, and LMDB tests |

The custom training probe produced `2.state`, `4.state`, `6.state` and validation images. Its first harness encountered a Windows temporary-directory cleanup error because the process logger still held its log file open after both training calls succeeded. A repeat with explicit logging shutdown again completed both runs with exit 0 and successfully cleaned up its temporary directory. This is a harness/lifecycle observation, not a failed optimization step or evidence of a growing leak.

The four Gloo failures reported `makeDeviceForHostname(): unsupported gloo device` in this environment. They do not prove the trainer's distributed implementation is incorrect. The sixteen entry-point fixture failures are actionable test-portability defects: native Windows paths were inserted into TOML double-quoted strings, so sequences such as `\U` were parsed as invalid escapes before the intended test behavior ran.

Existing [acceptance results](docs/refactor/analysis/acceptance.md) and [performance results](docs/refactor/analysis/perf.md) were read as historical evidence. They concern earlier commits, specific RTX 4090 workloads, and sometimes contemporaneous uncommitted changes. Their timings and conclusions are not new measurements of this checkout.

### Independent review and false-positive rejection

| Candidate | Reconciliation |
|---|---|
| AdanSF recurrence | GPT-6 reproduced first-step ascent; GPT-6.1 independently derived the same result. The repository already acknowledges the recurrence as deferred work |
| ASID disconnected attention | Both reviewers independently reproduced missing QKV/input/temperature gradients |
| MFHGMoE disconnected router | Both reviewers reproduced missing router and routing-input gradients; full reduced-model inspection corroborated affected feature branches |
| Output naming collisions | Independently found in the primary and data reviews |
| Multiple `train_*` datasets silently replacing the loader | **Rejected**: `load_config` already rejects suffixed training phases, and an existing test covers it |
| All Turing GPUs necessarily fail BF16 | **Rejected as too broad**: PyTorch's capability API can report emulated support. The narrower missing-capability-check issue remains |
| Every cache or high reserved-memory figure is a leak | **Rejected**: bounded pools, allocator reservation, and temporary tensors are distinguished below |
| OmniSR default network is broken by an undefined name | **Rejected**: the undefined name is in an unused helper, not the registered default forward path |
| Historical GAN detachment/accumulation/checkpoint issues remain present | **Not repeated without current evidence**: current code and tests already address many of them |

## 3. Learning correctness and architecture defects

### F01 — Schedule-free Adan does not implement a sound first-step recurrence

**P1; reproduced.** Evidence: [adan_sf.py:58](neosr/optimizers/adan_sf.py#L58), [evaluation transform:165](neosr/optimizers/adan_sf.py#L165). The issue is already recorded as deferred in [PLAN.md:79](docs/refactor/PLAN.md#L79), which should be retained as context rather than presenting this as a newly discovered historical defect.

For the first positive-learning-rate step, the cumulative averaging coefficient `ckp1` is 1. The code initializes `z` from the current parameter, applies weight decay to the live parameter, then interpolates the live parameter back to `z`. Both adaptive update coefficients contain `(1 - ckp1)`, so they vanish. Meanwhile `z` is updated with the raw gradient. The evaluation transform then extrapolates using `1 - 1 / beta1`, a negative coefficient for the normal beta range.

CPU probe: scalar parameter 1, gradient +1, learning rate 0.01, weight decay 0:

| Optimizer | Parameter after training step | Parameter used for evaluation |
|---|---:|---:|
| `adan_sf`, schedule-free enabled | 1.0000000000 | **1.0002040863** |
| `adan_sf`, schedule-free disabled | approximately 0.99 | approximately 0.99 |
| Plain `adan` | approximately 0.99 | approximately 0.99 |
| `adamw_sf` | approximately 0.99 | approximately 0.99 |

For a positive gradient of a scalar quadratic at parameter 1, the evaluation change is uphill. Independently, with zero gradient, learning rate 0.1, and weight decay 0.1, the first schedule-free step leaves the parameter at 1: the interpolation overwrites the decay.

This does **not** prove every existing SR run diverges; historical runs plainly made progress. It does establish that the update is not a harmless evaluation lag. The branch also multiplies its first-moment term by the bias-correction factor where ordinary Adan divides, and updates `z` with a different direction from the adaptive live-parameter update.

**Repair:** specify the intended Adan + schedule-free recurrence, compare several complete trajectories against that definition, and test zero-gradient decay, warmup, evaluation, and resume. Do not merely reverse the evaluation sign: the x/y/z update relationship is the problem. Existing checkpoints and 168 optimizer template sections make an explicit legacy/version policy appropriate. Use controlled SR quality A/B runs after mathematical tests pass. The schedule-free authors' [reference recurrence](https://github.com/facebookresearch/schedule_free) and the [official Adan implementation](https://github.com/sail-sg/Adan/blob/main/adan.py) are the appropriate baselines.

### F02 — ASID's default attention prevents learning in its QKV branch

**P1; independently reproduced.** Both [Channel_Attention](neosr/archs/asid_arch.py#L352) and [Channel_Attention_grid](neosr/archs/asid_arch.py#L412) execute SDPA inside `torch.no_grad()`. `flash_attn=True` is the architecture default.

The probe found `grad is None` for QKV weights, the depthwise QKV convolution, and temperature, while the output projection received gradients. Switching to the manual path restored gradients. Residual paths can still carry gradients through the larger network, so the network may appear to train while the attention transform does not learn as intended.

There is a second semantic mismatch: the manual path uses a learned temperature, while SDPA uses its default scale and never consumes that temperature. Removing `no_grad()` alone is insufficient.

**Repair:** retain autograd through SDPA and apply temperature differentiably to its query/logits with the intended scale. Compare forward values and gradients of all expected trainable parameters against the manual path. DDP with `find_unused_parameters=False` is additionally exposed to unused-parameter failures; this consequence was not tested on CUDA. See the [PyTorch 2.9 SDPA definition](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html) for scaling and mask semantics.

### F03 — MFHGMoE's hard routing disconnects its router and feature aggregation

**P1; independently reproduced.** [DualRouting](neosr/archs/mfghmoe_arch.py#L843) retains selected integer indices and constructs constant one-hot tensors at [lines 870–876](neosr/archs/mfghmoe_arch.py#L870). Selected probability values do not weight the expert outputs.

For a small `HMoE`, all router weights/biases and `group_embedding` had no gradients, as did the routing input `x_t`; the expert input still had gradients. A reduced complete MFHGMoE had 13 missing parameter gradients, including routing-feature convolution/aggregation parameters. Computing a softmax before throwing away its values does not make an argmax/top-k index differentiable.

**Repair:** define the intended routing algorithm, use selected differentiable gate weights, and return/use any intended auxiliary load-balancing objective. If genuinely discrete forward routing is required, use an explicit, justified gradient estimator rather than accidental detachment. Add expected-gradient checks that fail on missing gradients; exempt only intentionally frozen or conditionally unused parameters. The generic architecture smoke test currently filters out `None` gradients, which hides exactly this class of defect.

### F04 — SwinIR and SwinIR-dense flash switches change the model

**P1 when enabled; reproduced.** The SDPA branches in [swinir_arch.py:172](neosr/archs/swinir_arch.py#L172) and [swinirdense_arch.py:142](neosr/archs/swinirdense_arch.py#L142) omit the learned relative-position bias and shifted-window mask used by the manual paths.

A controlled window-attention comparison with identical weights/input and a mask produced a maximum absolute output difference of **0.06561327** in both implementations. The positional-bias table received no gradient in the flash path. This is more than backend rounding: the attention equation differs.

**Repair:** construct a correctly broadcast additive attention bias combining relative position and window mask, preserve scaling and training/evaluation dropout, and pass it to SDPA. Then measure which CUDA kernel is actually dispatched; an equivalent bias may change available fused kernels. Existing QKV-gradient tests are useful but do not establish equivalence. CRAFT's optional flash branch also omits its learned dynamic positional bias at [craft_arch.py:185](neosr/archs/craft_arch.py#L185); that branch needs its own output/gradient parity test before being advertised as an equivalent optimization.

### F05 — SRFormer activation checkpointing crashes

**P2; reproduced.** [srformer_arch.py:644](neosr/archs/srformer_arch.py#L644) calls `checkpoint(blk, x)`, but [PSA_Block.forward](neosr/archs/srformer_arch.py#L437) requires `(x, x_size)`. A reduced BasicLayer with `use_checkpoint=True` raises `TypeError: PSA_Block.forward() missing 1 required positional argument: 'x_size'`.

**Repair:** pass the spatial size and explicitly use the supported checkpoint implementation, for example `checkpoint(blk, x, x_size, use_reentrant=False)`. Confirm gradient parity through the actual trainer, which calls backward with explicit parameter inputs. PyTorch describes the relevant distinctions in its [checkpoint documentation](https://docs.pytorch.org/docs/2.9/checkpoint.html). A module-only forward success is insufficient.

### F06 — HMA and MFHGMoE silently ignore checkpointing options

**P2; source verified.** HMA stores `use_checkpoint` at [line 826](neosr/archs/hma_arch.py#L826), but its [block loop](neosr/archs/hma_arch.py#L895) never uses it. MFHGMoE does the same at [lines 534](neosr/archs/mfghmoe_arch.py#L534) and [581](neosr/archs/mfghmoe_arch.py#L581).

Users can select a memory-saving option and receive no memory saving or warning. Implement checkpointing with all required arguments and gradient/state parity checks, or reject/remove the unsupported option. This is particularly relevant to memory-heavy models, where silently ignoring it can lead to an otherwise avoidable OOM.

### F11 — DITN computes learned-temperature attention and discards it

**P2; reproduced.** [ditn_arch.py:118–123](neosr/archs/ditn_arch.py#L118) forms attention using the trainable temperature and softmax, then returns the result of a separate SDPA call that does not use those values. `ISA.temperature.grad` is `None` after backward.

This combines a disconnected parameter with unnecessary attention work in eager execution. Remove the discarded calculation and incorporate the intended temperature in the attention that actually contributes to the output. Verify both gradient participation and equivalence before calling the replacement an optimization.

### F12 — RCAN/KRGN MeanShift layers are not frozen

**P2; parameter flags reproduced.** [RCAN:21](neosr/archs/rcan_arch.py#L21) and [KRGN:16](neosr/archs/krgn_arch.py#L16) assign `self.requires_grad = False` to the module. That does not set `requires_grad=False` on its weight and bias; both parameters remain trainable.

The code indicates an intent to freeze these color transforms. Use `self.requires_grad_(False)` or freeze their parameters explicitly. Review old checkpoints before changing behavior: they may contain learned MeanShift values. Loading their values should remain possible even if future optimization stops changing them. This is not an invitation to replace all `.data` initialization mechanically; the concrete error is the ineffective freeze.

## 4. Deployment, shape support, and resource scaling

### F07 — Dynamic DySample ONNX export can silently produce incorrect images

**P1 for dynamic deployment; reproduced with ONNX Runtime.** [arch_util.py:114](neosr/archs/arch_util.py#L114) constructs the coordinate normalizer with `torch.tensor([W, H], ...)`. Under the legacy `dynamo=False` exporter used by [convert.py:288](convert.py#L288), these shape-derived values become constants from the trace input.

In the deterministic probe, DySample was exported at 8×8 and evaluated at another spatial size:

| Input | Maximum absolute PyTorch/ONNX error | Meaning |
|---|---:|---|
| 8×8, matching export size | `1.19e-7` | The ordinary one-shape verification passes |
| 12×10 | **0.5935176** | Output shape is correct but sampling coordinates and pixels are wrong |

This affects the shared primitive used by architectures including realplksr, spanplus, esc, moesr, and mosrv2 when they select DySample. It is not evidence that their eager training path is wrong.

**Repair:** retain shape values as graph-preserving arithmetic; do not wrap traced dimensions in a new constant tensor. Require multi-shape ONNX comparisons, including rectangular inputs and batch changes. A successful same-shape comparison cannot validate dynamic axes.

### F08 — Dynamic transformer exports are not adequately validated

**P2; reproduced for reduced SwinIR.** Window reversal and padding/mask paths contain Python shape conversions and branches, for example [swinir_arch.py:70](neosr/archs/swinir_arch.py#L70), [367](neosr/archs/swinir_arch.py#L367), and [1044](neosr/archs/swinir_arch.py#L1044). A reduced SwinIR exported at 8×8 ran at the original size but failed ONNX Runtime reshape operations at 12×8 and 1×1.

The converter labels float exports dynamic at [convert.py:262](convert.py#L262) but verifies a single dummy shape at [line 210](convert.py#L210). Other export sample sizes may trace different branches, so this result should not be generalized to every variant and trace size. It does demonstrate that current verification is insufficient for the advertised dynamic contract.

**Repair:** define per-architecture supported dimensions, use export-safe shape/mask construction, and test several sizes inside that contract. Otherwise export an explicitly static/constrained model. Moving to the newer exporter is an option to investigate, not an automatic fix for Python-specialized model logic.

### F09 — HiT-SRF position-bias construction defeats its memory scaling

**P1 for affected medium/large variants; allocation sizes verified analytically.** [hitsrf_arch.py:303–326](neosr/archs/hitsrf_arch.py#L303) first constructs dense pairwise relative coordinates and gathers a dense positional-bias tensor, then averages it into the smaller correlation representation.

Actual constructor settings produce the following intermediate sizes:

| Variant's largest window | Tokens `N` | `[N,N,2]` int64 coordinates | `[N,N,6]` FP32 gathered bias |
|---|---:|---:|---:|
| Medium: 128×128 | 16,384 | **4 GiB** | **6 GiB** |
| Large: 192×192 | 36,864 | **20.25 GiB** | **30.375 GiB** |

These are individual tensor-storage calculations, not measured peaks or a claim that all tensors coexist at their maximum. FP16/BF16 bias reduces the floating tensor size, but not the int64 coordinate cost. Index tensors, contiguous copies, gradients, and the rest of the model add further demands. Small input images do not eliminate large-window work when the architecture pads to those windows. Existing tests acknowledge heavy HiT variants as impractical to run in their normal forward/backward coverage.

**Repair:** process query chunks and reduce the positional bias without keeping the full pairwise tensor, or derive an equivalent aggregation over relative-offset frequencies. Verify positional-network gradients and outputs on small reference cases. Caching the full pairwise index would trade repeated allocation for enormous persistent memory, not solve the scaling problem.

### F10 — MFHGMoE pays dense expert compute and memory costs

**P2; source and arithmetic verified.** [mfghmoe_arch.py:955–980](neosr/archs/mfghmoe_arch.py#L955) executes every expert on the complete image, builds a Python list of all outputs, stacks them, and multiplies by the routing tensor afterward.

For the default 16 experts, 64 input channels, x4 output projection to 1,024 channels, batch 1:

| LQ shape | Expert stack alone, FP32 | Expert convolution MACs alone |
|---|---:|---:|
| 64×64 | **256 MiB** | approximately **21.47 billion** |
| 256×256 | **4 GiB** | approximately **343.60 billion** |

The MAC estimate uses eight 1×1 and eight 3×3 experts. It excludes the rest of the architecture. The list of outputs and subsequent weighted product can require additional storage. This is a dense mixture implementation despite sparse-looking selection.

Python conditions on CUDA tensors, including `group_mask.any()` and the NaN checks at [963](neosr/archs/mfghmoe_arch.py#L963) and [971](neosr/archs/mfghmoe_arch.py#L971), also force host decisions. Replacing NaNs with zeros inside the architecture can obscure the first invalid operation and should not substitute for a defined finite-value policy.

**Repair order:** fix F03 first, then define a dispatch algorithm. Selected-token/grouped GEMM can suit 1×1 experts. A 3×3 expert requires neighborhoods/halos; gathering isolated pixels would change the operation. A simpler transitional implementation can accumulate expert contributions without one giant stack, but still performs dense compute and needs a training-memory measurement.

### Shape and conversion capability gaps

| Area | Current evidence | Constructive action |
|---|---|---|
| Tiny input padding | LMLT reflect-to-window behavior fails small inputs such as 7×5/1×1; SRFormer reflect padding fails 1×1; ATD-light's one-time duplication is insufficient for some inputs smaller than its target window | Declare minimum dimensions and implement tested pad/crop behavior where arbitrary sizes are intended |
| Window alignment | HMA/MFHGMoE do not generally pad arbitrary input sizes; the converter's default 20×20 sample is not a valid universal input | Architecture capability metadata should drive validation and export samples |
| ASID minimum dimensions | Its ESA pooling imposes additional constraints; a small attention-unit test is not a whole-network small-input test | Check complete generator constraints |
| Aether full-half QAT preparation | `aether(...).half().prepare_qat()` fails because [line 619](neosr/archs/aether_arch.py#L619) primes with FP32 random input against half weights | Match priming dtype or explicitly reject unsupported full-half QAT; the normal FP32-before-autocast trainer path is not affected |
| Manual PLK deploy helpers | [plksr_arch.py:90](neosr/archs/plksr_arch.py#L90), [184](neosr/archs/plksr_arch.py#L184), and [255](neosr/archs/plksr_arch.py#L255) create identity/new-convolution tensors with device/dtype assumptions | Test conversion on CUDA and half tensors; allocate from the source weight's device/dtype |
| MetaGAN attention helper dropout | [metagan_arch.py:55](neosr/archs/metagan_arch.py#L55) passes nonzero configured SDPA dropout even in evaluation; the normal GatedCNNBlock constructs this attention with zero dropout | Make dropout conditional on training mode; this is a nondefault helper defect, not evidence of default MetaGAN evaluation randomness |
| PlainUSR deployment | The converter disables reparameterization; its source comment identifies PlainUSR's block deployment as broken | Repair and verify it before enabling that conversion path |
| OmniSR helper | `AdaConvGuidedFilter` uses undefined `channel` at [449–450](neosr/archs/omnisr_arch.py#L449); instantiation raises `NameError` | Remove dead code or repair its actual intended dimensions; default OmniSR does not call it |

A compact capability record per family—scales, input channels, minimum/multiple dimensions, checkpointing, AMP, compile, tiled inference, dynamic export, and deploy conversion—would prevent misleading universal support. It should capture verified behavior, not become a second complicated plugin system.

## 5. Data correctness and image-domain contracts

### F13 — Mean/std normalization is accepted but not supported end to end

**P2; reproduced.** [paired_dataset.py:123](neosr/data/paired_dataset.py#L123) normalizes both LQ and GT when mean/std are provided; the single dataset has the corresponding option. The trainer then clamps SR to `[1/255, 1]` at [image.py:224](neosr/models/image.py#L224), and validation converts output and GT to images without inverse normalization.

With a constant pixel value 64/255 and `mean=std=0.5`, GT becomes approximately **−0.498039**. The default clamp cannot produce that target; its minimum absolute error is approximately **0.501961**. Even if training clamp is disabled, a perfect prediction in the normalized domain is saved as pixel **0**, not 64. Metrics are then computed in the wrong/clipped domain.

**Repair:** either reject nonidentity dataset normalization for the current image-domain trainer, or implement normalization as a clearly defined model boundary and invert it for losses that require physical RGB, image output, and metrics. Validate paired mean/std presence, channel counts, finite values, and nonzero std. Do not rely on users to infer which losses expect normalized features versus RGB in `[0,1]`.

### F14 — Grayscale OTF data is incompatible with default JPEG

**P2; reproduced.** `color='y'` produces one-channel GT in [otf_dataset.py](neosr/data/otf_dataset.py#L230), while [Degrader](neosr/data/otf_degradation.py#L42) enables JPEG by default. [DiffJPEG:102](neosr/utils/diffjpeg.py#L102) assumes RGB channel structure.

A complete degradation call on a two-image, one-channel batch with impulse kernels failed with an invalid reshape. Existing grayscale noise tests do not exercise this JPEG integration.

**Repair:** perform RGB degradation before consistent luminance conversion, implement a genuine grayscale JPEG path, or reject this configuration early with a targeted `ConfigError`. Disabling JPEG alone should not be presented as full grayscale support without checking the selected network and loss channels.

### F15 — Hidden files are treated as directories during recursive scanning

**P2; reproduced.** In [misc.py:102–108](neosr/utils/misc.py#L102), a hidden file fails the visible-file branch and falls into `elif recursive`, which recurses into that file. A directory containing `.DS_Store` reproduced `NotADirectoryError` / Windows error 267. Paired folder discovery uses recursive scanning.

**Repair:** skip hidden entries according to an explicit policy and recurse only when `entry.is_dir()` is true. Avoid following directory symlinks unless cycle handling is intentional. Also make supported image-extension filtering consistent: the OTF folder path currently collects every nonhidden file, so an ordinary text file can fail later in image decoding.

### F16 — Filesystem ordering can invalidate distributed sample partitioning

**P2 for affected distributed datasets; reproduced with controlled enumeration.** [data_util.py:199–216](neosr/data/data_util.py#L199) retains unsorted GT discovery order. The sampler shards integer indices, implicitly assuming every rank has the same index-to-image mapping.

Using the same two images but valid enumeration orders `[a,b]` and `[b,a]` on two ranks, the probe made both ranks select `a`, omitting `b`. Same-machine filesystems often hide this issue; independently copied datasets on multiple nodes can expose it. Python explicitly leaves [directory enumeration order arbitrary](https://docs.python.org/3.13/library/os.html#os.scandir).

**Repair:** sort by a stable relative path before building dataset pairs. For distributed preflight, optionally compare a compact ordered-manifest fingerprint across ranks. A seed alone cannot correct different dataset index mappings.

### F17 — LMDB creation can silently overwrite distinct source images

**P2; reproduced through actual database creation.** [create_lmdb.py:22](create_lmdb.py#L22) uses `img_path.split('.png')[0]` to derive keys. This removes everything from the first occurrence of `.png`, not just the final extension.

`img.png.original.png` and `img.png.png` both become `img`. A serial creation probe with two different valid images produced **one LMDB entry**. `txn.put` overwrote the earlier value; metadata can still describe multiple source entries.

**Repair:** use `Path(path).with_suffix('')` / final-extension removal and validate key uniqueness before creating output. Support the same filename encoding in writer and reader. This is separate from paired-LMDB reading, which already uses final-extension removal and preserves dots correctly.

### F18 — LMDB conversion scales with the entire dataset in RAM

**P2 resource/robustness issue; source verified.** [lmdb_util.py:108–132](neosr/utils/lmdb_util.py#L108) defaults to an **80-process** pool, submits every image, and stores all encoded results in dictionaries before writing the database. The CLI always enables this mode. Memory grows with total compressed dataset size plus worker queues and process overhead, rather than a bounded write batch.

Async results are not retained or `.get()`-checked, and no error callback propagates worker exceptions. `read_img_worker` dereferences `img.ndim` without checking decode success; a corrupt file reproduced `AttributeError`. The parent can subsequently report an unrelated missing dictionary key or first-image failure. Database/file handles also lack comprehensive exception-safe context management, so an interrupted/erroring build can leave partial output that blocks the next attempt. The map size is estimated from one image times ten, which is unreliable for heterogeneous images.

**Repair:** use a bounded iterator of worker results, write incrementally, choose a reasonable configurable worker count, propagate the failing path and exception, and make partial-build cleanup/restart explicit. Validate nonempty input, unique keys, encoding success, and map capacity. This is a streaming problem; a larger default RAM budget is not a robust solution.

## 6. Trainer state, validation, and configuration

### F19 — Resume silently mixes saved and edited training settings

**P2; reproduced.** [Phase.load_state_dict](neosr/models/phase.py#L135) checks optimizer identity, schedule-free mode, trainable names, and scheduler class, then restores saved optimizer groups and scheduler state. [run_record](neosr/models/image.py#L398) does not record optimizer hyperparameters, loss configuration, accumulation, clamp, or most data/degradation settings.

A Phase created with AdamW LR `1e-4` resumed a saved Phase with LR `1e-3` and silently ended with **LR `1e-3`**. Restoring the saved LR is reasonable for exact continuation; failing to explain the override is the issue. Meanwhile changing losses or accumulation changes current behavior without the corresponding saved/current warning.

**Repair:** record and compare the effective optimization/objective/data contract. Report which values are restored, which are current, and which changes are incompatible. If intentional overrides are supported, make them explicit and test scheduler interaction. Otherwise direct users to a new fine-tuning run with pretrained weights. Include `ema_power` and QAT phase scheduling in the comparison; silent changes to averaging/quantization schedules are also relevant.

### F20 — Important numeric and cross-key configuration invariants are missing

**P2; reproduced for EMA.** The schema checks types but not the full range/domain. `train.ema=1.5` passes `load_config` and `validate_train`; [image.py:61](neosr/models/image.py#L61) enables it, and [Ema.update](neosr/models/phase.py#L208) uses it directly when warmup power is unset. After copying a parameter equal to 1, then changing the live parameter to 2, the EMA becomes **0.5**: it extrapolates away from the new model.

Without `ema_power`, `ema=1` freezes the EMA after its first copy and positive infinity creates invalid averaging; NaN can silently disable the `ema > 0` branch. With warmup enabled, its effective decay can remain below the invalid configured ceiling. The separate warmup power check does not replace decay validation.

Other accepted values reach less helpful downstream failures or surprising behavior: a worker count string other than `'auto'`, nonpositive enlarge ratio, negative validation frequency, missing validation/test dataset names, invalid metric `better` values, and partially specified mean/std. OTF validates some list lengths but not all probability/range semantics.

**Repair:** enforce finite EMA decay with documented bounds, enum values, positive counts/frequencies where applicable, paired normalization fields, and ordered finite degradation ranges/probabilities. Preserve clearly documented sentinel values such as `keep_states <= 0` meaning keep all. Do not label ordinary abstract-method `NotImplementedError` declarations as missing features.

### F21 — Validation artifacts can overwrite one another

**P2; source verified.** [image.py:339](neosr/models/image.py#L339) reduces a source path to its stem. Training output paths in [_img_path](neosr/models/image.py#L382) omit dataset identity; inference output paths retain dataset identity but discard recursive subdirectories. `setA/frame.png` and `setB/frame.png` therefore collide. Different extensions with the same stem can also collide.

LQ images use a shared basename path and are only saved if absent, so a later sample can be displayed beside the wrong previously saved LQ. TensorBoard names have similar ambiguity.

**Repair:** use dataset identity plus a stable relative source path for SR, LQ, and TensorBoard names, with a deterministic extension-disambiguation rule. Add a two-dataset/two-subdirectory test; a single flat validation set does not reveal the problem.

### F23 — Rank-zero validation failures are not distributed coherently

**P2 operational; source verified.** [base.validation](neosr/models/base.py#L177) enters a barrier, runs validation only on rank zero, then enters a second barrier. A rank-zero image/metric/write error exits before that second barrier; other ranks are already waiting there. This limitation is also acknowledged in [T2 design](docs/refactor/design/T2.md).

An elastic launcher may terminate peers promptly, while other launch arrangements can wait for the process-group timeout. This is an error-path robustness problem, not proof that normal DDP training hangs.

**Repair:** catch the rank-zero validation outcome, communicate a success/error status to peers, and have every rank leave or raise coherently. Keep metric cleanup in `finally`. Test failure injection in a bounded-time multiprocess test.

### Additional trainer policy issues

| Issue | Evidence and impact | Recommendation |
|---|---|---|
| Hard output clamp during training | [image.py:224](neosr/models/image.py#L224) uses a hard `[1/255,1]` clamp. Raw predictions outside this interval have zero local clamp gradient; exact black is excluded | Treat as an explicit objective choice. Compare unclamped reconstruction losses with bounded inputs only for losses needing them; monitor saturation fraction |
| `fast_matmul` bundles different numerical tradeoffs | [runtime.py:96–101](neosr/runtime.py#L96) enables medium FP32 matmul precision, cuDNN TF32, and **full FP16 accumulation** together | Separate/document FP16 accumulation. It has a different overflow risk from TF32; validate on the actual FP16 workload |
| BF16 hardware check is incomplete | [runtime.py:38–47](neosr/runtime.py#L38) branches on compute capability instead of the actual BF16 support query | Probe capability early; distinguish native and emulated support. Do not claim every pre-Ampere GPU necessarily fails |
| CPU AMP request is ineffective | [phase.py:74](neosr/models/phase.py#L74) hard-codes CUDA autocast/scaler policy | Reject or clearly report ineffective CPU AMP, or implement a distinct supported CPU policy |
| Seed and deterministic performance policy are coupled | A supplied `manual_seed` disables cuDNN benchmarking and selects deterministic algorithms with `warn_only=True` | Separate seed selection from strict deterministic policy. Warning-only is not a guarantee that every operation is deterministic |
| Exhaustive cuDNN benchmarking | [runtime.py:93](neosr/runtime.py#L93) sets `benchmark_limit=0` when nondeterministic | Measure startup and variable-resolution costs; use a finite budget if exhaustive search costs more than it recovers |
| Final metric bookkeeping is not in the final state | [train.py:248–251](train.py#L248) saves before final validation, so a newly best final metric is absent from that saved state's best dictionary | Low priority and already deferred in PLAN. If extending runs matters, persist updated metric metadata without weakening checkpoint guarantees |
| Finite-gradient skip is not a full transaction | BN/QAT/statistical buffers may change during a forward whose optimizer step is skipped | Document this; monitor skip streaks and statistics. Do not claim that a skipped update rolls back the whole model state |

PyTorch explicitly documents the increased overflow likelihood of full FP16 accumulation in its [CUDA numerical controls](https://docs.pytorch.org/docs/2.9/notes/cuda.html). Its [reproducibility guidance](https://docs.pytorch.org/docs/2.9/notes/randomness.html) also distinguishes seeding, deterministic kernels, and cross-platform/version reproducibility. These are tuning contracts that should be exposed accurately, not generic reasons to disable AMP or performance features.

## 7. Losses, metrics, speed, and memory

### F22 — TOPIQ computes attention matrices it does not need

**P2 resource/performance; allocation behavior reproduced.** The MultiheadAttention calls in [topiq.py:36](neosr/metrics/topiq.py#L36) and [68](neosr/metrics/topiq.py#L68) use default `need_weights=True` but discard the returned attention weights.

A CPU profiler probe with 256 tokens and four heads showed separate four-head logits and softmax matrices, each 1 MiB in FP32, plus a 256 KiB averaged attention matrix. A 3840×2160 image can produce a 120×68 coarse grid, or 8,160 tokens; each four-head FP32 attention matrix is then approximately **0.99 GiB**. At 7680×4320, the analogous matrix is approximately **15.64 GiB**. These are shape-based estimates, not measured GPU peaks.

Generator tiling does not solve this: metrics receive the reconstructed whole image. TOPIQ can therefore become the memory bottleneck after generator inference successfully finishes.

**Repair:** use `need_weights=False` for both calls and check metric score parity/tolerances with real weights. This allows PyTorch to choose its optimized attention path; the [MultiheadAttention documentation](https://docs.pytorch.org/docs/2.9/generated/torch.nn.MultiheadAttention.html) explicitly recommends that setting when weights are not needed. Backbone activations and attention compute still remain. Any image-resize policy must be explicit because it changes the metric protocol.

### Loss and metric observations

| Area | Current behavior | Improvement and validation |
|---|---|---|
| Focal frequency loss synchronization | [ff_loss.py:100–104](neosr/losses/ff_loss.py#L100) reads `.item()` twice to inspect a matrix already constrained in the implementation | Remove redundant checks or make expensive diagnostic checks opt-in after proving invariants; measure CUDA sync count |
| Perceptual metrics round trip | `image.validate` converts SR to CPU uint8; `calculate.py` converts that image back to RGB tensors and transfers it to the metric device | A tensor-native path could remove transfers, but must reproduce current clipping, rounding, channel order, crop, and normalization semantics |
| Work with no outputs/metrics | Validation always calls `tensor2img` even if image saving, TensorBoard images, and metrics are all disabled | Skip the conversion when nothing consumes it |
| DISTS/TOPIQ lifecycle | Models are cached for one validation call and cleared afterward | This bounds retained memory. For many validation datasets, consider reuse across the entire validation event, followed by one cleanup |
| Heavy perceptual objectives | 84 templates enable FDL, MS-SSIM, consistency, LDL, and GAN together | Profile loss forward/backward separately from G/D. Supply a lightweight reconstruction baseline before tuning complex objectives |
| FDL DINO precision | Existing implementation intentionally defaults to FP32, with BF16 backbone behavior opt-in | Preserve that caution; historical gradient differences are not eliminated by similar scalar loss values. Require SR quality A/B before changing the default |
| KL objective semantics | Channel-softmax KL does not constrain common channel offsets and its reduction can depend on spatial extent | Use only with a clearly stated objective and compatible reconstruction terms; do not treat it as a general pixel-distance substitute |
| Metric direction | Best tracking defaults to higher-is-better; provided DISTS examples correctly specify lower-is-better | Encode/validate known metric direction or reject invalid `better` strings; avoid accidental reverse selection in custom configs |

The scoped loss tests and 33 precision smoke probes found no new generic finite-value failure. That does not validate pretrained VGG/DINO/DISTS behavior, extreme frequency-domain gradients, all patch sizes, or CUDA autocast. The previously implemented LDL/MS-SSIM finite guards, detached targets where appropriate, and frozen feature weights should be retained.

### Memory: actual retention versus temporary demand

| Component | Observed lifetime | Risk and measurement |
|---|---|---|
| Loss context/logs | New context per micro-batch; logged tensors detached; meter resets | No accumulating autograd history identified in this path |
| OTF PairPool | Two fixed buffers, allocated once and reused | Large **bounded** VRAM demand; inspect configured queue and patch size |
| CUDA prefetcher | Current/next batch with consuming-stream recording | Bounded retention; overlap and allocator lifetime require CUDA validation, but `record_stream` is present |
| Metric models | Cache cleared in validation `finally` | Reserved allocator memory can remain after tensor release; this alone is not a leak |
| MFHGMoE expert outputs | Per-forward dense list/stack/product | Potentially large activation peak, addressed by F10 |
| HiT position bias | Per-forward quadratic construction | Algorithmic memory blow-up, addressed by F09 |
| TOPIQ attention | Per-metric quadratic temporaries | Addressed by F22; generator tiling does not bound it |
| Tiled output | CPU stitching, with full LQ still on device and full SR in host memory | GPU tiling bounds generator activation work, not total host/device image storage |
| Checkpoint exports | Numbered model exports accumulate; `keep_states` only prunes `.state` files | Unbounded **disk usage over training duration**, not a RAM/VRAM leak |
| LMDB preparation | Entire encoded dataset held until writing begins | Dataset-size-dependent host RAM and worker overhead, F18 |

For FP32 RGB OTF pairs, approximately:

`pool_bytes = rounded_queue_size × 3 × LQ_patch² × (1 + scale²) × 4`.

At queue size 180 and x4, LQ 64×64 uses approximately **143.44 MiB**; LQ 256×256 uses approximately **2.24 GiB**, before active batches, optimizer state, activations, or allocator overhead. Queue rounding can change the exact size. Expose this estimate at startup and make queue size a deliberate memory/mixture-diversity tradeoff.

A 3840×2160 RGB FP32 output is about 95 MiB before conversion copies; an 8K output is about 380 MiB. CPU stitching, float conversion, clamping, permutation, NumPy conversion, and uint8 output can overlap in memory. Avoid diagnosing these peaks as a per-iteration leak without measuring live allocations over time.

### Performance opportunities, ranked by expected mechanism

| Rank | Opportunity | Why it may help | Required evidence before adoption |
|---|---|---|---|
| 1 | Remove dead/disconnected computation and fix expert/HiT scaling | Eliminates substantial unnecessary work and memory | Gradient and mathematical equivalence first; then runtime/quality measurement |
| 2 | Avoid unused TOPIQ attention weights | Removes explicit quadratic matrices and enables optimized attention dispatch | Real-weight metric score comparison; full-resolution memory profile |
| 3 | Tune loss composition and backbone precision deliberately | Frozen feature networks can dominate a light generator's cost | Per-loss GPU traces, gradient comparison, SR quality A/B |
| 4 | Separate immutable deployment graphs from trainable eval graphs | SPAN/SPAN+/FlexNet recompute fused kernels every eval forward to preserve freshness | Explicit deploy conversion parity and lifecycle tests; do not reintroduce stale caches |
| 5 | Reuse graph-independent masks/grids with bounded caches | Several attention models rebuild masks, sometimes on CPU before copying to GPU | Shape/device/dtype keys, bounded capacity, device-move invalidation, compile compatibility |
| 6 | Remove redundant image/metric transfers and conversions | Validation does GPU→CPU→GPU work for neural metrics | Preserve exact existing metric input protocol or version the changed protocol |
| 7 | Tune loader workers, pinning, and CPU library threads | `'auto'` uses CPU count per rank; decode/OpenCV/BLAS threading can oversubscribe | Measure batch wait and CPU utilization with the real storage and image sizes |
| 8 | Benchmark memory layout separately from AMP | Current model layout follows `use_amp`; CNNs and transformer reshapes may prefer different layouts | Compare contiguous/channels-last per family, including layout-copy cost |
| 9 | Profile compile on representative fixed and variable shapes | Current generator-only compile choice avoids known spectral-norm D problems | Compile time, graph breaks, recompiles, steady throughput, and numerical equivalence |
| 10 | Reduce checkpoint/validation disruption and define retention | Save-time GPU/CPU transfer and disk I/O may dominate short intervals | Wall-clock training throughput including validation/saving; failure-safe behavior |

Mask caching needs restraint. For a 512×512 LQ image with 16×16 windows, a FP32 `[1024,256,256]` mask alone occupies **256 MiB**. An unbounded cache keyed by every validation size would create a real retention problem. Cache only bounded graph-independent tensors, never learned positional outputs with attached training graphs.

DySample also casts half-pixel coordinates to the input dtype before normalization. BF16 loses increasingly fine coordinate resolution as image dimensions grow. Maintaining FP32 grid arithmetic is a worthwhile high-resolution fidelity experiment, but the existing tests intentionally preserve historical rounding; changing this requires image/output comparison, not a silent cleanup.

### A practical CUDA profiling protocol

1. Fix the GPU, PyTorch/CUDA/cuDNN versions, driver, architecture, scale, batch, patch, losses, optimizer, AMP dtype, layout, and compile mode. Record other GPU processes and storage setup.
2. Separate initialization/compilation/cuDNN search from steady-state timing. Use sufficient warmup and repeated interleaved trials; synchronize only around measurement boundaries or use CUDA events appropriately.
3. Measure generator forward/backward, each perceptual loss, discriminator passes, optimizer/EMA, OTF degradation, loader wait, validation, and saving separately, then measure full wall-clock throughput including all scheduled work.
4. Track peak allocated and reserved VRAM, host RSS, worker RSS, pinned memory, and disk growth. Use allocated-memory plateaus and memory snapshots to investigate retention; `nvidia-smi` alone cannot identify live-tensor leaks.
5. Report median and variability, images/second and optimizer-updates/second, effective batch, peak memory, skip counts, and image-quality deltas. Accumulation changes the meaning of an iteration.
6. Accept an optimization only after forward/gradient checks and a representative training-quality run. Speed at a changed objective, reduced resolution, or silently different attention equation is not an equivalent speedup.

## 8. Tooling, installation, reproducibility, and maintainability

### F24 — Windows aliases do not persist their installation location

**P2; source verified.** [install_windows.ps1:123](install_windows.ps1#L123) constructs the profile functions inside a single-quoted here-string. They contain literal references to `$INSTALL_DIR`, but that variable is established by the installation process and not defined in the saved profile content. In a later PowerShell process, the functions cannot reliably change to the installation directory before invoking relative `train.py`/`test.py`/`convert.py` paths.

The script also does not consistently stop on nonzero native-command exit codes before printing installation success, and repeated runs append repeated profile definitions. Linux aliases similarly invoke relative script paths and therefore depend on the caller's working directory.

**Repair:** persist a properly quoted literal repository path in the generated wrappers, check native exit status, and update an identifiable profile block idempotently. Test alias invocation from a fresh shell in a different directory. Avoid reworking the whole installer into a large framework.

### F25 — The test suite needs capability-based coverage and portable execution

The existing suite is a strength, but the actual run demonstrated assurance gaps:

- The Windows entry-point fixture failure occurs before training behavior is exercised. Use a TOML serializer or properly quoted/escaped paths; `.as_posix()` is sufficient for these Windows test paths when used consistently.
- Gloo/NCCL/CUDA requirements need explicit environment checks and a maintained supported runner. Do not turn unexpected distributed failures into unconditional skips; distinguish unsupported infrastructure from failing assertions.
- The general architecture test checks gradients that exist but does not require gradients for every expected trainable branch. F02/F03/F11 can pass that style of test.
- CUDA-dependent architecture checks are skipped on CPU. Small device-independent gradient and argument-contract tests should remain runnable on CPU.
- Historical-output comparisons depend on unavailable baseline fixtures or earlier source revisions. State-dict shape hashes establish compatibility, not output/gradient correctness.
- No `.github/workflows` files or declared pytest/Ruff development dependency group were present in this checkout. A clean locked environment does not supply a complete documented audit/test command.
- Normal Ruff configuration excludes `*_arch.py`. The isolated correctness scan found undefined names and a duplicate `PatchMerging` definition that the default exclusion hides.

**Repair:** maintain a fast CPU correctness suite and a separate supported CUDA suite. Add targeted tests for the exact failure classes in this report. Keep architectural style exceptions if necessary, but run correctness checks such as undefined names across the entire source tree. A matrix should state what actually passed for each family and option.

### Reproducibility and dependency issues

| Area | Evidence | Recommended change |
|---|---|---|
| DINOv2 code is not pinned by the lockfile | [fdl_loss.py:180](neosr/losses/fdl_loss.py#L180) loads `facebookresearch/dinov2` without a commit/tag | Pin a reviewed revision, record model/weight identifiers, and support a local cached source. A dependency lock does not pin remote Hub code |
| Network access during model/loss construction | Feature backbones and metric weights can download on first use; DISTS can write beside source | Make cache/preflight behavior explicit; fail with a clear missing-artifact message for offline runs |
| Optional tools versus mandatory dependencies | `torchprofile`/W&B are optional imports; `pytorch-optimizer` has no import under current runtime source, while ONNX/ORT are needed mainly for conversion | Remove confirmed unused runtime dependencies after clean-environment verification; put export/logging tools in optional groups |
| Nightly and broad package requirements | `tb-nightly` and permissive minimum versions exist, although a lockfile is committed | Retain locked reproducible installs; distinguish tested lock updates from arbitrary dependency resolution |
| Checkpoint provenance | Saved state records network configuration and limited runtime settings, not a full environment/data/remote-backbone fingerprint | Store effective config, package versions, git revision, model artifact IDs, and dataset-manifest hash where practical |
| Resume reproducibility | Rank-zero model RNG is restored; data order/worker augmentation/OTF queue and other ranks' stochastic continuation are explicitly not order-exact | Preserve that honest contract. Do not market same-config synthetic resume tests as bit-identical real DataLoader continuation |
| Performance catalog | `architecture_categories.json` contains stale names and lacks a complete workload/hardware measurement protocol | Regenerate from a reproducible harness or label it historical/non-comparable |

### Complexity and overengineering review

The current explicit per-package lookup tables and small shared factory are reasonable. Replacing them with a new registry framework or deep class hierarchy would add complexity without addressing the observed bugs.

| Tag | Concrete simplification | Boundary |
|---|---|---|
| `delete` | DITN's discarded attention calculation and genuinely unreachable helpers such as OmniSR's broken guided-filter helper | Preserve the intended trained attention equation; do not delete parameters merely because a current bug disconnects them |
| `delete` | Unused `pytorch-optimizer` dependency if a clean dependency audit confirms no external tooling requires it | No runtime import was found in this tree |
| `shrink` | Duplicate `PatchMerging` definition in DRCT and small unused locals/imports found by Ruff | Check imported/public usage before removal |
| `native` | Use `need_weights=False` to access PyTorch's optimized attention implementation where equivalent | Preserve masks, positional bias, scale, and dropout |
| `stdlib` | Correct final-suffix/path handling and bounded pool iteration in LMDB utilities | Include duplicate-key/error handling, not just shorter syntax |
| `shrink` | Replace source-comment task labels/stale migration prose with stable contracts where they obscure current behavior | Preserve substantive design decisions and compatibility notes |
| `yagni` | Remove or reject no-op configuration flags instead of retaining an illusion of support | Implement the capability if it is actually needed |

No invented line-count or dependency-saving total is supplied: the safe deletion size requires implementation and compatibility checks. Repeated architecture code is not automatically removable duplication; different papers/models can require different equations and checkpoint layouts.

## 9. Missing functionality and TODO triage

These items are separated from bugs because absence is not necessarily a defect. Some are already explicitly deferred in [PLAN.md](docs/refactor/PLAN.md).

| Item | Current status | Value / priority |
|---|---|---|
| Plain-optimizer warmup | Legacy warmup removed; schedule-free optimizers provide their own warmup, ordinary optimizers lack equivalent supported composition | Useful for transformer training; add a small, tested scheduler composition rather than reviving removed flags |
| GAN start/delay and regularization | Delayed GAN objective and R1/R2 regularization are deferred | Evaluate for specific discriminator/data instability; not a universal requirement for all SR training |
| Long stability acceptance | Historical acceptance covers 5,000 updates across three workloads, not all families or long convergence | Add representative long runs after correctness fixes, including failure/skip and validation trends |
| Architecture capability metadata | Deferred and currently absent | High practical value for input shapes, checkpointing, AMP/compile, tiling, and export |
| Dynamic ONNX coverage | One-shape float verification is present; several architectures are excluded or constrained | High priority where deployment is advertised; multi-shape tests before universal dynamic claims |
| FP16 export flag | `-fp16` is documented as temporarily doing nothing; full optimization is disabled | Explicitly fail/warn or implement. Do not report success as if a half export was produced |
| New ONNX exporter | `dynamo=True` remains a converter TODO | Investigate per architecture with exact output/shape tests, not a flag-only migration |
| Configurable OTF blur-kernel sizes | Fixed range 7–21 and 21×21 padding/pulse assumptions remain | Only expose configurability when dataset generation, filtering, and validation agree; a schema field alone is insufficient |
| LMDB map-size strategy | An explicit TODO questions the single-image-times-ten estimate | Fix alongside bounded streaming and error handling |
| Best-model artifact selection | Best metrics are tracked; normal save exports current selected evaluation weights | Optional but useful: define best-checkpoint retention by metric and dataset, without conflating it with resume state |
| Model-export disk retention | `keep_states` retains training states, not numbered model exports | Add an explicit separate retention policy and report predicted disk cost |
| Exact distributed/data resume | Deliberately outside current scope | Only promise it if sampler cursor, worker/degrader state, queue, and rank-local RNG are explicitly persisted/tested |

## 10. Training-stability and SISR quality recommendations

The code can run, produce finite losses, and improve PSNR while still learning the wrong architecture or optimizer recurrence. Numerical finiteness, throughput, and reconstruction quality are separate acceptance dimensions.

1. **Create a correct reconstruction baseline.** Select an architecture whose active gradient paths are verified, use a tested ordinary optimizer or verified schedule-free implementation, and start with an explicit reconstruction objective. The shipped six AdamW/L1 pretraining configs are a useful simpler comparison point; they do not by themselves certify every architecture.
2. **Validate the image domain and pairing.** Check LR/HR alignment, scale, color order, bit depth, normalization, crop geometry, and duplicate paths. Separate train and validation manifests; this audit did not inspect the user's real datasets or establish absence of data leakage between splits.
3. **Measure loss influence, not just scalar magnitudes.** Loss weights and per-term parameter-gradient norms/directions reveal whether a perceptual/GAN term dominates. Frequency/angle objectives especially need gradient-level comparisons when changing precision.
4. **Make clipping and AMP behavior visible.** Record preclip gradient norm, clipping frequency, nonfinite skip streaks, scaler level for FP16, and saturation fraction from the hard output clamp. A permanently clipped run can hide a poorly scaled objective even when it never becomes NaN.
5. **Choose EMA deliberately.** Evaluate live and EMA weights during early investigations; schedule-free x/y and EMA are distinct weight sets. Warmup changes early behavior and needs resume provenance. Historical low early EMA PSNR should not be misdiagnosed without comparing these sets.
6. **Treat GAN performance as a quality tradeoff.** Track reconstruction metrics and perceptual quality/artifacts on a fixed validation set. Low adversarial loss or bounded logits alone does not establish a balanced perceptual-distortion outcome.
7. **Verify deployment with real images and shapes.** Compare eager eval, exported runtime, static/dynamic shapes, tiled/full inference, and QAT/fused variants on representative data. Report metric protocol: RGB versus Y, crop border, clamping/quantization, scaling, and reference alignment.
8. **Use ablations before claiming improvements.** Correct one major behavior at a time; compare equal data/update budgets and multiple seeds where variability matters. A new optimizer trajectory, different gate function, or BF16 feature backbone is an experiment requiring quality evidence, even if it fixes a mathematical defect or improves speed.

Aether's custom normalization, reparameterization, and QAT deserve a dedicated long-run validation lane. The current construction order correctly prepares topology before optimizer/EMA wrappers, and observer freezing during evaluation is a strength. Unconstrained learned temperatures, full-half public-method behavior, observer calibration, folded normalization equivalence, and actual INT8 runtime quality remain areas for targeted experiments rather than blanket claims of instability.

## 11. Recommended repair sequence and acceptance criteria

| Stage | Work | Acceptance criteria |
|---|---|---|
| 1: learning correctness | F01–F04 and F11/F12; first protect default ASID/AdanSF and MFHGMoE users | Reference optimizer trajectories; expected parameter gradients; manual/SDPA output and gradient parity; explicit compatibility policy |
| 2: data/deployment integrity | F07/F08 and F13–F17/F21 | Multi-shape runtime equivalence; correct normalization round trip; grayscale policy; hidden-file/order tests; unique LMDB keys and output paths |
| 3: advertised option reliability | F05/F06/F19/F20 | Checkpointed trainer backward parity; reject unsupported flags; clear resume diffs; finite/range validation |
| 4: resource scaling | F09/F10/F18/F22 | Peak-memory profiles at increasing patch/image sizes; bounded queue/worker/cache design; retained semantics and quality |
| 5: operational reliability | F23–F25, export retention, offline artifact policy | Bounded-time distributed failure tests; fresh-shell installer tests; portable CI; reproducible locked test commands |
| 6: optimization and quality | Layout/compile/loss precision/loader/deploy tuning | Repeated hardware benchmarks, long-run memory plateau, convergence and validation-quality comparison |

The first stage should not be hidden inside a broad cleanup commit. Optimizer semantics, attention semantics, routing semantics, and data-domain changes need separate reviewable patches with narrow regression tests. Conversely, do not build a new trainer framework to fix a small argument error, sorting omission, or output-path collision.

### High-value regression tests

| Test | What it must prove |
|---|---|
| Scalar and vector optimizer reference trajectories | Correct first/multiple steps, decay, warmup, evaluation, serialization, and continuation |
| Expected trainable-gradient census | No accidentally disconnected parameters in default and supported optional paths |
| Manual attention versus SDPA | Same masks, bias, temperature, dropout policy, outputs, and gradients within stated tolerance |
| Checkpoint on/off through actual Phase backward | Works with explicit backward inputs, accumulation, and relevant stochastic/stateful operations |
| ONNX spatial/batch matrix | Same-shape success plus rectangular, larger/smaller, boundary-padding and batch-change correctness |
| Normalized-domain identity round trip | Correct saved RGB and metrics for an analytically perfect output |
| Distributed manifest permutation | Identical images on different filesystem orderings still produce consistent index maps |
| LMDB duplicate/corrupt/empty-input cases | No silent overwrite; original failure path surfaced; partial output handled |
| Cross-dataset artifact names | Every source produces its own SR/LQ output and logging key |
| Resource-scaling test | Peak allocation follows intended algorithmic growth; no all-history cache growth |
| Resume with edited settings | Saved-versus-current conflicts are surfaced rather than silently mixed |

## 12. Reproduction examples and evidence references

The following minimal examples illustrate important findings without modifying project code. Run from the repository with a compatible PyTorch environment. Exact floating values may differ slightly by build; the tested invariant is the direction/absence of a gradient, not the last decimal digit.

### Schedule-free Adan first step

```python
import torch
from neosr.optimizers import adan_sf

p = torch.nn.Parameter(torch.tensor([1.0]))
opt = adan_sf([p], lr=0.01, weight_decay=0, schedule_free=True)
opt.train()
p.grad = torch.ones_like(p)
opt.step()
print("train:", p.item())  # observed: 1.0
opt.eval()
print("eval:", p.item())   # observed: 1.0002040863, uphill for positive gradient
```

### ASID gradient ownership

```python
import torch
from neosr.archs.asid_arch import Channel_Attention

m = Channel_Attention(dim=8, heads=1, window_size=4)
m(torch.rand(1, 8, 8, 8, requires_grad=True)).square().mean().backward()
print([name for name, p in m.named_parameters() if p.grad is None])
# Includes temperature, qkv.weight, and qkv_dwconv.weight on the default path.
```

### MFHGMoE router gradients

```python
import torch
from neosr.archs.mfghmoe_arch import HMoE

m = HMoE(4, 8, 4)
x = torch.rand(1, 4, 4, 4, requires_grad=True)
route_input = torch.rand_like(x, requires_grad=True)
m(x, route_input).square().mean().backward()
print("routing input has gradient:", route_input.grad is not None)  # False
print([name for name, p in m.gate.named_parameters() if p.grad is None])
```

### Hidden-file discovery failure

```python
import tempfile
from pathlib import Path
from neosr.utils.misc import scandir

with tempfile.TemporaryDirectory() as directory:
    Path(directory, ".DS_Store").write_bytes(b"metadata")
    print(list(scandir(directory, recursive=True)))  # NotADirectoryError
```

### Equivalent reproduction commands

The commands below preserve the executed selections; `-p no:cacheprovider` is added consistently to avoid repository pytest caches. Some agent invocations instead used `--disable-warnings`, which changes reporting but not the selected tests.

```text
python -m pytest -q -p no:cacheprovider tests/test_config.py tests/test_registry.py tests/test_hygiene.py tests/test_checkpoint.py tests/test_phase.py tests/test_phase_state.py tests/test_model.py tests/test_optimizers.py tests/test_ema_warmup.py tests/test_stop.py tests/test_entry.py tests/test_runtime.py

python -m pytest -q -p no:cacheprovider tests/test_data.py tests/test_otf_degradation.py tests/test_tiling.py tests/test_metrics.py -k "not real_weights"

python -m pytest -q -p no:cacheprovider tests/test_losses.py -k "not vgg_perceptual_loss and not dists_loss and not fdl_loss and not frozen_weights_and_options and not fdl_projections_are_seeded and not b1 and not dists_metric_path"

python -m pytest -q -p no:cacheprovider tests/test_losses_pipeline.py -k "not real_constructors"

python -m pytest -q -p no:cacheprovider tests/test_archs_a1.py tests/test_archs_a2.py

ruff check --isolated --no-cache --select F,E9 neosr train.py test.py convert.py create_lmdb.py --output-format concise
```

Detailed session evidence and larger reproduction scripts were saved outside the repository in the audit machine's temporary directory:

- `neosr-training-audit-gpt6.md`
- `neosr-architecture-audit.md` and `neosr-architecture-repro.py`
- `neosr-data-losses-audit-gpt61.md`, `neosr-data-losses-probe.py`, and `neosr-data-extra.py`
- `neosr-audit-core-tests.txt`, `neosr-audit-root-repros.txt`, and `neosr-audit-clean-e2e.txt`

These temporary files are not required to understand this report and are not durable project artifacts. Source locations, triggers, observed outcomes, limitations, and repair criteria are included here so the report remains useful after those files disappear.

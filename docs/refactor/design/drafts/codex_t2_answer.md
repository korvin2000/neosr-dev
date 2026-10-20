I inspected both projects and the installed **PyTorch 2.9.1+cu128** source. No files were changed.

1. **Checkpoint split — rank A > B > C. Make `.state` self-contained, including live G, live D, and EMA.**

   **Rationale:** `models/net_g_<iter>.pth` is a distributable inference artifact; `.state` is the authoritative training snapshot. Keep inference output as `{"params": inference_state_dict}`, selecting EMA when enabled. Putting live weights in `.state` duplicates storage, but removes dependency on matching files, export representation, filenames, and pruning history.

   B is reasonable when weights need independent retention, but requires a manifest and coordinated publication. The competitor’s split adds those dependencies without giving this project a compelling benefit. C makes inference distribution unnecessarily large and exposes training internals to inference consumers.

   Store a schema version, checkpoint ID, architecture/QAT metadata, named optimizer/scheduler/scaler states, EMA weights/count/decay, counters, and RNG. Include **all persistent model buffers**, not just parameters.

   **D EMA:** remove it from the ordinary training design. It can support specific teacher or ensemble experiments, but currently has no consumer. Resume needs live D and its optimizer; exporting D EMA provides neither.

   **Pitfalls:** use tensors and primitive containers so new checkpoints load with explicit `weights_only=True`; raw NumPy RNG objects can break that. Load onto CPU first. [PyTorch serialization](https://docs.pytorch.org/docs/2.9/notes/serialization.html)

   Keep legacy loading in one adapter: old `.pth` variants remain pretrained inputs; old `.state` resumes warn about missing fields and unrecoverable live weights. **Confidence: H.**

2. **Counters — rank explicit update counters > micro-batch `iter`. Save only at completed update boundaries.**

   Define `iter` and numeric filenames as **completed accumulation windows**, meaning logical optimizer-update attempts. Preserve existing update-count filenames. An AMP-skipped attempt still advances this logical clock; record actual optimizer execution separately.

   Persist:

   - `iter`/`update_step`;
   - `micro_batches_consumed`;
   - successful optimizer calls for G and D independently;
   - EMA update count;
   - current sampler epoch and next unconsumed batch position;
   - accumulation factor and phase, with phase required to be zero for normal checkpoints.

   Scheduler state already carries scheduler progress; explicitly define whether each scheduler advances on successful optimizer calls or logical attempts. I recommend successful calls for optimizer-linked schedules. Keep checkpoint, validation, and logging frequencies in logical update units.

   Save after optimizer calls, scaler updates, EMA, schedulers, counter advancement, and gradient clearing. Resume starts with the **next** micro-batch.

   **Pitfalls:** `GradScaler.step()` can skip G and D independently; its return value does not reliably identify a skip. Advance each EMA only when its optimizer actually executes. [AMP behavior](https://docs.pytorch.org/docs/2.9/notes/amp_examples.html)

   Mid-accumulation recovery requires gradients plus scaling and synchronization state; counters alone are insufficient. Defer graceful-stop requests until a boundary. An interrupt inside an update should retain the previous committed checkpoint. Legacy `micro_batches_consumed ≈ iter × accumulate` is an explicitly warned inference. **Confidence: H.**

3. **RNG/data order — rank documented stochastic continuation > an expensive exact-replay subsystem > saving no RNG.**

   Saving RNG is worthwhile because it is cheap and preserves useful state, even without exact replay.

   Save per-rank Python, torch CPU, relevant CUDA, NumPy global RNG, and **every owned NumPy `Generator.bit_generator.state`**. Also save explicit DataLoader generator state, sampler epoch/seed, and the consumed batch cursor. Restore training RNG after initialization/loading/calibration, immediately before training resumes.

   **Minimal honest policy:** restore model/optimizer progress and available RNG; reconstruct sampler order at the saved cursor; reseed workers; reset the OTF queue. State plainly that augmentations and degradation pairs can differ after restart.

   **Pitfalls:** spawned workers, persistent-worker RNG, prefetched batches, and GPU queue contents are outside ordinary parent RNG snapshots. A sampler iterator may have advanced beyond the batches actually consumed. Exact replay additionally needs queue tensors/pointer and reproducible worker augmentation, commonly through randomness keyed by sample/epoch/occurrence. Sampler epoch alone does not restore position. PyTorch also distinguishes independent NumPy generators from the global RNG. [Reproducibility](https://docs.pytorch.org/docs/2.9/notes/randomness.html)

   Encode RNG arrays as tensors/primitives and gather rank-local state before rank-zero writing. **Confidence: H.**

4. **Schedule-free — rank matched train-representation resume snapshots > matched eval snapshots. Use one inference-materialization protocol everywhere.**

   **Rationale:** saving live training weights `y` with optimizer `train_mode=True`, `z`, moments, and counters captures the working representation without introducing a swap.

   Protocol:

   - **Export:** export EMA directly when selected. Otherwise materialize SF eval weights `x`, then copy/export them. `model.eval()` alone does not perform the SF transformation.
   - **Resume save:** snapshot live G/D in train representation and their matching optimizer states; perform no SF swap.
   - **Validation:** use EMA directly; otherwise use the same `x` materialization as export.
   - **Resume:** restore weights and optimizer state together, then ensure train representation. Calling optimizer `.train()` is a no-op for a correctly restored train snapshot; an eval snapshot must first load its saved false mode flag, then transform once.

   A coherent eval snapshot is valid, but both model and optimizer must represent that same instant. The optimizer docstrings’ “save in eval mode” convention is useful for inference checkpoints; it does not make matched train-mode training snapshots invalid. See the local [AdamW SF transformation](/home/korvin/apps/neosr-dev/neosr/optimizers/adamw_sf.py:75).

   **Pitfalls:** eval/train inverse arithmetic need not restore weights bitwise. Prefer materializing on a copy; otherwise snapshot original parameters and mode flags and restore them in `finally`, including partial-entry failures. Rank-zero-only round trips must not introduce parameter drift relative to other ranks.

   Keep EMA explicitly defined as EMA of train-mode `y`, preserving current behavior. Never apply the live optimizer’s swap to EMA. SOAP needs additional persistence work described below. **Confidence: H for protocol; M for benefits of combining SF and EMA.**

5. **Wrapping/EMA — rank correctly constructed `AveragedModel` > custom EMA for this reference-quality framework. Retain an explicit bare-module handle.**

   Recommended lifecycle:

   `build bare → load fresh pretrained → architecture-specific fusion/QAT prepare → device/channels_last → fresh calibration → load resume model state → DDP → EMA from synchronized bare module → optimizer/scheduler/scaler setup and restore → compile`

   Resume first inspects metadata to construct the saved topology; it skips fresh calibration and loads QAT state **after preparation**. Construct optimizers only after all parameter-replacing transformations and device placement.

   For DDP, prefer `torch.compile(DDP(raw_model))`: PyTorch 2.9 explicitly recommends this order for its DDP bucket optimizations. Single GPU compiles the bare model. Treat DataParallel as a compatibility fallback; compile its underlying module if retained. [DDP compilation order](https://docs.pytorch.org/docs/2.9/notes/ddp.html)

   Create EMA from the bare module after DDP’s initial synchronization, never from compiled/DDP wrappers. Keep EMA uncompiled initially. `AveragedModel` already handles initialization, grouped updates, and buffer copying; its prefixes are serialization details, not sufficient reason to replace it.

   **Pitfalls:** save `n_averaged` and reconstruct the averaging function/decay: the function is absent from `state_dict`. Count zero makes the next update copy the source outright. With `use_buffers=False`, buffers are copied from the source; averaging every buffer is unsuitable for integer counters and QAT observer state. A custom EMA must reproduce these policies; private `torch._foreach_lerp_` adds maintenance exposure. [AveragedModel semantics](https://docs.pytorch.org/docs/2.9/generated/torch.optim.swa_utils.AveragedModel.html)

   Save canonical dictionaries from bare modules. For legacy input, remove known leading wrapper prefixes repeatedly, preserve state-dict metadata, and reject collisions. Remove only the recognized EMA wrapper’s `n_averaged`; never globally delete similarly named model buffers.

   Resume must validate architecture, optimizer type, and ordered parameter-group membership. Parameter names do not automatically control optimizer matching; initialize schedulers before restoring optimizer state. [Optimizer loading](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Optimizer.load_state_dict.html)

   Structurally, keep `base` small, `image` responsible for SR training, `otf(image)` responsible for degradation, and checkpoint serialization in shared helpers. **Confidence: H; M on architecture-specific QAT ordering.**

6. **Atomic writes — rank temporary file + file/directory fsync + replace > replace alone > direct save. Make retention opt-in.**

   Minimal durable Linux pattern:

   `unique temporary file in destination directory → torch.save(file) → flush → fsync(file) → close → os.replace → fsync(parent directory)`

   Clean up temporary files on failure and propagate failure. Each retry gets a fresh temporary file. Same-directory replacement provides atomic visibility; directory fsync addresses rename durability after power loss. [Python `os.replace`](https://docs.python.org/3.13/library/os.html#os.replace)

   **Rationale:** publish the self-contained `.state` first; inference export can be regenerated. Their checkpoint IDs associate them without requiring a transaction spanning two files.

   **Pitfalls:** `state_dict()` is a shallow snapshot containing references to live storage. Before asynchronous writing, recursively create independent CPU snapshots of model **and optimizer** tensors. A dictionary copy is insufficient. Atomic replacement also needs space for the temporary checkpoint while the previous one remains.

   Auto-resume should scan only integer-named finalized `.state` files and validate candidates newest-first. Report fallback explicitly; an explicitly requested corrupt checkpoint should fail.

   Offer `keep_last_n` for resume states, with three a reasonable setting. Prune only after successful publication; preserve pinned milestones. Keep inference retention separate and default deletion off. **Confidence: H.**

7. **Validation — rank EMA > reusable eval copy > guarded bare-live evaluation. For tiles, rank halo-and-crop > overlap blending > isolated tiles.**

   **Rationale:** EMA already provides the dedicated inference module. Without EMA, refreshing a reusable eval copy gives consistent export/validation behavior and isolates training state. Bare-live evaluation remains a useful memory-saving fallback.

   Use a context covering the whole validation pass: snapshot relevant module modes and framework flags, enter eval/SF representation, run inference, and restore in `finally`. Restore previous values instead of unconditionally calling `.train()`. Preserve training RNG around validation when it consumes randomness.

   **Pitfalls:** `torch.inference_mode()` disables autograd machinery but does not disable dropout, SF swaps, or QAT observers. QAT validation should freeze observer updates while retaining the intended fake-quant behavior. A QAT state dict is not automatically a portable inference artifact; export needs a defined topology/conversion contract.

   For tiling, read each output core with an input halo and discard halo output. Weighted overlap blending helps when sufficient context is impractical, but cannot reproduce missing context. Respect architecture/window alignment and scale coordinates. Global-context models remain approximate under tiling.

   Prefer streaming tiles into a **CPU FP32 output accumulator** for the memory-saving path; offer GPU stitching when the output fits. Avoid staging every processed tile. Reflect padding requires each pad extent to be smaller than its input dimension, so tiny images need special handling. **Confidence: H for lifecycle; M for universal tile equivalence.**

8. **DDP — rank rank-zero validation on a bare/eval module with coordinated barriers > all-rank distributed validation. Reject rank-zero-only DDP forward as the general protocol.**

   **Rationale:** all ranks finish the same update, synchronize, rank zero validates through the bare module or EMA, then all ranks synchronize before training resumes.

   **Pitfalls:** DDP forward is a synchronization point and may broadcast buffers even during evaluation; `.eval()` and inference mode do not remove that obligation. A rank-zero DDP forward can conflict with other ranks’ barriers or training collectives. [DDP synchronization contract](https://docs.pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html)

   Bare forwarding also requires that the model itself performs no evaluation collectives. Define buffer authority explicitly, especially for normalization/QAT buffers. Coordinate validation failure reporting before peers resume; account for validation duration in process-group timeouts.

   Likewise, gathering per-rank checkpoint RNG must happen on **all ranks**, outside a rank-zero-only save decorator. **Confidence: H.**

Additional correctness points likely to be overlooked:

- **SOAP’s ordinary optimizer serialization is insufficient.** [soap_sf.py:629](/home/korvin/apps/neosr-dev/neosr/optimizers/soap_sf.py:629) maps state to `(parameter, split_index)` keys. PyTorch 2.9.1’s loader does not remap those tuple keys; I confirmed this in memory. Its eval path also looks up original parameters after the mapping is cleared. Audit serialization and SF swapping together.
- **SAM has another active optimizer.** [image.py:405](/home/korvin/apps/neosr-dev/neosr/models/image.py:405) constructs `sam_optimizer_g` separately from the registered G optimizer; [fsam.py:30](/home/korvin/apps/neosr-dev/neosr/optimizers/fsam.py:30) constructs its own base optimizer. Saving `self.optimizers` misses active SAM/base-optimizer state.
- **Explicit parameter selection is overridden.** [base.py:413](/home/korvin/apps/neosr-dev/neosr/models/base.py:413) chooses recognized keys before honoring `param_key_g`; [convert.py:92](/home/korvin/apps/neosr-dev/convert.py:92) has different precedence. Share one loader with explicit-key-first semantics.
- **Final completion produces no new resume state.** [train.py:628](/home/korvin/apps/neosr-dev/train.py:628) passes `-1`; [base.py:461](/home/korvin/apps/neosr-dev/neosr/models/base.py:461) suppresses state saving for it.
- **QAT behavior includes unsaved Python flags.** [aether_arch.py:530](/home/korvin/apps/neosr-dev/neosr/archs/aether_arch.py:530) and [aether_arch.py:551](/home/korvin/apps/neosr-dev/neosr/archs/aether_arch.py:551) change fusion/quantization flags. Tensor loading alone cannot reconstruct them.
- **Epoch budgeting mixes units.** [train.py:149](/home/korvin/apps/neosr-dev/train.py:149) calculates updates per epoch, then divides a micro-batch total by it. Fix that alongside counter semantics.
- **“Unused” legacy models still execute imports.** [models/__init__.py:23](/home/korvin/apps/neosr-dev/neosr/models/__init__.py:23) imports every model file. Their import-time configuration/RNG effects survive until discovery changes.

Codex session ID: 01a15ae4-2bea-75b1-86d2-a4b05b9a1ec9
Resume in Codex: codex resume 01a15ae4-2bea-75b1-86d2-a4b05b9a1ec9

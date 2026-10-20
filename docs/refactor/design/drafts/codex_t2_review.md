**(1) Ranked defects and omissions**

Read-only review; no files edited. Probes used `.venv/bin/python -B`, PyTorch **2.9.1+cu128**, CPU, and in-memory serialization.

1. **The proposed EMA crashes on ordinary bufferless generators.**  
   **Evidence:** `_foreach_lerp_` on deepcopy parameter lists succeeds, but `_foreach_copy_([], [])` raises `RuntimeError: Tensor list must have at least one tensor`. ESRGAN contains convolutions without registered buffers ([esrgan_arch.py:187](/home/korvin/apps/neosr-dev/neosr/archs/esrgan_arch.py:187)). Uncalibrated QAT introduces another failure: per-channel observer buffers grow after the first forward; the probe failed copying shape `[3]` into `[1]`.  
   **Fix:** Guard empty lists; support dynamically sized observer buffers, preferably through established module loading/copy behavior. **Confidence: H.**

2. **Saving actual best metrics breaks `weights_only=True` resume.**  
   **Evidence:** PSNR returns a NumPy scalar ([calculate.py:65](/home/korvin/apps/neosr-dev/neosr/metrics/calculate.py:65)), stored unchanged in best results ([base.py:112](/home/korvin/apps/neosr-dev/neosr/models/base.py:112)). Probe: serializing `{"best":{"val":np.float64(32)}}` succeeds; loading raises `UnpicklingError`, rejecting `numpy._core.multiarray.scalar`.  
   **Fix:** Convert metric values to Python floats before storage; test a checkpoint containing results from real validation. **Confidence: H.**

3. **Schedule-free export/validation mutates training weights despite `try/finally`.**  
   **Evidence:** The two in-place lerps are mathematical inverses, not floating-point inverses ([adamw_sf.py:75](/home/korvin/apps/neosr-dev/neosr/optimizers/adamw_sf.py:75)). CPU probe: `eval()` → `train()` changed weights by up to `5.96e-08`. Thus the promised unchanged hash fails. Additionally, none of the SF optimizers defines `train_mode_ok`, making §7.3’s assertion ineffective.  
   **Fix:** Derive export weights on copies, or restore an exact saved `y` snapshot; identify SF optimizers explicitly and check their actual groups. **Confidence: H.**

4. **“Self-contained resume” omits the optimizer actually used by SAM.**  
   **Evidence:** SAM creates a separate optimizer ([image.py:405](/home/korvin/apps/neosr-dev/neosr/models/image.py:405)); FSAM owns momentum and another base optimizer ([fsam.py:30](/home/korvin/apps/neosr-dev/neosr/optimizers/fsam.py:30), [fsam.py:59](/home/korvin/apps/neosr-dev/neosr/optimizers/fsam.py:59)). Saving `optimizer_g` alone misses both.  
   **Fix:** Define serialization for the active optimizer stack, including FSAM momentum and its base optimizer, or explicitly reject SAM resume. **Confidence: H.**

5. **The bit-identical resume acceptance criterion contradicts omitted data state.**  
   **Evidence:** `rng()` creates independent NumPy `Generator` instances ([rng.py:14](/home/korvin/apps/neosr-dev/neosr/utils/rng.py:14)); restoring `np.random.get_state()` does not restore them—confirmed by probe. Worker crops/flips consume additional randomness ([paired_dataset.py:151](/home/korvin/apps/neosr-dev/neosr/data/paired_dataset.py:151)); persistent workers are configurable ([data/__init__.py:142](/home/korvin/apps/neosr-dev/neosr/data/__init__.py:142)). Sampler slicing cannot recover those streams or the omitted pair pool.  
   **Fix:** Separate approximate training continuation from exact replay. Exact replay needs explicit data position and reproducible per-sample randomness, plus pool state where applicable. **Confidence: H.**

6. **Checkpoint-before-validation creates another reproducibility gap.**  
   **Evidence:** §7.6 saves before validation. Probe: iterating even a workerless validation `DataLoader` advances Torch RNG. The saved best metrics also exclude validation at that same update.  
   **Fix:** Checkpoint after scheduled validation, or isolate validation RNG and persist whether validation remains pending. **Confidence: H.**

7. **Distributed checkpoint ownership and per-rank state are missing.**  
   **Evidence:** Existing writes are explicitly `@master_only` ([base.py:310](/home/korvin/apps/neosr-dev/neosr/models/base.py:310), [base.py:450](/home/korvin/apps/neosr-dev/neosr/models/base.py:450)); §7.3 drops that protection. Every rank would race on the same `.tmp`. Conversely, saving only rank-zero RNG cannot restore all ranks: initialization deliberately uses `seed + rank` ([options.py:213](/home/korvin/apps/neosr-dev/neosr/utils/options.py:213)).  
   **Fix:** Specify one writer, collective collection of per-rank runtime state, and world-size compatibility rules. **Confidence: H.**

8. **QAT reconstruction happens too late to use checkpoint metadata.**  
   **Evidence:** §7.6 resolves resume after construction; §3 prepares structure before reading the state. Therefore saved `qat: true` cannot drive reconstruction. A Boolean also omits preparation configuration. QAT exports retain prepared/fused keys, while testing loads pretrained weights before preparation. The hook additionally mutates Python flags ([aether_arch.py:551](/home/korvin/apps/neosr-dev/neosr/archs/aether_arch.py:551)) and calibration leaves the model in eval mode ([aether_arch.py:590](/home/korvin/apps/neosr-dev/neosr/archs/aether_arch.py:590)).  
   **Fix:** Inspect and validate checkpoint construction metadata before building; define separate QAT resume/export contracts and explicitly restore training mode. **Confidence: H.**

9. **Halo tiling changes exterior boundary conditions, even with sufficient receptive field.**  
   **Evidence:** For a single zero-padded 3×3 convolution with all-one weights/input, direct output at the corner was **4**, proposed tiled output **9**, maximum difference **5**. Reflecting an exterior halo replaces the model’s normal boundary behavior. ESRGAN uses zero-padded convolutions ([esrgan_arch.py:187](/home/korvin/apps/neosr-dev/neosr/archs/esrgan_arch.py:187)).  
   **Fix:** Clip exterior tile context to the image and crop using actual offsets. Validate architecture-specific alignment from the constructed module; patch divisibility alone does not preserve attention-window phase. **Confidence: H.**

10. **GPU stitching can defeat the purpose of tiled inference.**  
    **Evidence:** §5 allocates the entire padded SR result on GPU, whereas current stitching allocates on CPU ([image.py:792](/home/korvin/apps/neosr-dev/neosr/models/image.py:792)). Large outputs can OOM even when individual tiles fit.  
    **Fix:** Offer CPU or streamed output assembly; avoid allocating output for discarded padding. **Confidence: H.**

11. **Resume validates too little configuration and structure.**  
    **Evidence:** §7.3 iterates whatever checkpoint entries exist, silently leaving newly configured networks/optimizers fresh. Optimizer state loading associates parameters by position, not semantic identity. Changing accumulation retains a `micro` value potentially incompatible with the new boundary modulus. Probe: loading disabled scaler state `{}` into an enabled scaler raises `RuntimeError`.  
    **Fix:** Validate exact component sets, optimizer types and named parameter groups, QAT/EMA policy, precision, and accumulation before mutating anything. Reject incompatible changes or implement explicit migrations. **Confidence: H.**

12. **SOAP’s problem is parameter remapping, not merely weights-only deserialization.**  
    **Evidence:** SOAP indexes state through `(parameter, slice_index)` ([soap_sf.py:629](/home/korvin/apps/neosr-dev/neosr/optimizers/soap_sf.py:629)). A CPU probe using Torch’s optimizer serialization passed `weights_only=True`, but loading retained the old parameter inside the tuple: `key rebound to new parameter: False`.  
    **Fix:** Require stable serialized parameter/slice identifiers and explicit remapping. Test the next update against uninterrupted execution; successful loading is insufficient. **Confidence: H.**

13. **Catching `KeyboardInterrupt` does not establish a consistent update boundary.**  
    **Evidence:** G and D step sequentially ([image.py:748](/home/korvin/apps/neosr-dev/neosr/models/image.py:748)); FSAM temporarily perturbs parameters ([fsam.py:80](/home/korvin/apps/neosr-dev/neosr/optimizers/fsam.py:80)). An interrupt can occur after only one optimizer advances, during perturbation, or before scheduler/counter updates. Zeroing gradients cannot repair this.  
    **Fix:** Defer graceful interruption until a completed transaction boundary; otherwise retain the previous committed checkpoint. **Confidence: H.**

14. **The migration misses existing public contracts and loop behavior.**  
    **Evidence:** `CUDAPrefetcher` exposes `next/reset`, not iteration ([prefetch_dataloader.py:104](/home/korvin/apps/neosr-dev/neosr/data/prefetch_dataloader.py:104)); the proposed inner-only `break` permits further epochs. `test.py` passes `current_iter=` ([test.py:55](/home/korvin/apps/neosr-dev/test.py:55)); the proposed signature renames it. Logger still divides updates by accumulation ([logger.py:143](/home/korvin/apps/neosr-dev/neosr/utils/logger.py:143)). Deleting `check_resume` leaves an import in [utils/__init__.py:25](/home/korvin/apps/neosr-dev/neosr/utils/__init__.py:25).  
    **Fix:** Include these callers/files explicitly; terminate both loops and handle already-completed resumes before consuming data. **Confidence: H.**

15. **Atomic-save failure handling is incomplete.**  
    **Evidence:** Injected mid-write failure into `torch.save` produced `RuntimeError: unexpected pos …`, bypassing the `OSError` retry/cleanup. File fsync plus rename also omits parent-directory fsync for crash durability.  
    **Fix:** Clean up temporary files on all serialization failures; retry only classified transient failures, preserve the cause, and fsync the containing directory where supported. **Confidence: H.**

16. **Pretrained compatibility and migration expectations overreach.**  
    **Evidence:** Unconditional `mmap=True` rejects old non-ZIP Torch checkpoints; installed Torch checks this explicitly at `torch/serialization.py:1537`. OTF’s proposed `(p/scale)` shape test contradicts the existing **LR** patch-size contract ([transforms.py:117](/home/korvin/apps/neosr-dev/neosr/data/transforms.py:117)). Storage accounting is also wrong: export G + live G + EMA G adds **two** G copies over the old layout.  
    **Fix:** Retry legacy files without mmap; preserve LR patch semantics; correct retention estimates. **Confidence: H.**

**(2) Correct but fragile—keep explicit tests**

- Bare-module loading with default `assign=False` preserves parameter identity; the installed `OptimizedModule` registers `_orig_mod` directly. The actual compile probe was blocked because Torch’s import attempts temporary-file creation in this read-only environment; compiled execution remains unverified.
- Python RNG tuples, Torch RNG tensors, and NumPy keys stored as `torch.uint32` passed the proposed weights-only round-trip.
- Disabled GradScaler state is indeed `{}`; test disabled→disabled and define precision-change behavior.
- `AveragedModel` first-update copying is correct: probe outputs were **10**, then **11** at decay `0.9`. The active warmup skeleton does not preserve this behavior. Test first-copy semantics, constant decay, buffers, and resumed averaging. [PyTorch 2.9 documentation](https://docs.pytorch.org/docs/2.9/generated/torch.optim.swa_utils.AveragedModel.html).
- The supplied legacy `3236.state` and both associated weight files load safely; the state reports epoch **4**, iteration **3236**. Test its adapter numerically, while clearly describing legacy continuation as lossy.

**(3) Simpler equivalents**

- Keep `AveragedModel` around the bare final structure; export `ema.module.state_dict()` and persist `n_averaged`. This removes prefix surgery without implementing another averaging engine.
- Retain existing prefetcher and logger interfaces; change counter ownership at their boundaries.
- Separate ordinary checkpoint recovery from exact replay requirements. Deliver the former honestly before promising the latter.

Codex session ID: 01a15aec-0dec-7ab1-a3c8-b80cd351aaf4
Resume in Codex: codex resume 01a15aec-0dec-7ab1-a3c8-b80cd351aaf4

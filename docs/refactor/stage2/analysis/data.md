# Stage 2 analysis: data pipeline, OTF degradation, dataset tooling

Lane: `neosr/data/*`, `neosr/models/otf.py`, `neosr/utils/{diffjpeg,img_util,color_util,lmdb_util,misc}.py`,
`neosr/data/file_client.py` (the brief lists `neosr/utils/file_client.py`; the module lives in `neosr/data/`),
`create_lmdb.py`, `tests/test_data.py`, `tests/test_otf_degradation.py`; competitor `traiNNer/data/`,
`traiNNer/ops/batchaug.py`, `traiNNer/models/realesrgan_model.py`.

Branch `refactor/stabilize` at f360edd. CPU only (`CUDA_VISIBLE_DEVICES=`). Probes and raw outputs:
`/tmp/claude-1000/-home-korvin-apps-neosr-dev/c1b40b02-f454-4498-9d89-be0afa4e2973/scratchpad/data/p*.py|out`.

**Measurement caveat.** Other lanes were running on the same 32-thread Ryzen 9 5950X while I measured, with load
average between 30 and 120. Single-process stage costs (p1, p2, p8) were taken at the lowest load and use CPU time
where possible. Wall-clock DataLoader throughput (p4, p17) is understated and noisy. Treat those numbers as lower
bounds and compare them only with each other. GPU timings are not measured; the GPU is reserved for the profiling
lane. Every GPU-side cost below is an estimate and is marked as a hypothesis.

Real data: `/home/korvin/apps/data/GT` holds 10607 PNGs of 256x256, 98% of them **16-bit** (some RGBA), 3.4 GB.
`LQ` holds 64x64 16-bit PNGs. The LQ is bicubic-like: PSNR(LQ, bicubic-antialias(GT)) = 51.2 dB (p3).

---

## 1. Summary (priority order)

| ID | P | Title | Evidence | Changes numerics? | Effort |
|---|---|---|---|---|---|
| D-01 | P1 | One unreadable or mis-sized training image kills a multi-day run | reproduced | no (valid data unchanged) | S |
| D-02 | P1 | The OTF canvas of at least 512 px causes 4x degrade work and 3x host traffic, and on 256-px tiles makes 75% of GT pixels mirrored padding | measured + simulated | yes (border statistics), A/B | S |
| D-03 | P1 | The template batch augmentation is applied to 73% of batches, because a hidden `multi_prob=0.3` inflates mixup/cutmix/resizemix 2.4x over the stated weights. It is enabled in all 84 templates | measured | yes, A/B | S |
| D-04 | P2 | resizemix downsizes GT and LQ without antialias, so the targets contain aliasing (29.6 dB at tao 0.2). The competitor fixed this | measured | yes (target fix) | S |
| D-05 | P2 | The OTF noise path makes 2 host syncs per batch (`bool(gray_noise.any())`) | counted (source-verified) | RNG stream only | S |
| D-06 | P2 | Large images (2K): the loader converts the full image to float before cropping (+29% per sample). The 4 default workers give about 5 batch8/s, and there is no tile tool (`extract_subimages.py` is referenced but missing) | measured | crop: no; tiles: sampling | S/M |
| D-07 | P2 | LMDB keys: `split('.png')` merges distinct images (F17). Names with spaces are unreadable. The writer uses ASCII and the reader UTF-8 | reproduced | no | S |
| D-08 | P2 | LMDB build: the whole dataset is held in RAM, 80 processes by default, worker errors are swallowed (then `KeyError`), partial output is left behind, and the `map_size` estimate fails with MapFullError (F18) | reproduced | no | M |
| D-09 | P2 | Hidden files crash recursive scans, hidden directories are traversed, and non-image files reach the decoder (F15) | reproduced | no | S |
| D-10 | P2 | The paired-folder order is unsorted, so DDP index mappings can differ across ranks (F16) | reproduced | sample order only | S |
| D-11 | P2 | Grayscale OTF (`color='y'`) crashes in DiffJPEG (F14) | reproduced | no | S |
| D-12 | P2 | Dataset mean/std normalisation is accepted but nothing inverts it (F13) | reproduced | no (reject) | S |
| D-13 | P2 | The log has no data-wait metric, so worker tuning (Codex perf rank 7) cannot be done from evidence | source-verified | no | S |
| D-14 | P3 | 16-bit PNGs pay the 16-bit decode cost and are then truncated to 8 bit (`>>8`) | measured | tiny (precision) | S |
| D-15 | P3 | PairPool stores fp32. uint8 storage is exact and 4x smaller (140 to 35 MiB at the template) | measured | no | S |
| D-16 | P3 | DiffJPEG zero-pads partial MCUs (libjpeg replicates). Bottom/right edge error is 5.84 vs 4.49 levels | measured | yes, tiny | S |
| D-17 | P3 | Worker policy: `'auto'` ignores CPU affinity and reserves no core for the main process, there are up to 2xW batches in shm+pinned, and cv2 threads are not limited in workers | source + measured (noisy) | no | S |
| D-18 | P3 | Path lists in workers grow copy-on-access by 358 B per entry per worker (341 MiB/worker at 1M images) | measured | no | S |
| D-19 | P3 | Hygiene: OTF kernel keys are unvalidated (TypeError inside a worker), and `np.seterr` is global at import | source-verified | no | S |
| D-20 | P3 | `cudnn.benchmark` with `benchmark_limit=0` autotunes about 500 distinct second-blur shapes in the Degrader | hypothesis | no | S |

Not bugs, keep as is: the Degrader's batched noise, with `_poisson_levels` replacing the per-sample
`torch.unique` loop. The competitor still pays 2xb host syncs per noise call there
(`traiNNer/data/degradations.py:791,802`). Also keep: resize modes without antialias (they are part of the
Real-ESRGAN degradation model), the Degrader running outside autocast in fp32 (keep it so: DCT coefficients reach
about 1000 and bf16 would destroy quantisation), and the indexed-swap PairPool, which has the same distribution as
Real-ESRGAN's shuffle-and-take-first-b.

**Requests to the profiling lane (GPU):** (1) `Degrader.__call__` ms for batch 8 at canvas 512 vs 288; (2) step time with and
without the two `bool(gray_noise.any())` syncs on a small model (e.g. compact/span, L1 only); (3) data-wait share with
the paired 256-px loader at 4 workers for the same small model; (4) whether `cudnn.flags(benchmark=False)` around the
Degrader changes its ms.

---

## 2. Codex verdicts

| Item | Verdict | Note |
|---|---|---|
| F13 | confirmed | `paired_dataset.py:125-126` and `single_dataset.py:77` normalise. Nothing inverts it (grep: no mean/std use in models/metrics/losses). With mean=std=0.5 on val GT01, values span [-1,1] and 22.6% of pixels sit below the `clamp(1/255,1)` floor (`image.py:224`). The OTF dataset ignores mean/std silently. No template uses it, so **reject** it. |
| F14 | confirmed | A full Degrader on (2,1,64,64) fails with `RuntimeError: shape '[2, 80, 80, 1]' is invalid`. With `jpeg_compress=false` it works (p6). |
| F15 | confirmed and extended | `.DS_Store` gives NotADirectoryError. Hidden **directories** are also traversed (`.thumbs/c.png` was returned). `Thumbs.db` (not hidden) enters the OTF and single path lists and fails decode (p7). |
| F16 | confirmed | `data_util.py:201` keeps scandir order. In the probe the order came out `b.png, a.png`. otf, single and lmdb sort; meta_info follows file order. |
| F17 | confirmed and extended | 4 sources gave 3 LMDB entries, and meta_info lists `img` twice. Also `my photo.png` becomes the key `my` on read (`lmdb_keys` takes the first whitespace token), which raises "No data returned" (p10). |
| F18 | confirmed | A corrupt image under the Pool path is swallowed by `apply_async`, then `KeyError('bad')` is raised after reading everything, and `data.mdb, lock.mdb, meta_info.txt` stay behind, so the next run gets FileExistsError. The `map_size` estimate (first image x N x 10) gave MDB_MAP_FULL on a small set. |
| PairPool memory | partial | The formula is right, but the pool rounds 180 down to **176**: measured 140.2 MiB, not 143.4 MiB. At LQ 256 it is 2.19 GiB. It is lazy, fixed and bounded (not a leak). uint8 storage is exact (D-15). |
| Perf rank 7 (workers) | partial | Confirmed: `'auto'` = `os.cpu_count()//world` (32 here). Workers already run with 1 torch thread, so BLAS oversubscription does not apply. cv2 threads do not matter on 256-px tiles (2.85 vs 2.77 ms/sample). With 8 workers on 2K images, cv2 threads=1 measured 15-34% faster, under heavy contention. The real limits are decode and full-image float on large images, and the missing data-wait metric (D-06, D-13). |

---

## 3. Measurements

### 3.1 Paired loader, per sample (template: LQ 64 / GT 256, x4), single process, warm cache (p1)

| Stage | ms | share |
|---|---|---|
| read bytes (GT+LQ) | 0.10 | 3% |
| `cv2.imdecode` IMREAD_COLOR (GT 16-bit 256 + LQ 64) | 2.19 | 77% |
| `astype(float32)/255` | 0.24 | 8% |
| crop + flip/rot | 0.06 | 2% |
| `img2tensor` (cvtColor + from_numpy) | 0.14 | 5% |
| collate (per sample; 0.96 ms per batch of 8) | 0.12 | 4% |
| **total** | **2.85** | 351 img/s per busy worker, i.e. 44 batch8/s or 22 batch16/s per worker |

Under the load of the audit, `thread_time` inside real workers was 3.56 ms/sample (281 img/s) (p5). The DataLoader
reached 52.8 batch8/s at 4 workers, 136/s at 8 and 210/s at 16. Main-process CPU was 6-18%, so IPC, collate and
the queue are cheap. cv2 thread setting made no difference on 256-px tiles.

**Updates/s the loader can sustain** (256-px 16-bit tiles, idle machine): about 175 batch8/s or 88 batch16/s with
the default 4 workers. That is enough for any model on this data. With 8-bit PNG tiles decode drops from 2.36 to
1.44 ms (p8), which gives about 1.5x more.

### 3.2 OTF loader (p2)
`__getitem__` takes 3.2-4.2 ms/sample. The breakdown: decode 2.2 ms, `copyMakeBorder` to 512 0.33 ms,
`cvtColor` 512² 0.11 ms, and kernels 0.1-0.2 ms. Kernel generation is cheap (0.07-0.09 ms per kernel). The one
exception is `skewed` at 1.1 ms (scipy `multivariate_normal.cdf`), which the templates do not use. Collate costs
6.98 ms per batch8, because the GT batch is 24 MiB. That is about 30 batch8/s per worker on the CPU side.

### 3.3 Large images (DIV2K-like 2040x1352 PNG made of real tiles, p9/p17)
`decode only` takes 73 ms and the current path 94.6 ms/sample: float32 conversion of the full image plus a crop of
only 256 px is +29%. Cropping first, then converting, takes 66-73 ms. In a real DataLoader with 8 workers this gave
3.8-5.6 batch8/s under load. On an idle machine with the default 4 workers it is about 5 batch8/s, which starves
almost any model.

### 3.4 Augmentation statistics at the template probabilities (p3, 4000 draws)
`augmentation = [none, mixup, cutmix, resizemix, cutblur]`, `aug_prob = [0.5, .1, .1, .1, .5]`; the hidden
default is `multi_prob=0.3` (`augmentations.py:189,225-229`; `image.py:209` does not pass it).

| aug | stated share (weights) | actual share of batches | LQ pixels changed when applied |
|---|---|---|---|
| any mixing aug | 61.5% | **73.3%** | |
| mixup | 7.7% | **18.9%** | 91.9% |
| cutmix | 7.7% | **17.8%** | 29.6% |
| resizemix | 7.7% | **18.0%** | 22.7% |
| cutblur | 38.5% | 52.9% | 16.2% |

Why: the multi branch picks 2-4 augmentations *without replacement* (`rng.choice(..., replace=False, p=w)`). After
`none` and `cutblur` are taken, the small weights are renormalised upward. Only 26.7% of batches are untouched
pairs. Mixup attenuates LQ noise by 0.71x on average (`sqrt(l²+(1-l)²)`, l ~ U(0.4, 0.6)) and blends two JPEG grids.
Neosr's cutblur works at native resolution: it pastes antialiased-bicubic(GT), that is a *clean* LR, into LQ. On
OTF data that makes 0.529 x 0.162 = **8.6% of all LQ training pixels degradation-free**. On the owner's bicubic
paired data it is nearly a no-op (51 dB). This is not the published CutBlur, which pastes HR into upsampled LR at
HR size.

resizemix aliasing (GT downscaled by tao, no antialias vs antialias):
tao 0.2: 29.6 dB · 0.35: 32.6 · 0.5: 37.4 · 0.7: 41.7 · 0.9: 49.4 dB.

### 3.5 OTF canvas (p12, p6)
Real tiles are 256x256 and the GT patch is 256 (LQ 64, x4):

| canvas | padded GT pixels | patches with mirror seam | degrade pixels vs GT patch | GT batch8 H2D |
|---|---|---|---|---|
| 512 (current) | **75.0%** | 96.9% | 4.00x | 24.0 MiB |
| 320 | 23.5% | 99.6% | 1.56x | 9.4 MiB |
| 288 (= patch + 32, competitor) | 12.1% | 98.8% | 1.27x | 7.6 MiB |

The CPU proxy of the Degrader scales with pixels: 1579 ms at 512 and 653 ms at 320 per batch8 on 8 threads,
noisy. CPU is dominated by `aten::poisson` and the DCT `mm`, and GPU cost will be dominated by different ops.

### 3.6 Memory
- Worker RSS is about 450 MiB, mostly shared. PSS is 40-116 MiB and private-dirty 16 MiB (paired) to 38 MiB (OTF)
  per worker (p5).
- The path list costs 358 B/entry as dict+str: 3.6 MiB for 10.6k images, **341 MiB per worker at 1M images**
  once touched (copy-on-access). As numpy `S46` bytes it would be 46 B/entry, shared and never written (p14).
- The PairPool is 140.2 MiB fp32 at the template and 35.1 MiB as uint8 (p6).
- In-flight batches are about `2·W` (prefetch_factor 2). For OTF at 512 that is 24 MiB each: 4 workers come to
  about 0.2 GiB shm plus pinned memory, `'auto'` (32) to about 1.5 GiB each. These are estimates from the source;
  pinned blocks are cached by the host allocator.

### 3.7 Host syncs per OTF batch (p15)
I counted Tensor→Python conversions around Degrader + PairPool + apply_augment: exactly **2 per batch**, at
`degradations.py:642` (gaussian) and `:720` (poisson); one per stage. PairPool `randperm`/indexing and
`apply_augment` have none.

---

## 4. Findings

### D-01 (P1, reliability): one bad image kills training
**Evidence (reproduced, p7).** A truncated PNG raises `OSError: Failed to decode image: …/b.png`, and a zero-byte
file raises `OSError: No data returned`. `paired_random_crop` raises `ValueError` for a scale mismatch or a
too-small image. All of these propagate worker → DataLoader → `train()` → "training failed", exit 1.
`tests/test_data.py:557` (`test_unreadable_image_in_worker`) asserts exactly this. Under random order a single bad
file in a 100k-image set is hit within one epoch. After an auto-resume it is hit again, and each crash loses up
to `save_checkpoint_freq` updates. The pre-refactor code had a broken retry (`random.randint(0, len)`
off-by-one, retried only on OSError, which the decoder never raised), and the refactor removed it. The competitor
wraps the *prefetcher* in broad `except Exception` retries (`traiNNer/data/prefetch_dataloader.py:120-200`). Do
not copy that.

**Fix.** In the training datasets only (`paired_dataset.py:86`, `otf_dataset.py:132`), move the body to
`_load(index)` and add:
```python
def __getitem__(self, index):
    for _ in range(10):
        try:
            return self._load(index)
        except (OSError, ValueError) as e:   # unreadable/corrupt file or mis-sized pair: a data defect
            if self.opt["phase"] != "train":
                raise
            get_root_logger().warning(f"skipping training sample: {e}")
            index = random.randrange(len(self))   # worker-seeded, reproducible
    msg = "10 consecutive unusable training samples; check the dataset"
    raise RuntimeError(msg) from e
```
Validation and test keep raising. The bound of 10 consecutive failures still stops a run whose dataroot is wrong.
This does not damp learning: valid data is untouched. Optionally keep a worker-local set of bad indices so the
warning is printed once.
**Numerics/compat:** none for valid data. It is a contract change, so update `test_unreadable_image_in_worker`.
**Test:** a train loader with one corrupt LQ yields full batches and logs one warning; the same file in phase
`val` still raises OSError; 10 corrupt files in a row raise RuntimeError.

### D-02 (P1, speed and OTF data quality): the canvas is at least 512
**Evidence.** `otf_crop_size` returns `ceil(max(512, patch·scale)/scale)·scale` (`otf_dataset.py:24-31`). For
LQ 64 at x4, the dataset pads or crops every GT to 512², the Degrader processes 512², and only then
`otf.feed_data` crops 256² (`models/otf.py:30-32`). That is 4.00x the pixels of the patch in stage 1 (blur,
resize, noise, JPEG), 24 MiB per batch through collate (7 ms), pinning and H2D, and about 3x the transient VRAM.
On tile datasets the cost is also quality: with the owner's 256² GT, padding happens at the bottom and right with
`BORDER_REFLECT_101`, so **75% of the training GT pixels are mirror images and 96.9% of patches contain a
mirror seam** (p12). The model is trained on symmetric seams that never occur in real images; the effect on quality is a hypothesis for the A/B. The 512 floor was carried
over from an older hard-coded constant (Real-ESRGAN used 400). Competitor: `crop_pad_size = gt_size + 32`
(`traiNNer/data/realesrgan_dataset.py:139`).

**Fix.** `return ceil((patch·scale + 32)/scale)·scale` (288 at x4/64). Keep reflect padding; the competitor pads
with black via `embed`, which is worse. Optionally apply `basic_augment` after the pad/crop (see D-06).
**Expected:** stage-1 Degrader work is 0.32x of today; H2D, pin and collate go from 24 to 7.6 MiB per batch;
mirrored pixels on 256 tiles go from 75% to 12%. Degrader ms is a GPU hypothesis (estimated 3-5 ms down to about
1-1.5 ms per batch8 on a 4090); ask the profiling lane.
**Numerics:** interior degradation statistics do not change, because every op is local or scale-relative. The
share of patches near a canvas border rises, and so does the share of real image content. This needs a short
A/B: same seed, about 20-40k updates, validation PSNR/DISTS and visual inspection of LQ samples.
**Compat:** none (no checkpoint impact); update `test_otf_crop_size`.
**Variant for tile datasets:** canvas = patch exactly (margin 0) removes padding entirely when images are at
least the patch size. Mention it in the docs; do not make it the default.

### D-03 (P1, training efficiency): the effective augmentation rate differs from the stated weights
**Evidence (measured, §3.4).** Every training template sets `aug_prob = [0.5, 0.1, 0.1, 0.1, 0.5]` (84 of 84). The
hidden `multi_prob=0.3` cannot be configured (`image.py:209-211`) and turns the stated 7.7% per mixing
augmentation into 18-19%. Only 26.7% of batches are clean, and 19% of batches are mixup ghost images (both GT
and LQ, so a GAN discriminator learns that ghosted composites are "real"). Neosr's cutblur makes 8.6% of OTF LQ
pixels degradation-free. The competitor ships the same family off by default (`use_moa: false` in all its
templates) and draws exactly one augmentation per batch. This is the kind of hidden regulariser the owner
dislikes: it lowers the share of real training signal per update.

**Fix (two steps).**
1. Contract fix (S): make the stated weights mean what they say. Either default `multi_prob=0` or read
   `datasets.train.aug_multi_prob` (default 0.3 for compatibility) and pass it through `image.py:209`. Add it to
   the schema (`config.py` `datasets.*`). That is one key and one line.
2. A/B on the templates (needs evidence before changing 84 files). Arms: (a) current, (b) `multi_prob=0`,
   (c) augmentation off. Run paired and OTF with the same seed and 40-80k updates, and compare validation
   PSNR/SSIM/DISTS curves. My expectation, a hypothesis: (b) or (c) learns faster early on. For OTF, also try
   (d) dropping cutblur.
**Numerics:** yes, A/B required. **Compat:** no checkpoint impact; a new optional config key.
**Test:** `apply_augment` with `multi_prob=0` hits each augmentation with frequency equal to its weight
(chi-square over 4000 draws).

### D-04 (P2, correct targets): resizemix aliasing
**Evidence.** `augmentations.py:125-130` uses `F.interpolate(..., mode="bicubic")` without `antialias` for
downscales of 0.2-0.9x. At tao=0.2 the GT target is only 29.6 dB away from the antialiased version (p3), so the
model is trained to output aliasing inside the box. The competitor fixed this
(`traiNNer/ops/batchaug.py:290-300`: `antialias=True`, and scope (0.5, 0.9)). It applies to 18% of template
batches.
**Fix:** `antialias=True` on both calls; antialias is a no-op for upscales. Consider raising the scope floor from
0.2 to 0.5, as the competitor does; that needs the D-03 A/B.
**Numerics:** yes (cleaner targets), low risk, and it can ride along with the D-03 A/B.
**Test:** downscale a high-frequency checkerboard and assert that the box content has a low maximum deviation from
the box mean.

### D-05 (P2, speed): two host syncs per OTF batch
**Evidence.** `generate_gaussian_noise_pt` and `generate_poisson_noise_pt` compute
`bool(gray_noise.any())` on a CUDA mask (`degradations.py:642,720`). It runs once per stage, so twice per batch
(p15). `feed_data` runs right after the previous update was *queued*, so the first sync waits for the whole
previous forward, backward and optimizer step. The GPU then idles while the CPU dispatches the degrader and the
next forward. Estimated impact: about 1-3 ms of GPU idle per step. That is 5-15% on small, launch-bound models
and about 1% on the template realplksr with DINOv2 (hypothesis; profiling lane request 2).
**Fix.** Draw the per-sample gray mask on the CPU, `m = torch.rand(b) < gray_prob`; take `cal = bool(m.any())`,
which needs no device; then move the mask with `m.to(img.dtype).pin_memory().to(img.device, non_blocking=True)`.
The pinned copy avoids the implicit sync of a pageable H2D. Simpler alternative: always compute the gray field
(+1/3 of the gaussian noise work, about +40% of the poisson noise work, no branch).
**Numerics:** same distribution, different RNG stream (not bit-identical to old runs).
`test_equals_reference_on_three_batches` uses the same functions, so it stays green.
**Test:** on CUDA, `torch.cuda.set_sync_debug_mode("error")` around `Degrader.__call__` raises nothing.

### D-06 (P2, speed on large images)
**Evidence.** `read_img → imfrombytes(float32=True)` (`data_util.py:35`, `img_util.py:184-186`) converts the
*whole* decoded image to float32 before `paired_random_crop`. For a 2K image that is a 48 MiB temporary and +29%
per sample (94.6 vs 73 ms, p9). The OTF path also flips and transposes the full float image before its crop
(`otf_dataset.py:145`). Decode itself dominates: 73 ms for a 4.6 MiB 2K PNG. The 4 default workers give about
5 batch8/s. `create_lmdb.py:29` tells users to run `extract_subimages.py`, which does not exist in the repo; the
competitor ships one.
**Fix.**
(a) Decode to uint8/uint16 (`read_img(..., float32=False)`), then crop, flip and transpose in the integer domain,
then convert only the patch, scaling by the dtype maximum. Validation still converts the full image. Expected
gain: −25-30% CPU per sample on 2K data, none on 256-px tiles.
(b) Ship a small `extract_subimages.py`: tiles of e.g. 512 with step 384, `Pool.imap` with bounded chunks, and
errors raised with the path. Per-sample decode cost scales with tile area: a 2K image at 73 ms becomes a
512² tile at about 7 ms, roughly 10x loader throughput.
(c) Document worker sizing together with D-13.
**Numerics:** (a) none (crop then convert gives an identical array); (b) the crop distribution becomes
tile-restricted, which is standard practice.
**Test (a):** same seed and same sample, old and new `__getitem__` outputs match exactly.

### D-07 (P2, tooling correctness): LMDB keys
**Evidence (p10).** `create_lmdb.py:22` (`split(".png")[0]`) maps `img.png.original.png` and `img.png.png` to
the same key `img`: 4 sources, 3 entries, and duplicate meta lines (F17). `lmdb_keys`
(`data_util.py:46`, `line.split()[0]`) and `paired_paths_from_meta_info_file` (`:157`, `split(" ")[0]`) truncate
names at the first space, so `my photo` is read as `my` and training crashes. The writer encodes keys as ASCII
(`lmdb_util.py:155,213`) while the reader uses UTF-8 (`file_client.py:101`); non-ASCII names raise
UnicodeEncodeError on write. meta_info is written with `encoding="locale"` and read as UTF-8 (cp1252 on Windows).
Only `suffix="png"` is collected: `.PNG`, `.jpg` and `.webp` are skipped silently.
**Fix.** keys = `Path(p).with_suffix("").as_posix()`; reject duplicates before opening the env; collect the shared
`IMG_EXTENSIONS` case-insensitively; encode and write meta in UTF-8. Parse meta lines as
`re.sub(r" \(\d+,\d+,\d+\)( \d+)?$", "", line.rstrip("\n"))` in both readers. That works for old files and names
with spaces.
**Compat:** existing LMDBs keep reading (same keys for space-free names).
**Test:** round trip with `a.b.png`, `a.png`, `my photo.png` and `ü.png`; reject the duplicate set.

### D-08 (P2, tooling memory and reliability): LMDB build (F18)
**Evidence (p10).** The default is `n_thread=80` (`lmdb_util.py:45`), with no CLI option, on a 32-thread
machine. Every encoded image is collected in a dict before writing, so RAM equals the encoded dataset size.
`apply_async` without `error_callback` swallows the exception from a corrupt file; the job then ends in
`KeyError('bad')` *after* reading everything. The partial `*.lmdb` stays and blocks the rerun (FileExistsError).
`map_size = first image × N × 10` (`:144`) raised MDB_MAP_FULL on a small set, and a small first image breaks
heterogeneous sets.
**Fix (about 60 lines).** Write to `out.lmdb.tmp` and rename on success; delete the tmp directory on any
exception. Use `with Pool(n) as p: for key, buf, shape in p.imap(encode, items, chunksize=16)` (ordered, errors
re-raised with the path) and write incrementally, committing every N images. Set `n = os.process_cpu_count()` and
add `--workers`. Start `map_size` at 2x the sum of source file sizes, and on `lmdb.MapFullError` commit, call
`env.set_mapsize(2*size)` and retry. Use `txn.put(..., overwrite=False)` as a second duplicate guard.
**Test:** a corrupt file fails with its path and leaves no output; 1000 small images build with the initial
map_size forced tiny.

### D-09 (P2, reliability): F15
**Evidence (p7).** `misc.py:102-107`: a hidden file falls into `elif recursive` and raises NotADirectoryError;
hidden directories are recursed into; `Thumbs.db` and `desktop.ini` are not hidden and enter the OTF and single
lists, then fail decode at run time.
**Fix:** `if entry.name.startswith('.'): continue`; `if entry.is_file(): …`; `elif recursive and
entry.is_dir(follow_symlinks=False): …`. Filter otf, single and paired with one module constant `IMG_EXTENSIONS`,
case-insensitive. With D-01, a leftover bad file can no longer kill a run either.
**Test:** a tree with `.DS_Store`, `.thumbs/x.png` and `Thumbs.db` yields only the real images.

### D-10 (P2, DDP correctness): F16
**Fix:** `gt_paths = sorted(...)` at `data_util.py:201`. Consider stem matching as the competitor does
(`traiNNer/data/data_util.py:251-275`, LQ `.jpg` paired with GT `.png`) as a later feature.
**Numerics:** sample order for a given seed changes once. **Test:** a monkeypatched reversed `os.scandir` gives
identical pairs.

### D-11 (P2): F14 grayscale OTF
**Fix (minimal):** in `models/otf.py.__init__`, `if cfg.get("color") == "y" and self.degrader.jpeg: raise
ConfigError(...)`.
**Fix (proper, about 8 lines, if grayscale OTF is wanted):** the OTF dataset always loads RGB, and after
`degrader()` the trainer converts gt and lq with `rgb_to_grayscale` and re-quantises lq to 8 bit. That gives
real-world chroma-JPEG artifacts in luma.
**Test:** the config is rejected (or a gray batch comes out with 1 channel).

### D-12 (P2): F13 normalisation
**Fix:** raise ConfigError in `validate_train` / dataset `__init__` when `mean` or `std` is set and differs from
identity. This also covers half-specified pairs, which crash in `normalize(img, mean, None)`. An end-to-end
normalised domain would touch every loss and metric; it is not worth it with no user.
**Test:** a config with mean/std is rejected with a clear message.

### D-13 (P2, tooling): data-wait metric
**Evidence.** `train.py:228` (`while (batch := prefetcher.next())`): the loop does not time the wait.
`AvgTimer`/`MessageLogger` report only the total it/s. Codex rank 7 asks for "batch wait" evidence, and today
there is none.
**Fix:** `t = perf_counter()` before `prefetcher.next()`; accumulate the time in a window; log
`data: 12% (3.1 ms)` next to it/s. Under DDP, rank 0 only is fine. The cost is negligible.
**Test:** a fake slow loader shows a data share near 100%.

### D-14 (P3, speed and precision): 16-bit PNGs
**Evidence.** IMREAD_COLOR truncates 16 to 8 bit (`>>8` matches 100%, p1). Decode costs the same as a 16-bit read
(2.17 vs 2.19 ms) and is 1.65x an 8-bit PNG (2.36 vs 1.44 ms, p8). The owner's dataset therefore pays about
0.9 ms per sample (about 30% of loader CPU) for precision it throws away. The competitor keeps 16 bit
(`tensor2float32`, `/iinfo.max`).
**Fix:** `cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH` (verified: 3-channel uint16, also on RGBA sources, p18) and scale
by the dtype maximum, which pairs naturally with D-06(a). Or document "convert to 8-bit for speed".
**Numerics:** at most 1 level difference (mean 0.03 levels); OTF output is 8-bit anyway.

### D-15 (P3, VRAM): PairPool as uint8
Both pool members are exact multiples of 1/255: the LQ after `Degrader` quantisation, the GT from an 8-bit decode
plus pad and flip. Store `(x*255).round().to(torch.uint8)` and return `.float()/255`. That gives 140 → 35 MiB at
the template and 561 → 140 MiB at LQ 128, bit-exact. With D-14, GT may be 16-bit; then store GT as fp16 or uint16.
Log the pool size at startup, as Codex suggests.

### D-16 (P3, OTF realism): DiffJPEG padding
`diffjpeg.py:533` pads to a multiple of 16 with zeros; libjpeg replicates edge pixels. Over the last 10 rows the
error is 5.84 levels vs 4.49 in the interior, and the distance to cv2 is 4.91 vs 2.43 with replicate (p11).
**Fix:** `mode="replicate"`. Tiny numerics change; bundle it with D-02, because a smaller canvas makes border
pixels relatively more frequent.

### D-17 (P3): worker policy and thread hygiene
- `'auto'` (`data/__init__.py:55`): use `os.process_cpu_count()` (affinity-aware, Python 3.13), subtract 1 for the
  main process, and log the result.
- `cv2.setNumThreads(1)` in `worker_init_fn`. It does nothing on tiles. On 2K data with 8 workers it measured
  +15-34% under contention (p17), so it is a hypothesis on an idle machine. It is standard practice (mmcv).
- Up to `2·W` batches are in flight in shm and pinned memory (§3.6); for OTF D-02 cuts each by 3x.
- Validation runs in the main process (`workers=0`). For large validation sets, 1-2 workers would overlap decode
  with inference (minor).

### D-18 (P3, host RAM): path lists
For datasets of 1M or more images, store paths as numpy bytes arrays (`np.array([p.encode() for p in paths])`,
decoded at access) in paired, otf and single. Paired needs two arrays. That avoids 341 MiB per worker of
copy-on-access growth (p14). No impact at 10k images.

### D-19 (P3, hygiene)
- `otf_dataset.py:174,198,221` compare `rng.uniform() < self.opt.get("sinc_prob", None)`, so a missing key raises a
  TypeError inside a worker. Validate `kernel_list/kernel_prob/blur_sigma/sinc_prob*/final_sinc_prob` in
  `otf.__init__`, as the Degrader already does for its keys.
- `degradations.py:526`: `np.seterr(divide="ignore", invalid="ignore")` at import silences numpy warnings for the
  whole process. Move it into `circular_lowpass_kernel` as `with np.errstate(...)`.
- `kernel_range` is hard-coded (TODO at `otf_dataset.py:121`); the competitor makes it configurable. Feature only.

### D-20 (P3, hypothesis): cudnn autotune storms in the Degrader
`runtime.py:91-93` sets `benchmark=True, benchmark_limit=0` (exhaustive). The second blur runs on
`int(512·s1)²`, so there are about 500 distinct shapes. Each first occurrence autotunes every algorithm, roughly
500 x 10-40 ms over the first thousands of updates, with workspace spikes. Candidate fix: `with
torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=cudnn.deterministic, allow_tf32=cudnn.allow_tf32):`
around `Degrader.__call__`. Note that `flags()` defaults to `enabled=False`, so every argument must be passed.
Measure first (profiling lane request 4).

---

## 5. Don't do

- No adaptive damping: no automatic loss normalisation, gradient clipping driven by data statistics, or dynamic
  batch/patch. Every item above improves throughput or makes the training signal correct; none of them reduces
  the learning signal.
- Do not adopt the competitor's prefetcher retry loops (broad `except Exception`, iterator resets). D-01 handles
  data defects where they occur, with bounded retries and a clear error.
- Do not port pyvips just for crops. For large images, pre-tiling (D-06b) gives more for less.

## 6. Probe index (scratch dir)
p1 stage costs · p2 OTF getitem + kernel costs · p3 augmentation stats/aliasing/LQ-vs-bicubic · p4 DataLoader
throughput · p5 worker memory/CPU · p6 Degrader CPU proxy + F14 + PairPool size · p7 F15/F16/corrupt files ·
p8 decode variants · p9/p17 2K images · p10 LMDB (F17/F18) · p11 DiffJPEG padding · p12 canvas padding sim ·
p13 F13 · p14 path-list memory · p15 host-sync count · p18 16-bit decode flags.

# Codex (gpt-6-astra, high) — NaN root-cause analysis (read-only, CPU probes)

Correction to the smoke report: the "fp16 NaN at iter 24" (smoke dr3.log:58-61) was a CUDA OOM after l_g_total=640.77,
of which 623.81 is the GAN loss; MS-SSIM and FDL were improving. The NaN check (image.py:717) inspects the forward G loss
BEFORE optimizer steps / EMA, which constrains iteration-1 explanations to forward computation on freshly initialised nets.

| rank | hypothesis | evidence | decisive experiment |
|---|---|---|---|
| 1 very high | MS-SSIM fractional powers of negative cs/ssim (ssim_loss.py:149); esrgan output unbounded (esrgan_arch.py:217), clamp disabled in config | CPU probe, random-init esrgan + fixed pair: 8/10 seeds NaN, all with a negative power base; happens in fp32 too | replay failing seed, log the five power bases; clamp_min(1e-6) before power -> finite loss and grads |
| 2 very high (huge logits) | metagan spectral norm never updated: D always eval() (image.py:609,654); legacy spectral_norm updates u,v only in train forwards | probe: one init gave max |logit| 25,172; one train-mode forward moved eval mean from -7,904 to -0.019. G GAN loss under inference_mode has no grad | compare eval D before/after train-mode SN updates; replay fp16 with D train forwards |
| 3 medium (later) | adan_sf warmup not applied: lr computed at adan_sf.py:260, base lr passed at :315 | warmup 1,600 vs 16,000 -> identical first three updates; first step moves z by full base lr | compare p/z deltas with corrected lr plumbing |
| 4 low (latent) | LDL var**0.2 without epsilon, differentiable weights (ldl_loss.py:92) | constant residual -> loss 0, NaN grads; random-init outputs finite | log min residual variance; epsilon before power |
| 5 low | FDL DINOv2: custom_fwd casts to fp32 (fdl_loss.py:317), FFT phase grads (:329) | fp16 log: FDL finite, decreasing 23 -> 17 | isolate phase term in fp32 |
| 6 low | consistency_loss clamps inputs to [1/255,1] first (consistency_loss.py:156) | fp32 probes finite | CUDA autocast vs disabled |
| 7 very low | GradScaler on bf16 (image.py:149; G init 32, D 65536), EMA | cannot explain NaN in an unscaled first forward | disable scaler for bf16 |

BCEWithLogits at logits 358 and 5,000 is finite; infinite logits give NaN.

Fixes: MS-SSIM clamp_min(1e-6) on each reduced cs/ssim before the power (fp32); D optimisation forwards in train mode, SN estimates
initialised before the first G adversarial forward, no inference_mode around D(sr) (freeze D params instead); adan_sf pass the
scheduled lr to the kernel; LDL variance in fp32 floored before pow(0.2); isfinite checks on every loss before backward and on grads
before stepping; GradScaler only for fp16. Missing manual_seed -> new random seed each start (options.py:193), consistent with
init-dependent failures.

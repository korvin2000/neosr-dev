# Codex Astra (gpt-6-astra, high, read-only) independent R3GAN review, 2026-10-28

Verbatim output relayed by the codex-rescue forwarder (thread 01a182cd-e205-7c12-ac06-57a06f1ec41b). Part 1 = original review, Part 2 = efficiency addendum. CPU-only probes, no GPU timings.

Codex (gpt-6-astra, high effort, read-only) answered the original request and then the efficiency addendum. The addendum went to the same thread (01a182cd-e205-7c12-ac06-57a06f1ec41b) via `--resume-last`. Both answers are below, verbatim, without the `[codex]` progress lines. Codex says its CPU probes ran on PyTorch 2.9.1 with no GPU, so there is no GPU, compile or DDP qualification and no measured timings.

=== PART 1: ORIGINAL REVIEW (Codex stdout) ===

**Ranked findings**

| Rank | Finding | Evidence | Confidence |
|---|---|---|---|
| 1 | **Use neosr's current loss infrastructure as the migration base.** Its logistic signs, detached fake samples, differentiable penalties, and update-based timing are substantially better than the experiments. | [gan_loss.py:85](/home/korvin/apps/neosr-dev/.worktrees/refactor/neosr/losses/gan_loss.py:85), [gan_loss.py:127](/home/korvin/apps/neosr-dev/.worktrees/refactor/neosr/losses/gan_loss.py:127) | High |
| 2 | **The proposed scalar discriminator currently breaks both neosr's penalty and the minimal loss.** It returns `[B]`; their reductions require additional dimensions. I reproduced both failures. | [discriminator:491](/home/korvin/apps/traiNNer-redux-dev/traiNNer/archs/rpgan_r3gan_scalar_discriminator_arch.py:491), [gan_loss.py:149](/home/korvin/apps/neosr-dev/.worktrees/refactor/neosr/losses/gan_loss.py:149), [minimal loss:65](/home/korvin/apps/r3gan/r3gan_loss.py:65) | High |
| 3 | **The "alternative" is not R3GAN, and its safety fallback is unacceptable.** It uses relativistic-average hinge with detached comparison means. Falling back to `create_graph=False` removes the penalty's parameter gradients; common second-derivative failures occur outside its handler anyway. | [alternative:233](/home/korvin/apps/r3gan/alternative/r3gan_loss.py:233), [alternative:70](/home/korvin/apps/r3gan/alternative/r3gan_loss.py:70) | High |
| 4 | **The old SR integrations are mutually incompatible snapshots.** `sr_model.py` also applies the minimal G loss weight twice; `sr_model2.py` calls nonexistent methods in the supplied losses; `sr_model3.py` does not implement paired D loss plus R1/R2. | [sr_model.py:457](/home/korvin/apps/r3gan/sr_model.py:457), [sr_model.py:491](/home/korvin/apps/r3gan/sr_model.py:491), [sr_model2.py:466](/home/korvin/apps/r3gan/sr_model2.py:466), [sr_model3.py:629](/home/korvin/apps/r3gan/sr_model3.py:629) | High |
| 5 | **The existing map penalty regularizes the mean logit, not every patch logit.** Local gradients can cancel completely. This is a legitimate surrogate only if explicitly intended; it is not automatically equivalent to scalar R3GAN. | [gan_loss.py:149](/home/korvin/apps/neosr-dev/.worktrees/refactor/neosr/losses/gan_loss.py:149), [gan_loss.py:162](/home/korvin/apps/neosr-dev/.worktrees/refactor/neosr/losses/gan_loss.py:162) | High |
| 6 | **Full trajectory-exact resume is not currently provided.** neosr explicitly restarts data order, augmentation, and the OTF pool, and restores model-side RNG only on rank 0. | [models/base.py:379](/home/korvin/apps/neosr-dev/.worktrees/refactor/neosr/models/base.py:379) | High |
| 7 | **Two requested "paper details" need correction.** The reference activation has forward gain **1**, not √2. Its Adam β₂ is scheduled **0.9→0.99**, rather than fixed at 0.99 throughout. | [reference FusedOperators.py:31](https://github.com/brownvc/R3GAN/blob/main/R3GAN/FusedOperators.py#L31), [reference train.py:228](https://github.com/brownvc/R3GAN/blob/main/train.py#L228) | High |

I made no file edits. I inspected the working tree, including its existing uncommitted changes; some trainer files changed during the review. Evidence below refers to the versions read. I ran isolated, in-memory CPU probes with PyTorch **2.9.1**. CUDA was unavailable, and normal package imports encountered the read-only environment's temporary-directory restriction, so this is **not** a completed GPU/compile/DDP qualification.

For compact references below:

- `N/` = `/home/korvin/apps/neosr-dev/.worktrees/refactor/`
- `O/` = `/home/korvin/apps/r3gan/`
- `T/` = `/home/korvin/apps/traiNNer-redux-dev/traiNNer/`
- `D:` = lines in `T/archs/rpgan_r3gan_scalar_discriminator_arch.py`
- Paper references use PDF pages; PDFs have no stable source-code line numbers.

**1. Audit of the owner's implementations**

The reference minimization objectives are:

L_G = E softplus(D(x_r) - D(x_f)),

L_D = E softplus(D(x_f) - D(x_r)) + (gamma1/2) E ||grad_{x_r} D(x_r)||^2 + (gamma2/2) E ||grad_{x_f} D(x_f)||^2.

Fake images are detached for D; R2 does not regularize G directly. The reference uses both penalties every D update. Evidence: [reference Trainer.py:14](https://github.com/brownvc/R3GAN/blob/main/R3GAN/Trainer.py#L14), [Trainer.py:28](https://github.com/brownvc/R3GAN/blob/main/R3GAN/Trainer.py#L28). **Confidence: high.**

| Implementation | What it actually implements | Audit and disposition |
|---|---|---|
| `O/r3gan_loss.py` | Pairwise logistic RpGAN, elementwise for map outputs; every-call R1/R2 of each image's **mean logit**. | Correct objective signs and `create_graph=True`. Salvage the mathematical kernel, not its state or integration. `:54–83`. **High.** |
| `O/v2/r3gan_loss.py` | Same loss and penalty algorithm. | The difference is **not just defaults**: `max_img` is used directly, whereas the first file multiplies it by 32. Schedule units differ. `:43–52` versus original `:43–52`. **High.** |
| `O/alternative/r3gan_loss.py` and its traiNNer copy | Relativistic-average hinge, with detached opposing means in D; generator hinge counterpart. | Not pairwise logistic R3GAN. Detached means also make its D gradient differ from the ordinary fully differentiated RaHinge objective. Reject as the migration base. `:233–239`, `:330–335`. **High.** |
| `O/sr_model.py` | Integration targeting the minimal loss's `generator_penalty`/`discriminator_penalty` API. | Closest historical integration, but double weighting, AMP, compile dispatch, and state problems make copying it counterproductive. `:456–459`, `:491–492`, `:548–555`. **High.** |
| `O/sr_model2.py` | Integration targeting a different, missing `generator_forward`/`discriminator_forward` API. | Cannot establish the intended mathematical implementation because neither supplied loss defines those methods. As supplied, it fails. `:466`, `:604`; supplied loss class definitions. **High.** |
| `O/sr_model3.py` | Missing generator API, separate real/fake D calls, plus lazy R1 only. | Not a complete R3GAN integration. With the supplied alternative loss, its positional D calls fail the required-keyword interface; with an ordinary GAN loss, it remains ordinary GAN + R1. `:496`, `:629–650`; alternative `:389–398`. **High.** |

The minimal loss has these concrete defects:

| Defect | Consequence | Evidence / confidence |
|---|---|---|
| Hardcoded `critics.mean([1,2,3])` | `[B]` and `[B,1]` scalar outputs fail. | `O/r3gan_loss.py:65`. **High; reproduced.** |
| Penalty shaped `[B,1,1]` added to `[B,1,H,W]` | Broadcasts to `[B,B,H,W]`. This wastes memory and mixes axes. Because the expression is additive and finally averaged, its scalar value still equals the intended mean penalty in this common case; this is not necessarily a changed gradient. | `:75–83`. **High; shape reproduced.** |
| Internal G weighting plus trainer weighting | Effective coefficient is w|w|, ordinarily w^2. For `w=0.01`, the GAN term becomes 100 times smaller than intended. | Loss `:60`; `O/sr_model.py:491`. **High.** |
| No FP32 penalty boundary | D forward, input gradients, and reductions can use mixed precision; fake input leaves can remain bf16. Scaling the final loss does not repair precision already lost in the inner gradient. | Loss `:69–83`; model `:548–552`. **High.** |
| Mutable Python image counter | Not in `state_dict`; advances per D call, including microbatches and subsequently skipped optimizer updates. Local batch size is counted, not global images. | Loss `:49–52`, `:79–82`. **High.** |
| Inconsistent image-count initialization | Original defaults initialize with `current_img*8` but use a horizon of `max_img*32`; v2 changes both conventions. Exact continuation depends on manual reconstruction. | Original and v2 `:43–52`. **High.** |
| No schedule validation | Invalid horizon/warmup combinations can divide by zero; negative/nonfinite settings are not rejected. | `:7–33`, `:38–52`. **High.** |
| Real G forward is not explicitly `no_grad` | Fine when D is frozen by this trainer; otherwise it unnecessarily builds D parameter gradients. Detaching HQ alone does not freeze D. | Loss `:55–57`; model `:402`. **High.** |
| Unprotected diagnostic ratio | `r1/r2` can produce NaN/inf when R2 is zero; `raw_loss` is actually a margin, not logistic loss. | Model `:560–561`; loss `:88`. **High.** |

The alternative's problems are more fundamental:

- **Its fallback is not a regularizer.** `create_graph=False` makes the returned gradient norm independent of D parameters for subsequent backpropagation. Additionally, `grid_sample` commonly permits construction of the gradient graph and fails only when differentiating that graph. My probe constructed a finite penalty, then failed during `penalty.backward()`, outside the handler. **Evidence:** `O/alternative/r3gan_loss.py:70–100`. **Confidence: high.**
- **The FP32 claim is incomplete.** The differentiable leaf is created before conversion to float. A bf16 input therefore receives bf16 input gradients; the result is explicitly cast back to the input dtype. Only the D forward is inside the disabled-autocast block, and the device type is hardcoded to CUDA. **Evidence:** `:52–57`, `:105–110`. **Confidence: high; bf16 penalty reproduced.**
- **The augmentation explanation is misleading.** Detaching already augmented images severs the upstream augmentation graph. An upstream `grid_sample` is therefore not traversed by this penalty; one inside D still is. Regularizing unaugmented images is an objective choice, not a universal double-backward fix. **Evidence:** `:52`, `:261–279`. **Confidence: high.**
- **Lazy scaling is algebraically present:** weights are multiplied by the interval, then halved in the total. But `_gp_step` counts loss calls rather than optimizer updates; gradient accumulation changes the cadence. The buffer is serializable in isolation, but the supplied trainer checkpoint code does not save loss-module state. **Evidence:** `:143`, `:241–247`, `:284–285`, `:310`; `T/models/base_model.py:738–769`. **Confidence: high.**
- **`gp_crop` changes the regularized function.** D is penalized on independently chosen smaller real/fake crops, potentially using different boundary effects, receptive-field coverage, and scalar pooling. It is not an unbiased implementation of the full-image penalty. Its GPU `.item()` calls also synchronize. **Evidence:** `:203–213`, `:287–305`. **Confidence: high for semantics; medium for practical impact.**
- **Several interfaces conceal omissions:** only the last output head participates; `MultiScaleR3GANLoss` rejects R3GAN; `current_iter` is accepted but ignored; `_skip_discriminator` returns zero when called internally but cannot be passed through the public forward's keyword filter. **Evidence:** `:228–231`, `:353–355`, `:405–430`, `:458–461`. **Confidence: high.**
- Its `loss_weight` is stored but not applied internally. That is compatible with an external-weighting trainer, but differs from the minimal loss's contract. **Evidence:** `:135`, `:377`, `:421–436`; `O/sr_model.py:491`. **Confidence: high.**

Additional integration failures:

| Location | Finding | Confidence |
|---|---|---|
| `O/sr_model.py:309`, `:553`; analogous code in model2/model3 | Compiling the loss wraps it in `OptimizedModule`; the subsequent `isinstance(..., R3GANLoss)` dispatch does not unwrap it. This can route execution into the wrong API branch. | High |
| `T/models/base_model.py:237–245` | Historical wrapping is DDP around compiled D. None of these loss implementations supplies neosr's eager penalty escape. Merely disabling autocast cannot enable compiled double backward. | High |
| `O/sr_model2.py:478–492`; model3 `:508–522` | Adaptive D skipping violates the stated policy. Decisions are based on local losses, creating a DDP divergence risk if ranks disagree. | High for policy; medium for distributed failure |
| `O/sr_model3.py:641–650` | R1 is correctly differentiated through `T/losses/gan_loss.py:184–187`, with `gamma=10`, interval 4 and scalarized maps. It has **no R2** and no eager compile escape. The extra forward is outside the shown autocast scope. | High |
| `/home/korvin/apps/aaa.diff:32`, `:89`, `:336` | Adds map averaging, lazy/cropped penalties, and retains the zero-loss escape. It does not repair the underlying hinge objective or fallback. Its interval default is 8 versus 4 in the supplied alternative. | High |
| `/home/korvin/apps/s3.diff:64–70`, `:127–131` | Updates D only on `apply_gradient`, then divides its sole microbatch loss by `accum_iters`: effective D batch and scaling differ from accumulated G training. Also weights ordinary D loss with the GAN weight. | High |

**Discriminator audit.** Its core architecture is useful: normalization-free residual blocks, correct L^{-1/4} initialization scaling, zero initialization of each residual branch's third convolution, and the reference FIR construction. **Evidence:** `D:112–154`, `D:240–264`. **Confidence: high.**

It is not a drop-in migration:

- `[B]` output triggers the loss bugs above. **Evidence:** `D:491`. **High.**
- The head **always** adaptively pools non-4x4 features, despite the `force_4x4=False` comment claiming strict reference behavior. Setting the flag true moves pooling before the final stage's blocks, so the flag is not merely redundant. **Evidence:** `D:301–307`, `D:395–397`, `D:480–483`. **High.**
- Default five-stage widths imply four downsamplings: 128/256 inputs reach 8x8/16x16, followed by this extra pooling. **Evidence:** `D:381`, `D:426`, `D:305`. **High.**
- Hardcoded `stage_dtype` conflicts with an external AMP policy. Setting it to bf16 defeats an FP32 penalty scope and can produce a bf16-input/FP32-weight error in the ordinary head layers. I reproduced that error. Default FP32 stage casts also interfere with a clean autocast-only residual dtype policy. **Evidence:** `D:296–309`, `D:353–354`, `D:388`, `D:473–475`. **High.**
- Prefer native PyTorch activation/resampling operations for the first port. Optional external fused implementations add an unqualified bf16/compile/double-backward dependency. The native fallback completed a CPU parameter-gradient penalty probe. **Evidence:** `D:52–58`, `D:76–86`, `D:146–154`. **Confidence: high for fallback correctness; medium for performance.**

**Cost per microbatch**, excluding G's own forward and reconstruction losses:

| Path | D forwards | Additional regularization work | Evidence / confidence |
|---|---|---|---|
| Minimal RpGAN | Two during G; two during D | Two input-gradient computations plus differentiation of both penalties, every call | `O/r3gan_loss.py:56–76`. **High.** |
| Alternative | Two during G; two during D; two additional cropped forwards every k calls | Two input-gradient computations and penalty backward on scheduled calls | Alternative `:296–305`, `:358–359`. **High.** |
| neosr RpGAN | Two during G; one concatenated `2B` D forward, or two with BN | One combined input-gradient call and penalty backward every k updates; regularized D forward is FP32 eager | `N/neosr/losses/gan_loss.py:89–91`, `:118–155`. **High.** |

Lazy penalties reduce average compute, **not the peak memory requirement of a regularized update**. Exact speed ratios require GPU measurement. **Evidence:** the retained derivative graph at `N/neosr/losses/gan_loss.py:150`; the scheduled full penalty path at `:130–155`. **Confidence: high.**

**2. neosr's current implementation**

| Area | Assessment | Evidence / confidence |
|---|---|---|
| Logistic signs | Correct: G minimizes `softplus(real-fake)`; D minimizes `softplus(fake-real)`. Subtraction and softplus use FP32. | `N/neosr/losses/gan_loss.py:92`, `:162`. **High.** |
| Pairing order | Correct indexwise pairing: concatenation followed by `chunk(2)` preserves correspondence. No batch-average comparator is introduced. | `:125`, `:162`. **High.** |
| Detachment and weighting | Correct: fake is detached for D; real is constant for G; only the G term receives its configured loss weight. | `:89`, `:129`; `N/neosr/losses/base.py:88`; GAN `:108`. **High.** |
| Scalar output | Broken on `[B]`; use a shape-safe per-image reduction such as `out.reshape(out.shape[0], -1).mean(1)`. Also reject incompatible real/fake shapes before subtraction. | GAN `:149`, `:162`. **High.** |
| Penalty construction | Correct for a batch-independent scalar critic: differentiate the **sum over samples**, square and sum input coordinates per sample, then average samples. No erroneous 1/B^2 factor. | `:149–155`. **High.** |
| Lazy scaling | Correct interval multiplier for adding the penalty to an existing D update. Do **not** automatically apply StyleGAN's separate-extra-step optimizer correction: this implementation adds no extra optimizer step. | `:155`; `N/neosr/models/image.py:369`. **High.** |
| Timing | `Context.it` is shared across accumulation microbatches and ranks. Penalties use the absolute attempted-update index, not D-active-update count; first activation need not be a penalty update. | GAN `:130`; image `:322`, `:380`. **High.** |
| FP32/eager escape | Correct for neosr's current wrapper order: `compile(DDP(bare_D))` becomes `DDP(bare_D)`, preserving the reducer. Disabling autocast and selecting math SDPA addresses the intended higher-order path. | GAN `:16–18`, `:139–150`; models/base `:157–173`. **High structurally; medium for untested GPU combinations.** |
| Wrapper limitations | One-level `_orig_mod` removal assumes the current wrapping order and no internally compiled submodules or forced low-precision layers. It is not a general "make arbitrary D eager and FP32" function. | GAN `:18`; models/base `:170`; owner `D:353`. **High.** |
| Defaults | `gan_type="rpgan"` alone is **unregularized RpGAN**: both gammas default to zero. No gamma schedule exists. | GAN `:46–54`, `:74`. **High.** |
| Validation | Bounds checking does not enforce finite gamma values or integer schedule/interval arguments. Validate those explicitly when adding schedules. | GAN `:66–68`. **High.** |

`backward(inputs=...)` is **not** a reason to remove `create_graph=True`. The inner call constructs the graph of grad_x D; the outer backward differentiates its squared norm into the selected D parameters. Selecting accumulation targets does not detach intermediate derivatives. My linear-critic probe produced the expected penalty parameter gradient, **0.5**. Evidence: GAN `:150`; `N/neosr/models/phase.py:166`; [PyTorch 2.9 backward documentation](https://docs.pytorch.org/docs/2.9/generated/torch.autograd.backward.html). **Confidence: high.**

The existing tests are useful but incomplete qualification: there is a CPU DDP regularized-RpGAN test and a compiled spectral-normalized-D test, but the latter uses BCE, and they do not establish the full CUDA bf16 + compile + DDP + scalar-D combination. Evidence: `N/tests/test_s2_losses.py:594–605`, `:611–648`. **Confidence: high.**

The map-output distinction matters mathematically. With M logits:

P_current = || grad_x (1/M) sum_m d_m(x) ||^2   !=   P_local = (1/M) sum_m ||grad_x d_m(x)||^2.

For logits wx, -wx, the current penalty is zero while the average individual-logit penalty is w^2; I reproduced this. Meanwhile, the adversarial term averages **softplus of each difference**, rather than applying softplus to the difference of mean scores. Thus the adversarial and regularized objects are different. Evidence: GAN `:149` versus `:162`. **Confidence: high; mathematical derivation.**

Two further restrictions:

- **Separate BN forwards do not restore per-sample independence.** They avoid mixing real and fake statistics, but each image's gradient still contains derivatives of other images' scores. Custom minibatch-statistics layers are not detected by `batch_coupled` either. Reject batch-coupled D for the R3GAN preset. Evidence: GAN `:11–13`, `:123–125`, `:149`; paper p.5 explicitly identifies this incompatibility. **Confidence: high.**
- **Wavelet mode changes what is regularized.** neosr detaches *after* `AdvView`, so the penalty is with respect to wavelet-view coordinates, not original RGB coordinates through the transform. That is consistent with its "D input" documentation, but differs from regularizing D(A(x)) with respect to x. Start the new recipe with RGB. Evidence: GAN `:129–137`; `N/neosr/losses/wavelet_guided.py:42–49`; [reference Trainer.py:29–36](https://github.com/brownvc/R3GAN/blob/main/R3GAN/Trainer.py#L29). **Confidence: high.**

Logging currently exposes only the combined, interval-scaled penalty on penalty updates. Its logged mean is therefore conditional on those updates, not the all-update average. Split raw R1/R2, weighted contributions, and adversarial D loss; distinguish sampled penalty statistics from per-update contributions. Evidence: GAN `:155–157`; `N/neosr/models/phase.py:422–438`. **Confidence: high.**

**3. SR-specific design**

| Decision | Recommendation | Evidence and confidence |
|---|---|---|
| Scalar versus patch | **Start with a scalar critic over each training crop.** A scalar score can still respond to local artifacts. It gives a clear reference-compatible penalty; a local head is a subsequent quality/cost experiment. | Owner `D:291–309`; reference [Networks.py:87](https://github.com/brownvc/R3GAN/blob/main/R3GAN/Networks.py#L87). **Medium recommendation.** |
| Patch extension | Either scalarize the map **before both** logistic loss and penalty, or explicitly define a local objective and its local/Jacobian penalty. Retaining today's map loss plus mean-score penalty is an intentional surrogate, not exact equivalence. | GAN `:149`, `:162`. **High mathematical distinction.** |
| Pairing for SR | Pair each restored image with its corresponding GT crop, preserving spatial alignment for map comparisons. This is a reasonable SR objective, but correlated SR/GT pairs differ from the paper's independent noise/real draws; its convergence result cannot simply be transferred. | GAN `:27`, `:125`; paper p.2 Eq.2; reference Trainer `:14–20`. **High distinction; medium recommendation.** |
| Penalty interval | Establish correctness and quality with `reg_every=1`; benchmark `4`, then possibly `8`. Lazy regularization changes optimization dynamics even with correct scaling. Avoid making `16` the new R3GAN preset's starting point. | GAN `:54`, `:130–155`; reference Trainer `:35–42`. **Medium recommendation.** |
| D optimizer | Plain Adam, beta1=0, no weight decay, beta2 either fixed 0.99 for a simple baseline or explicitly scheduled 0.9→0.99 for the reference-style recipe. SF optimizers are separate experiments, not substitutes with established equivalence. | Current template `:67–73`; [reference training_loop.py:329–343](https://github.com/brownvc/R3GAN/blob/main/training/training_loop.py#L329). **High distinction; medium recommendation.** |
| G optimizer | Keep the established SR optimizer initially. Pixel/perceptual-dominated G training is different from the paper's adversarial-only generator; changing G momentum simultaneously would confound the experiment. | Template `:59–63`, `:76–100`. **Medium.** |
| GAN start/ramp | Keep existing `start_iter`/`ramp_iters`. Start immediately when fine-tuning an already competent reconstruction model; otherwise use a fixed reconstruction warmup and fixed G ramp. D currently starts with G's GAN term and receives no corresponding ramp. | GAN `:76–98`, `:115–116`. **Medium recommendation; high behavior.** |
| EMA | Keep G-only EMA and its saved state. Existing power warmup is valid but is not the reference's cosine half-life schedule. Add an optional half-life schedule only for the new preset. | `N/neosr/models/phase.py:349–405`; [reference training_loop.py:349–355](https://github.com/brownvc/R3GAN/blob/main/training/training_loop.py#L349). **High.** |
| Adaptive behavior | No adaptive D skipping or `"spike"` clipping in this preset. Use clipping off or a fixed threshold, and retain explicit nonfinite-failure handling. | Owner model2 `:478–492`; neosr phase `:180–211`. **High policy alignment.** |

**Gamma:** there is no defensible universal "128px/256px SR gamma." The reference itself uses **2→0.2 for FFHQ-64** and **150→15 for FFHQ-256**, with different networks and data; these are not a resolution-only formula. Evidence: [reference train.py:198–228](https://github.com/brownvc/R3GAN/blob/main/train.py#L198), paper Table 9, p.23. **Confidence: high.**

For the proposed compact **scalar**, RGB [0,1] SR critic, I would screen equal R1/R2 values **0.1, 1, 10**, initially constant. A **10→1** cosine schedule is a plausible scheduled candidate, not an established optimum. Do not inherit 150→15 unquestioningly, and do not multiply D's gamma by the small G GAN weight. Evidence: neosr weights G separately at GAN `:42`, `:108`; reference dataset-specific settings above. **Confidence: low for numerical choices; high for the weighting distinction.**

Input coordinates matter: for the equivalent critic D_01(x) = D_11(2x-1), input gradients double and squared penalties quadruple. Matching the penalty therefore requires gamma_01 = gamma_11/4. This is a coordinate conversion, not an empirical SR gamma rule. The reference inputs are [-1,1]; neosr's G output is normally clamped to [0,1]. Evidence: [reference training_loop.py:298](https://github.com/brownvc/R3GAN/blob/main/training/training_loop.py#L298); `N/neosr/models/image.py:329–332`. **Confidence: high; derivation.**

For architecture, these distinctions matter:

| Feature | Assessment for SR |
|---|---|
| Fix-up scaling and zero residual-branch last convolution | Keep. They control initial residual variance without normalization. Zero **each residual branch's final convolution**, not the discriminator's final scoring head. `D:240–255`, `D:296–299`. **High.** |
| FIR downsampling | Keep. The reference `(1,2,1)` setting produces an effective normalized `(1,3,3,1)` separable filter before decimation. Its filtering convention and padding matter. `D:112–124`, `D:150–153`; reference [Resamplers.py:6](https://github.com/brownvc/R3GAN/blob/main/R3GAN/Resamplers.py#L6). **High.** |
| No batch statistics | Essential to the simple per-sample penalty interpretation. GAN `:149`; paper p.5. **High.** |
| No spectral norm | Follow the intended recipe; existing MetaGAN/UNet/ResGAN are not that architecture. R1/R2 already provide the chosen regularization mechanism. `N/neosr/archs/metagan_arch.py:122`, `:177–180`; `unet_arch.py:21`; `resgan_arch.py:17`. **High distinction; medium SR preference.** |
| Grouped convolution/inverted bottleneck | Useful capacity/compute choices, not prerequisites for RpGAN correctness. Tune width/cardinality against actual GPU throughput. `D:246–253`; paper p.6. **Medium.** |
| ConvNeXt branding or large kernels | Not requirements. The reference keeps 3x3 grouped convolution and leaky ReLU, rather than copying a standard normalized GELU ConvNeXt block. `D:246–258`; paper p.6. **High.** |
| Activation gain | Preserve plain leaky ReLU 0.2 and initialization gain sqrt(2/(1+0.2^2)). A separate sqrt(2) forward multiplier would be a different parameterization requiring deliberate validation. `D:68–86`; reference FusedOperators `:7–31`. **High.** |

**4. Minimal complete design for neosr**

Keep `Context`, `Loss.expand()`, G/D phases, single weighting in `Loss.compute()`, update-index start/ramp, the eager FP32 penalty path, and existing checkpoint plumbing. Add an architecture through the explicit `ARCHS` table. Do not import the old SRModel branches or registry-based loss wrappers. Evidence: `N/neosr/losses/base.py:33–49`, `:85–94`; GAN `:100–101`; `N/neosr/archs/__init__.py:13`. **Confidence: high.**

A concrete first architecture, intentionally smaller than the paper model (SUPERSEDED for sizing by the addendum in Part 2, which recommends a narrower start):

| Component | Proposed specification | Basis / confidence |
|---|---|---|
| Input/stem | RGB, bias-free 1x1 convolution | `D:418–419`. **High.** |
| 128px preset | Widths `(32,64,128,256,256,256)`; five /2 transitions, ending at 4x4 | Reference-style stage construction at `D:426–460`. **Medium sizing choice.** |
| 256px preset | Add one 256-channel stage and one /2 transition | Same construction. **Medium sizing choice.** |
| Blocks | Two per stage; expansion 2; 1x1 -> biased leaky ReLU -> grouped 3x3 -> biased leaky ReLU -> zero-initialized 1x1; identity addition | `D:240–264`; paper p.6. **Medium capacity; high structure.** |
| Groups | Start with `groups=C/8`, giving 16 expanded channels/group for expansion 2 | Reference cardinality pattern; `D:237–252`. **Medium.** |
| Downsampling | Fixed normalized 4x4 FIR, stride 2, padding 1; optional 1x1 channel projection | `D:112–154`, `:279–288`. **High.** |
| Head | Depthwise 4x4 convolution, linear scalar projection; output `[B,1]`; validate configured crop size | `D:294–309`; fixes the shape ambiguity at GAN `:149`. **High.** |
| Precision | FP32 parameters; ordinary bf16 autocast; no internal forced dtype or external fused-op dependency initially | Existing eager penalty contract at GAN `:135–150`; owner dtype issue at `D:353`. **High.** |

For 1x restoration, select this by **actual D input size**, not SR scale. In neosr's paired dataset, `patch_size` denotes LQ crop size; the present template's 64px at x4 means 256px GT/D inputs. Evidence: `N/neosr/data/paired_dataset.py:40`, `:109–115`; template `:4`, `:15`. **Confidence: high.**

The remaining additions should be small and explicit:

| Addition | Required behavior | Evidence / confidence |
|---|---|---|
| Scalar-safe penalty reduction | Support `[B]`, `[B,1]`, and maps without changing existing map semantics silently. | GAN `:149`. **High.** |
| Optional gamma schedule | For example `r1_gamma_end`, `r2_gamma_end`, `gamma_decay_iters`; absent means today's constant values. Use `Context.it`, never a mutable loss counter. | GAN `:74`, `:128`; Context `:46`. **High.** |
| Shared fixed schedule function | Clamped cosine interpolation from initial to final value, then hold final. Explicitly define its origin relative to GAN start. Validate finite endpoints and integer horizons. | Reference [training_loop.py:29–38](https://github.com/brownvc/R3GAN/blob/main/training/training_loop.py#L29); GAN `:76`. **High.** |
| Optional lr/beta2 scheduling | Generic phase-owned schedule hooks, applied once per optimizer update; no GAN-specific trainer branch. Keep immutable schedule configuration separate from mutable optimizer values for resume comparison. | `N/neosr/models/base.py:33–73`; phase `:83–90`, `:276–331`. **High design recommendation.** |
| Optional EMA half-life scheduling | In update units, decay is 2^(-1/h(t)), with zero half-life meaning copy live weights. Preserve existing EMA behavior when absent. | Phase `:371–397`; reference training loop `:351`. **High.** |
| Diagnostics | Raw R1/R2, effective gammas, weighted penalty contributions, D adversarial loss, G raw/weighted GAN terms, margin distribution, optimizer lr/beta2, EMA decay, regularized-update flag. | Current limited logs at GAN `:155–169`. **High.** |
| New preset only | Preserve current BCE/MetaGAN defaults. Make RGB, scalar D, both penalties, Adam D, bf16, and schedules explicit in a separate preset. | Template `:49–73`, `:97–100`; GAN constructor defaults. **High.** |

All four schedules should share a documented attempted-update convention. Existing neosr clocks advance on skipped and idle updates; the loss schedule should not silently choose a different clock. EMA can evaluate its coefficient at that index while still updating weights only after a successful G step. Evidence: image `:360–361`, `:380`; phase `:170–174`, `:204–205`. **Confidence: high.**

For exact resume, two acceptance levels must be distinguished:

1. **State/schedule exactness:** same network, optimizer moments, EMA, and schedule values for an identical subsequent input stream. Existing plumbing is a suitable base.
2. **Trajectory exactness:** also requires rank-specific RNG and continuation of sampler/crop/augmentation/OTF state. That is additional trainer/data work; a stateless gamma schedule alone cannot provide it.

Evidence: `N/neosr/models/base.py:317–328`, `:379–392`. **Confidence: high.**

**A/B and acceptance**

Run the shipped **MetaGAN+BCE template** against the complete new recipe, with the same G initialization, reconstruction losses, data split, and matched seeds. Report both equal-update and equal-wall-clock results. This is a system comparison: the shipped baseline also differs in D optimizer and its wavelet switch. Attribution to RpGAN alone requires a subsequent controlled loss x architecture comparison with optimizer and adversarial view held fixed. Evidence: template `:49–73`, `:97–100`; GAN view handling `:79–80`. **Confidence: high experimental-design recommendation.**

| Measure | Required reporting | Evidence / confidence |
|---|---|---|
| Perceptual fidelity | DISTS lower-better, LPIPS lower-better, full-reference TOPIQ higher-better; use TOPIQ `dataset="pipal"` for the SR-oriented comparison, and keep variants fixed across runs. | `N/neosr/metrics/calculate.py:382–407`. **High metric semantics; medium dataset preference.** |
| Checkpoint selection | Set `better="lower"` explicitly for LPIPS. The current lower-is-better default set contains only DISTS, so LPIPS otherwise selects larger values. | `N/neosr/config.py:36`; `N/neosr/models/base.py:220–237`. **High.** |
| Real-photo quality | Blind comparisons of skin, hair, foliage, text, smooth areas, noise, halos, ringing, invented texture, and color shifts. Evaluate xSR and 1x restoration separately. | SR/GT image contract at image `:333–339`; full-reference metric contract at calculate `:395`. **Medium evaluation recommendation.** |
| GAN influence | Measure G gradient norms/shares from GAN versus reconstruction terms; do not judge influence from loss magnitudes alone. Use offline diagnostics, without automatic reweighting. | `N/scripts/term_grads.py:32–68`. **High.** |
| Stability | Separate raw R1/R2, margins, G/D gradient norms, clipping and skipped updates; inspect validation trends across seeds. | GAN `:155–169`; image `:373–379`; phase `:188–211`. **High.** |
| Cost | Warmed-up throughput, ordinary versus regularized step latency, peak allocated VRAM, compile overhead, and time to target validation quality. | Distinct execution paths at GAN `:130–150`. **High.** |

My proposed acceptance gate is:

- **Correctness first:** analytic parameter-gradient checks for R1 and R2 separately; scalar/map shape tests; G/D gradient isolation; accumulation equivalence; native FIR forward/backward checks; CUDA bf16 compile-on/off and two-rank DDP coverage; resume across a penalty boundary and schedule endpoint. Existing tests provide starting points, but not this full matrix. **Evidence:** `N/tests/test_s2_losses.py:465–572`, `:594–648`. **Confidence: high.**
- **Quality:** across at least three seeds, paired held-out comparisons show DISTS improvement and TOPIQ improvement with uncertainty intervals, LPIPS stays within a predeclared non-inferiority margin, and blind inspection shows no increased artifact rate. Define a practical improvement threshold before running; statistical significance alone is insufficient. **Evidence:** metric implementations above; this is a proposed experimental criterion. **Confidence: medium.**
- **Efficiency:** the improvement must survive comparison at the owner's wall-clock and memory budget. If every-step penalties lose that comparison, test interval 4 against the verified every-step baseline; do not assume interval scaling preserves quality. **Evidence:** GAN `:130–155`. **Confidence: medium.**

**Open questions**

1. Does "exact resume" mean state/schedule exactness, or identical data and stochastic trajectory? The latter is currently unmet. `N/neosr/models/base.py:383–392`. **High.**
2. Should reference fidelity override the requested sqrt(2) activation multiplier? The inspected reference explicitly uses gain 1. [FusedOperators.py:31](https://github.com/brownvc/R3GAN/blob/main/R3GAN/FusedOperators.py#L31). **High.**
3. What are the actual GT/D crop sizes, effective global batches, and GPU memory limits for xSR and 1x runs? These determine architecture depth and affordable penalty cadence. `N/neosr/data/paired_dataset.py:109–115`; GAN `:130–155`. **High relevance.**
4. Are the validation photographs paired and accurately aligned? The available DISTS/TOPIQ/LPIPS paths require reference images; unpaired photographs need separate blind evaluation. `N/neosr/metrics/calculate.py:382–407`. **High.**
5. Is the wavelet switch part of the required production recipe, or a baseline-specific choice? It changes both D's input distribution and the penalty coordinates. Template `:55–56`; GAN `:129–137`. **High.**

=== PART 2: ADDENDUM ANSWER, resource efficiency (Codex stdout, same thread) ===

The efficiency criterion changes my architecture recommendation: **start considerably narrower than my earlier two-block proposal.** Keep neosr's current G-before-D execution order and FP32 penalty path initially; benchmark a fixed `reg_every=4` against every-step penalties.

Three important corrections/additions:

| Finding | Evidence | Confidence |
|---|---|---|
| The owner's discriminator has fewer parameters than MetaGAN but approximately **12x its counted forward FLOPs** at both 128 and 256 pixels. High-resolution computation dominates. | Owner discriminator `:246–255`, `:381–385`; `neosr/archs/metagan_arch.py:129–155`; calculations below | High |
| Lazy regularization with interval scaling is **not exactly equivalent** to every-step optimization. It preserves nominal average weighting, not Adam dynamics or the training trajectory. | `neosr/losses/gan_loss.py:118–143` | High |
| Current G-before-D ordering prevents straightforward reuse of D-phase real logits for G. Reordering trades compute savings for additional simultaneous activation storage. | `neosr/models/image.py:328–369` | High |
| DUNet is currently **incompatible with these gradient penalties**, because its DySample contains `grid_sample` without the needed double backward. I reproduced the failure. | `neosr/archs/dunet_arch.py:16`; `neosr/archs/arch_util.py:164` | High |
| R3GAN did **not** train everything in bf16: parameters remained FP32 and selected high-resolution stages used bf16. | Paper p.22; [reference training/networks.py:43–44](https://github.com/brownvc/R3GAN/blob/main/training/networks.py#L43) | High |

No files were edited. Parameter counts and operation counts below were calculated using isolated CPU execution of the inspected classes. **They are not GPU timing or VRAM measurements.** The current `gan_loss.py` has changed since the first review; citations below use the newly inspected version.

Abbreviations: `N/` is the neosr worktree; `O/` is `/home/korvin/apps/r3gan/`; `D:` refers to `/home/korvin/apps/traiNNer-redux-dev/traiNNer/archs/rpgan_r3gan_scalar_discriminator_arch.py`.

**(a) Per-update computation, memory, and synchronization**

Let:

- `B` = local microbatch size.
- `A` = accumulation microbatches per optimizer update.
- `k` = regularization interval.
- `F(B)` = one D forward over B images.
- "Inner backward" = `autograd.grad(..., create_graph=True)` computing input gradients.
- "Double backward" = differentiating those input gradients into D parameters during the final loss backward.

The double backward is normally **part of the final D backward**, not another separately invoked `.backward()`.

Counts below assume both penalties are enabled and no cached predictions are supplied.

| Implementation | D forwards per microbatch | Backward work | Penalty precision | Evidence / confidence |
|---|---|---|---|---|
| Minimal loss and v2 | G: `2xF(B)`; D: `2xF(B)` | G backward through fake D; **two inner backwards every call**; one outer D backward combining adversarial gradients and both double-backward paths | Inherits trainer autocast; input leaves are not promoted to FP32 | `O/r3gan_loss.py:54–83`; `O/sr_model.py:548–587`. **High** |
| Alternative hinge loss, ordinary call | G: `2xF(B)`; D: `2xF(B)` | G backward; ordinary D backward | Trainer precision | `O/alternative/r3gan_loss.py:337–377`. **High** |
| Alternative hinge loss, regularized call | Above, plus **two additional cropped D forwards** | Two inner backwards on cropped graphs; final D backward includes adversarial and penalty graphs | D forward attempts FP32, but bf16 input leaves produce bf16 input gradients; returned penalty is cast back | Alternative `:52–57`, `:105–110`, `:296–310`. **High** |
| neosr RpGAN, ordinary update | G: `2xF(B)`; D: **`F(2B)`** | G backward through fake D; one ordinary D backward | Ordinary bf16 autocast when configured | `N/neosr/losses/gan_loss.py:77–86`, `:106–120`. **High** |
| neosr RpGAN, regularized update | Same forward counts; **D's existing `F(2B)` becomes FP32 eager** | One combined inner backward over both inputs; one outer D backward containing adversarial and both penalty derivatives | FP32 leaves, activations and reductions, subject to runtime reduced-precision arithmetic settings | GAN `:121–145`. **High** |
| `sr_model3.py`, ordinary-GAN branch | G: `F(B)`; D: `2xF(B)`; extra `F(B)` every fourth iteration | Ordinary G/D backwards, then one inner backward and a separate outer R1 backward | Extra R1 forward is outside its shown autocast scope | `O/sr_model3.py:505–506`, `:624–650`. **High** |

The last row is **not a working R3GAN implementation**. Likewise, model2's missing `generator_forward`/`discriminator_forward` implementations cannot receive an honest executable cost estimate. The `s3.diff` integration runs D only on the accumulation boundary, then divides by `A`; its reduced D work comes with the previously identified accumulation error. Evidence: `O/sr_model2.py:466`, `:604`; `/home/korvin/apps/s3.diff:64–70`, `:131`. **Confidence: high.**

For neosr, multiply microbatch work by `A`: **every microbatch in a regularized update receives the penalty**, because all share `Context.it`. The alternative instead counts D-loss calls, so its cadence changes relative to optimizer updates when accumulation changes. Evidence: `N/neosr/models/image.py:322–346`; alternative `:241–247`. **Confidence: high.**

For warmed-up execution, measure:

T_average ~ (1 - 1/k) T_ordinary + (1/k) T_regularized.

Do not estimate the penalty as simply "one more forward" or assume a fixed 2x multiplier: convolution input-backward and mixed-derivative kernels have different costs. Evidence: GAN `:131–144`; phase `:166`. **Confidence: high for the accounting; hardware-dependent timing.**

What remains alive:

| Path | Memory retained until the relevant backward completes | Evidence / confidence |
|---|---|---|
| G phase | G's saved activations and fake-D backward activations. The real-D forward is already under `no_grad`. | GAN `:81–83`; image `:328–341`. **High** |
| Regularized D phase | D's saved forward tensors; derivative-graph tensors created by the inner backward; real/fake input gradients needed for the squared norm; input conversions and concatenation storage where allocated | GAN `:123–141`. **High** |
| Alternative regularized phase | Original adversarial D graphs **plus** two additional penalty-forward graphs, cloned cropped inputs, and their derivative graphs | Alternative `:52`, `:296–310`, `:358–359`. **High** |
| Minimal loss | Both D graphs and derivative graphs, plus the accidental `[B,B,H,W]` broadcast result for ordinary one-channel patch maps | Minimal `:75–85`. **High** |

`create_graph=True` does **not** construct a dense Hessian. It constructs a differentiable graph for the input-gradient calculation; retention of the original graph defaults to enabled in this case. The expensive objects are saved activations and derivative intermediates, not the scalar penalty. [PyTorch autograd.grad documentation](https://docs.pytorch.org/docs/2.9/generated/torch.autograd.grad.html). **Confidence: high.**

The current order is beneficial for peak memory: G backward completes before the regularized D forward begins. G's usual saved backward tensors can therefore be released before the higher-order D graph is built. Lazy regularization reduces average time, but **does not reduce the peak of its regularized update**. Evidence: image `:341–346`; phase `:166`. **Confidence: high.**

Host synchronization:

| Implementation | Relevant synchronization |
|---|---|
| Minimal loss | Python counter increment, tensor shape lookup and NumPy cosine introduce **no GPU-to-host synchronization themselves**. Its trainer does synchronize on G-loss finiteness. `O/r3gan_loss.py:79–82`; model `:495`. **High** |
| Alternative | With `k>1`, `_gp_step...item()` synchronizes every D-loss call. Actual cropping adds **four `.item()` synchronizations** for two coordinates on each of real/fake. `:245–247`, `:211–212`, `:287–294`. **High** |
| neosr GAN kernel | Host integer scheduling and detached logging add no explicit scalar GPU read. GAN `:118`, `:145`; loss base `:60–61`. **High** |
| neosr optimizer phases | One explicit `bool(norm.isfinite())` synchronization per active phase per optimizer update, normally two total for G+D. Logging later performs aggregated reads/collectives. Phase `:187–188`, `:409–438`. **High** |
| Old adaptive integrations | Adaptive loss comparisons add synchronization and potentially rank-dependent decisions. They should not be imported. `O/sr_model2.py:478–492`. **High** |

**(b) Exact savings versus changed training**

Here "exact" means the same mathematical objective/gradient under stated assumptions. It does not promise bitwise equality across different batch shapes or kernels; PyTorch explicitly distinguishes these. [Numerical-accuracy documentation](https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html). **Confidence: high.**

| Change | Classification | Practical implication |
|---|---|---|
| Combine real/fake into one D forward | Mathematically exact for a deterministic, batch-independent D with unchanged state | Fewer launches and potentially better utilization, **not half the convolution FLOPs**. It still processes `2B` images and allocates the concatenation. GAN `:113`. **High** |
| Combine R1/R2 into one inner `autograd.grad` | Exact for independent real/fake branches | Reduces dispatch overhead; arithmetic still covers both inputs. Already implemented. GAN `:137–141`. **High** |
| Use the same forward for D adversarial loss and penalties | Exact when both would otherwise use the same precision/function/state | Avoids the alternative's two extra penalty forwards. Already implemented. GAN `:131–144`; alternative `:296–305`. **High** |
| Remove unnecessary input clones | Exact if subsequent operations do not mutate the shared storage | `detach().float().requires_grad_()` is sufficient; `.float()` allocates only when conversion is necessary. Alternative `:52`; neosr `:123–125`. **High** |
| Replace folded-channel FIR with equivalent depthwise convolution | Same filtering operator with matched kernel/padding | Can avoid a channels-last reshape copy. Benchmark kernels; fewer copies do not guarantee lower latency. `D:150–154`; image `:115–116`. **High equivalence; medium speed benefit** |
| Freeze D parameters during G's forward | Same G derivative if D's behavior/state is otherwise unchanged | Can reduce tensors saved for D weight gradients. Requires compile/DDP qualification; toggling requirements may create additional compiled variants. `N/tests/test_gan.py:313–349`. **High mathematical equivalence** |
| Lazy penalty with `k` scaling | **Not exact every-step optimization** | Preserves nominal average coefficient; changes gradient timing, optimizer moments and clipping. GAN `:118–143`. **High** |
| Random `gp_crop` | Changes the regularized function/distribution | Smaller input, different boundary/receptive-field/pooling behavior; no exact equivalence to full-crop R1/R2. Alternative `:287–305`. **High** |
| Relativistic-average instead of pairwise | Changes the adversarial objective | The average comparator is not a cheaper algebraic form of pairwise logistic loss. Alternative `:233–239`; neosr GAN `:150`. **High** |
| `create_graph=False` fallback | Removes the penalty's parameter gradient | Not an efficiency option for R3GAN. Alternative `:94–100`. **High** |

A simple illustration of the lazy distinction: a constant penalty gradient g every step has mean square g^2. Applying k*g every k steps has the same average gradient but mean square k*g^2. Adam therefore sees different second-moment statistics even in this simplified case. Fixed clipping can additionally truncate the larger pulses. Evidence: penalty multiplier at GAN `:143`; clipping at phase `:180–182`. **Confidence: high; mathematical derivation.**

Concatenation is also **not generally equivalent for the existing MetaGAN**: spectral-normalization updates, dropout and DropPath make forward count/order relevant. BN would introduce cross-sample coupling. A new deterministic normalization-free R3GAN D makes this optimization much cleaner. Evidence: `N/neosr/archs/metagan_arch.py:106`, `:122`, `:156–180`; GAN `:11–13`. **Confidence: high.**

**Real-logit caching, checked against the actual phase order**

Current order is:

`G forward -> G loss/backward -> D loss/backward -> G optimizer step -> D optimizer step`.

Thus D weights do remain unchanged across the two losses, but **the useful D-phase real logits do not exist when G needs them**. Evidence: image `:328–369`. **Confidence: high.**

There are two possible restructurings:

1. **Prepare real D logits with a graph during G's phase**, give G a detached view, and retain the graph for D. This can reduce four image-batch forward equivalents to three, but keeps the real D graph alongside G's graph and complicates the current concatenated/DDP path.
2. **Run D loss/backward before G backward**, cache its detached real logits, and delay the D optimizer step until G backward finishes. This can reuse the concatenated D forward, but G's large saved activation graph now overlaps the entire higher-order D computation.

Both require identical D inputs, precision, parameters and relevant state. They are mathematically plausible for the proposed deterministic D, **not unconditional drop-in optimizations**. On current regularized updates, D logits are FP32 while G's original logits are produced under bf16 autocast; replacing one with the other also changes numerical evaluation. Evidence: GAN `:81–83`, `:128–131`; phase `:163–166`; image `:341–369`. **Confidence: high.**

**Recommendation:** retain the present ordering initially. Measure caching only after the smaller D is working; saving one D forward may be a poor exchange for overlapping G and penalty graphs. **Confidence: medium; memory-lifetime evidence above.**

**(c) FP32, TF32, or bf16 penalty execution**

The historical comparison needs qualification:

- **R3GAN:** FP32 parameters, selected high-resolution bf16 stages, and penalties differentiated through that mixed-precision network. It does not establish that every activation, input gradient and reduction should be bf16. Evidence: local paper p.22; [reference training/networks.py:43–44](https://github.com/brownvc/R3GAN/blob/main/training/networks.py#L43), [Trainer.py:10–12](https://github.com/brownvc/R3GAN/blob/main/R3GAN/Trainer.py#L10). **Confidence: high.**
- **StyleGAN2-ADA PyTorch:** its R1 loss calls ordinary `run_D`; it does not explicitly force the whole R1 D pass into FP32. Configured blocks can use FP16, while the final epilogue uses FP32. Therefore "StyleGAN2 always runs the entire R1 pass in FP32" is too broad. Evidence: [loss.py:111–125](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/loss.py#L111), [networks.py:556–558](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/networks.py#L556), [networks.py:645](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/networks.py#L645). **Confidence: high.**

For neosr:

| Mode | Assessment |
|---|---|
| **FP32 tensors, eager D, current fast arithmetic settings** | Keep as the initial production/default penalty path. It is simple and avoids retaining the higher-order graph in bf16. GAN `:123–141`. **Medium recommendation; high implementation evidence** |
| **Strict IEEE FP32** | Use as the numerical comparison baseline, with reduced-precision matrix/convolution arithmetic disabled. Otherwise "FP32" does not establish full FP32 arithmetic. Runtime `:106–110`. **High** |
| **bf16 mixed-precision penalty D** | Worth benchmarking for this simple conv/leaky-ReLU D. Keep FP32 parameters and **FP32 input leaves**, cast logits to FP32 before score reduction, and square/sum returned input gradients in FP32. Keep the higher-order path eager. **Medium recommendation**, grounded in GAN `:123–141` and the reference mixed-precision implementation |
| **Hard-cast the entire D and penalty leaves to bf16** | Do not use as the first implementation. It sacrifices optimizer/master-weight precision and/or input-gradient precision without being required for activation savings. Owner `D:353–354`, `D:388`; alternative `:52–57`. **High** |

One nuance: `fast_matmul=true` sets **`"medium"`** matmul precision, not merely a universal "TF32 mode." PyTorch permits BF16-based internal matrix arithmetic under that setting; cuDNN convolution TF32 is controlled separately. Tensor storage can remain FP32 throughout. Evidence: `N/neosr/runtime.py:106–110`; [PyTorch precision-setting documentation](https://docs.pytorch.org/docs/2.9/generated/torch.set_float32_matmul_precision.html). **Confidence: high.**

Numerical risks of bf16 higher-order execution:

- **Mantissa precision, rather than FP16-like exponent range, is the primary concern.** Rounding affects small differences and cancellation in input/parameter gradients.
- Leaky ReLU's piecewise-constant input derivative does **not** make R1/R2 ineffective: grad_x D still depends on convolution weights, so mixed input/weight derivatives are nonzero.
- Rounding preactivations near zero can change the selected leaky-ReLU slope. Grouped convolution is mathematically compatible with double backward, but actual backward kernels, reductions and throughput need GPU qualification.
- FP32 input leaves and reductions help, but do not recover information lost inside bf16 convolutions.

Evidence: owner residual operations at `D:246–264`; penalty differentiation at GAN `:138–141`; [PyTorch numerical-accuracy guidance](https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html). **Confidence: high for mechanisms; medium for their practical magnitude on this workload.**

Compare FP32/TF32/bf16 using **both penalty values and their parameter-gradient vectors**, including norms, cosine similarity, per-layer outliers and zero/nonfinite fractions at initialization and trained checkpoints. Finite scalar losses alone are insufficient. This is a proposed qualification procedure. **Confidence: medium.**

**(d) Right-sized discriminator and calculated parameter/FLOP comparison**

I recommend this **lean starting candidate**, with one residual block per stage:

| GT crop | Widths, high to low resolution | Cardinalities |
|---|---|---|
| 128 | `(16,32,64,128,192,256)` | `(4,8,16,32,48,64)` |
| 256 | `(16,32,64,128,192,256,256)` | `(4,8,16,32,48,64,64)` |

Use expansion 2, grouped 3x3 convolution, bias-free linear convolutions, reference-style fix-up initialization, learned activation biases, leaky ReLU 0.2, FIR /2 transitions, and the 4x4 scalar head. These cardinalities give **eight expanded channels per group**. Both layouts reach an actual 4x4 final feature map without adaptive pooling. Evidence: owner configurable construction `D:235–258`, `D:267–309`, `D:426–460`. **Confidence: high structural validity; medium-low that this capacity is sufficient for the owner's data.**

Also benchmark cardinalities divided by two, i.e. **16 expanded channels/group**. More arithmetic can nevertheless run faster on a particular GPU because heavily grouped convolutions can be inefficient. **Confidence: medium; hardware-dependent recommendation.**

Calculated forward arithmetic:

- One RGB image, one D forward.
- **1 MAC = 2 FLOPs.**
- Includes convolution, linear layers, attention QK/AV products, and fixed FIR convolution.
- Excludes activation/normalization arithmetic, spectral-normalization iterations, interpolation/grid-sampling arithmetic, copies, optimizer work and all backward work.
- Parameter counts include trainable biases/norm parameters, not buffers.

| Discriminator | Parameters at 128 / 256 | Counted GFLOPs at 128 | Counted GFLOPs at 256 |
|---|---:|---:|---:|
| **MetaGAN defaults** | 6.017M / 6.017M | **0.319** | **1.297** |
| **Lean candidate, 8 expanded channels/group** | **0.687M / 0.987M** | **0.331** | **1.332** |
| Lean candidate, 16 expanded channels/group | 0.786M / 1.123M | 0.477 | 1.919 |
| Width-32 candidate, one block/stage (1) | 1.210M / 1.546M | 1.218 | 4.884 |
| My earlier width-32, two-block proposal (1) | 2.372M / 3.045M | 2.375 | 9.522 |
| Owner's defaults | 1.730M / 1.730M | **3.840** | **15.362** |
| **DUNet defaults** | 3.231M / 3.231M | **11.669** | **46.674** |

(1) Widths `(32,64,128,256,256,256)` for 128; append 256 for 256. Cardinality C/8, expansion 2.

Evidence: owner `D:246–255`, `D:381–386`; `N/neosr/archs/metagan_arch.py:38–61`, `:92–155`; `N/neosr/archs/dunet_arch.py:26–43`; DySample offset/scope convolutions at `N/neosr/archs/arch_util.py:127–128`. Counts were calculated from these implementations, including observed tensor shapes. **Confidence: high for the stated operation-count convention; low as a direct latency predictor.**

This revises my earlier recommendation: **two wide blocks at every resolution are unnecessarily expensive as the first SR experiment.** MetaGAN reduces spatial dimensions aggressively before its heavy blocks; the owner's early dense convolutions operate at full and half resolution. Parameter count conceals that difference. Evidence: MetaGAN `:129–155`; owner `D:246–255`, `D:381–383`. **Confidence: high for cost explanation; medium for capacity recommendation.**

The activation-memory difference is substantial even before double backward. At local `B=8`, concatenated D batch `2B=16`, and GT 256:

- Owner/width-32 first expanded feature: `[16,64,256,256]` = **256 MiB in FP32**.
- Lean first expanded feature: `[16,32,256,256]` = **128 MiB in FP32**.
- Each becomes half that size in bf16.

These are **individual tensor sizes**, not measured peaks or sums of all saved tensors. Evidence: expansion at `D:235`, `D:246`; concatenation at GAN `:113`. **Confidence: high; arithmetic.**

The owner's FIR implementation folds batch and channels with `reshape`. With neosr's channels-last tensors, that can require a copy. A native depthwise fixed-filter convolution avoids that layout transformation while preserving the filter. Evidence: `D:150–154`; image `:115–116`. **Confidence: high for the layout concern; medium for net speed improvement.**

DUNet should not enter the R1/R2 benchmark until its resampler changes: my CPU probe failed with `derivative for aten::grid_sampler_2d_backward is not implemented`. Replacing DySample with fixed interpolation would create a different architecture, so label that experiment separately. Evidence: DUNet `:16`; arch_util `:164–170`. **Confidence: high.**

Finally, matching **D-forward** FLOPs does not make the complete recipe as cheap as MetaGAN+BCE. Current RpGAN processes four B-sized D forwards' worth of images per microbatch; BCE processes three. For the lean 256 candidate versus MetaGAN, that is about **37% more counted forward arithmetic before adding penalties**. Evidence: GAN `:81–86`, `:113`; counts above. **Confidence: high for arithmetic, not elapsed time.**

**(e) Every-step versus lazy penalties and the resource A/B**

My recommendation is:

1. **Correctness/reference arm:** lean D, both penalties every step, eager FP32 penalty execution.
2. **Efficiency candidate:** same model/objective coefficients, fixed `reg_every=4`, with the existing factor-of-four penalty scaling.
3. **Precision candidate:** the same interval-4 configuration with bf16 mixed-precision D penalty execution and FP32 leaves/reductions.
4. Test interval 8 only if interval 4 remains too expensive or offers a useful quality/time tradeoff.

These are fixed configurations, with no adaptive interval, damping or automatic gamma changes. The current default interval 16 should remain preserved for existing configurations, but should not be assumed suitable for the new recipe. Evidence: GAN `:37–38`, `:118–143`. **Confidence: medium recommendation.**

A small G GAN coefficient is **not a computational saving**, and it does not directly weaken D's objective: D has its own unweighted term. It may make SR quality less sensitive to adversarial changes, but that must be measured; it does not justify omitting R2. Evidence: GAN `:42`, `:96`, `:143–150`; phase `:164–166`. **Confidence: high for implementation; medium for SR sensitivity.**

The A/B should report:

| Dimension | Required measurement | Why |
|---|---|---|
| Update latency | Mean ms/**optimizer update**, ordinary and regularized updates separately; p50/p95; amortized interval average | Prevents cheap ordinary updates from concealing expensive penalty spikes |
| Memory | Peak allocated and reserved VRAM over complete intervals, after optimizer state and compile warmup; maximum across ranks | Lazy regularization retains the regularized-step peak |
| Throughput | GT images/second at identical effective batch and accumulation; additionally maximum feasible throughput under the same VRAM cap | Separates kernel speed from batching advantages |
| Compilation | Startup time, graph count/recompilations, warm steady-state time | Compiled ordinary passes and eager penalty passes have different startup/runtime behavior |
| Quality | DISTS lower-better, TOPIQ-PIPAL higher-better, LPIPS lower-better and blind artifact review at equal updates **and equal wall-clock time** | Tests whether saved compute buys useful SR quality |
| Regularization | Raw R1/R2, weighted contributions, parameter-gradient norms, clipping incidence, skipped updates | Detects lazy pulses being clipped or bf16 changing effective regularization |
| Resume | Continue across an ordinary->regularized boundary and compare schedule/optimizer state under identical subsequent batches | Confirms efficiency changes do not introduce hidden counters |

Evidence for these measurement targets: GAN's two paths at `:118–145`; phase clipping/synchronization at `:180–205`; wrapping at `N/neosr/models/base.py:144–173`; metric implementations at `N/neosr/metrics/calculate.py:382–407`. **Confidence: high for measurement necessity; medium for proposed experiment design.**

Use CUDA events for device work and synchronized wall-clock measurements for end-to-end updates, but avoid adding a synchronization after every internal operation in the throughput run. Record GPU, batch/accumulation, crop, memory format, TF32/matmul settings, compile mode and optimizer alongside results. **Confidence: high; the relevant execution settings are at runtime `:100–110`, image `:115–116`, and models/base `:170–173`.**

**Acceptance:** promote interval 4 or bf16 only if it improves the measured speed/memory/quality tradeoff against the verified every-step FP32 arm. Require no meaningful DISTS/TOPIQ regression across repeated seeds, no new artifact pattern, and no hidden clipping/nonfinite problem. Preserve the FP32 every-step arm as the reference even if the efficient preset wins deployment. Numerical and quality non-inferiority margins should be fixed before examining results. **Confidence: medium; proposed acceptance criterion, not an observed result.**
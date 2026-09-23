# Final Boss rank-8 distillation

This release follows Anima's original / distill / off comparison format. Supra's
teacher is already an ordinary rank-16 LoRA. Its distill is a smaller ordinary
rank-8 LoRA fitted to teacher activations, not a nonlinear particle conversion.
There are separate distills for the measured 1,600-step run and the selected
28,000-step checkpoint from the convergence run.

## Fit

For a teacher branch, let `D` be its down projection, `U` its up projection,
and `X` the observed input rows. Its output is `X Dᵀ Uᵀ` (alpha/rank = 1).
Accumulate `C = (X Dᵀ)ᵀ (X Dᵀ)` on calibration inputs. Take the leading eight
left singular vectors `Q` of `U sqrt(C)`. The student factors are:

```
U8 = Q
D8 = Qᵀ U D
```

This is the rank-constrained least-squares solution on those input activations:
the teacher branch output is projected into its highest-energy output subspace.
Only the small 16×16 covariance and an output-width×16 SVD are needed per
projection. Tests compare against the direct SVD optimum on non-isotropic
inputs, including singular calibration covariance.

The fit uses the six neutral training captions plus the three preservation
captions, seeds 49001 and 49002, teacher-enabled 50-step Euler trajectories,
CFG 3, time positions 0/5/10/…/45, and 128 evenly spaced tokens per projection
call, including conditional and unconditional halves. The bridge guardian,
fruit prompt, and sample seeds 42/1234 are excluded. Rank 8 is fixed in advance.
There is no development-guided rank sweep or additional optimizer training.

## Evaluation

Seeds 59001/59002 evaluate projection fitting on the same nine calibration
captions without fitting to their activations. Every projection's squared error,
teacher output norm, token counts and relative MSE are saved.

An additional evaluation covers all six subject captions plus the unseen
guardian and fruit. It measures CFG velocity error at ten positions on each
teacher trajectory and endpoint error after an entire student trajectory.
Velocity error is normalized by the teacher-minus-base velocity edit, summed
across cases. Endpoint errors use the teacher-minus-base latent displacement
per case, then are averaged. Near-zero teacher edits can magnify ratios; the
JSON reports retain per-case results. These are approximation diagnostics,
not image-quality metrics or evidence that different prompts are unaffected.

Both students are saved, zeroed, reloaded, and checked for exactly identical
velocity predictions. Strength zero must exactly equal the teacher runtime's
unadapted model. Published images are rendered from the reloaded exports.
All original / distill / off images share prompt, seed, sampling settings and
base model. PNG sidecars name the adapter SHA-256 and pinned model revisions.

## Format

The adapters use `supra-native-lora-v1`, with `<module>.down.weight` and
`<module>.up.weight`. Rank and alpha are both 8. Context projection and every
self/cross-attention projection use the same targets as the rank-16 teacher.
The distill has 849,408 fp32 parameters, half the teacher's 1,698,816.
Native inference automatically reads the rank. ComfyUI loading is unverified.

Use `scripts/distill_final_boss.py --teacher <teacher.safetensors> --out <new-dir>`.
The script records teacher SHA-256, metadata, fitting method, calibration
details, timing, evaluation, and complete matched samples. See
[release reproduction](release.md) for the concrete commands.

# Ordinary LoRA distillation

**TODO.** No distilled LoRA is fitted in this revision. `validation/distillation-primitive.json` records that.

## What this file will describe

Anima publishes two artifacts per slider: the original particle adapter, and an ordinary LoRA regressed onto that adapter so a standard loader can approximate it. The Supra gallery is reserved for the same pair. Particle stays the reference. Distill is the linear approximation. See the Anima write-up for the ridge fit this product intends to follow once a particle teacher exists: [anima-concept-sliders DISTILLATION.md](https://github.com/HyperGAN/anima-concept-sliders/blob/main/DISTILLATION.md).

That fit needs a trained nonlinear teacher, cached projection activations, and a Supra export for both a native runtime and a Comfy graph. None of those exist yet.

## What training is today

[particle-sliders PR 130](https://github.com/HyperGAN/particle-sliders/pull/130) trains a LoRA directly with `train_lora_supra.py` (UNI, unused-token hold, `--lora_targets cross`, rank 16). That LoRA is the slider itself. It is not a distill of a particle teacher, and this repo does not report a projection MSE for it.

When a distill lands, this file should record:

- teacher checkpoint hash
- rank, alpha, and which Supra modules were fitted
- calibration rows (train) versus report rows (dev)
- the ridge objective and the held-out projection error
- the image pairs at the strengths in `data/release-samples.json`

Until those numbers are measured, leave the error blank. Do not copy Anima's MSE figures onto Sunlit.

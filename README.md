# Supra Concept Sliders

**Scaffolding.** Product repo for **[SupraLabs/Supra2-IMG](https://huggingface.co/SupraLabs/Supra2-IMG)** concept sliders, parallel to [anima-concept-sliders](https://github.com/HyperGAN/anima-concept-sliders). This revision has no gallery images, no particle weights, and no distilled LoRAs.

Train and infer live in [HyperGAN/particle-sliders](https://github.com/HyperGAN/particle-sliders) [`train_lora_supra.py`](https://github.com/HyperGAN/particle-sliders/pull/130) ([pull request 130](https://github.com/HyperGAN/particle-sliders/pull/130)). This repo keeps the product catalog, the release identity, and the reproduce notes. It does not vendor the DiT trainer. The pin is [`backend.lock.json`](backend.lock.json).

When matched samples exist, the top of this README becomes the gallery, in the same order Anima uses. Until then the sections below are the contract those samples have to meet.

## Particle, Distill, Off

Each comparison will read **Particle → Distill → Off**, left to right, on one prompt and one seed.

| Column | What it is |
|---|---|
| Particle | The original slider, strength 1 (the trained endpoint) |
| Distill | An ordinary LoRA fitted to that slider, same strength |
| Off | The base Supra2-IMG model, strength 0 |

Distill is the approximation you can load without a particle plugin. Particle is the reference effect. They are not interchangeable until a gallery shows otherwise. [Distillation](DISTILLATION.md) and the [ComfyUI plugin](COMFYUI.md) are **TODO** in this revision, so there is nothing to download yet.

Reserved gallery settings, from [`data/release-samples.json`](data/release-samples.json):

| | |
|---|---|
| Slider | Sunlit |
| Prompt | a person standing in a doorway, neutral daylight, even illumination |
| Seed | `29001` |
| Size | 256 × 256 |
| Sampler | Euler flow, 50 steps, CFG 3.0 |
| Strengths | Particle 1, Distill 1, Off 0 |
| Images | none yet |

That prompt is a held-out dev row. Training rows stay in [`data/train.json`](data/train.json).

### Sunlit

Lighting slider: neutral daylight to a warm golden sun. The student and the sample caption stay on the neutral line. The warm line is the teacher only. Age is outside this catalog.

<details><summary>Train prompts (four rows)</summary>

Neutral captions, which the student sees:

- a woman sitting by a window, neutral daylight, even illumination
- a man reading at a table, neutral daylight, even illumination

Teacher captions:

- a woman sitting by a window, warm golden sunlit glow
- a man reading at a table, warm golden sunlit glow
- a woman sitting by a window, amber sunlight, warm glow
- a man reading at a table, amber sunlight, warm glow

</details>

A second lighting card, Candlelit, sits in [`configs/supra/candidates/candlelit-v1.json`](configs/supra/candidates/candlelit-v1.json) as a draft. It is not in the train file.

## Get the adapters

| Slider | Original particles | Distilled LoRA | Status |
|---|---|---|---|
| Sunlit | — | — | not released |
| Candlelit | — | — | draft card only |

Particles, once they exist, will load through the Comfy stub in [`comfy_particle.py`](comfy_particle.py) or the native path in [REPRODUCE.md](REPRODUCE.md). Distilled LoRAs, once they exist, will use a standard Load LoRA on a Supra graph. Neither path runs today.

## Model

[SupraLabs/Supra2-IMG](https://huggingface.co/SupraLabs/Supra2-IMG), Hub revision `10dec6e4b4b5d1c44fd1d7d3fe5e50137333da5b`. Tiny text-to-image DiT, about 104.1M parameters (`SupraDiT()` is 104,094,736), trained from scratch. Apache-2.0 weights stay on the Hub.

| | |
|---|---|
| Encoder | frozen Flan-T5-Base, context 128, `D_CTX=768` |
| VAE | SD-VAE-FT-MSE, scale `0.18215` |
| Resolution | **256²**, latent **32²**, patch **2** |
| DiT | `D_MODEL=576`, `DEPTH=14`, `N_HEADS=9`, `HEAD_DIM=64`, `MLP_RATIO=4` |
| Sampling | Euler flow, `t = i/K`, `z <- z + (1/K) * v` |
| Recommended sample | CFG **3.0**, **50** steps |

Prompts in this catalog stay short on purpose. Flan-T5 context is 128 tokens.

The live train card is cross-attention LoRA, rank 16, UNI with unused-token hold, `--lm_target trajectory`. Flan-T5 and the VAE stay frozen. Details: [FORMULATION.md](FORMULATION.md). How to call the trainer: [REPRODUCE.md](REPRODUCE.md).

## Layout

| Path | Role |
|---|---|
| [`configs/supra/`](configs/supra) | Model pin, character and lighting cards, train card, release identity |
| [`data/`](data) | Train, dev, and canary prompts the trainer can load |
| [`backend.lock.json`](backend.lock.json) | particle-sliders PR 130 commit |
| [`REPRODUCE.md`](REPRODUCE.md) | Calls `train_lora_supra.py` in that checkout |
| [`FORMULATION.md`](FORMULATION.md) | UNI lighting card the backend actually trains |
| [`DISTILLATION.md`](DISTILLATION.md) | **TODO** ordinary-LoRA fit |
| [`COMFYUI.md`](COMFYUI.md) | **TODO** plugin |
| [`lumen_studio/`](lumen_studio) | **TODO** studio. The Anima studio is the template and is not copied here |
| [`validation/`](validation) | Scaffold status. Nothing here is a passed release check |

## License

Product source is MIT ([LICENSE](LICENSE)). Supra2-IMG weights are Apache-2.0 and are not included. See [NOTICE](NOTICE).

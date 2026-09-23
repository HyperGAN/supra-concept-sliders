# Shared core

Supra2-IMG trains the shared particle game. It does not keep a second copy
of that game in this repository.

Pin [`particle-sliders-core`](https://github.com/HyperGAN/particle-sliders/tree/a119ca1ecd3d5d6c437065839d22739b04f2f4d8/packages/particle-sliders-core)
from HyperGAN/particle-sliders:

```text
particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@a119ca1ecd3d5d6c437065839d22739b04f2f4d8#subdirectory=packages/particle-sliders-core
```

Import `particle_sliders.winning_formulation`. The pip name
`concept-slider-core` and the import `concept_slider_core` are the old Anima
extraction. This product does not install or import them.

```python
from particle_sliders import winning_formulation

stamp = winning_formulation()
stamp.require({**stamp.as_dict(), "g_lr": 1e-4, "adv_batch": 4})
bridge = stamp.bridge()
regularizer = stamp.regularizer()
d_loss, g_loss, vic = stamp.losses()
```

`require()` locks the gmix architecture and the current formulation overlay
(`particle-gmix-1600-v2`, provisional until ParticleGAN #38 crowns a full
live leaderboard winner). Generator learning rate and batch size are a
Supra model surface. Hub ids, prompts, and sampling are not formulation knobs.

| Stays in particle-sliders-core | Stays in this repo |
|---|---|
| Routed particle adapter, global-mix critic, paired losses, particle VIC, gradient penalty | Train entry `scripts/train_lora_supra.py` |
| `winning_formulation()` and `require()` | Euler sample card: 256², 50 steps, CFG 3, `t = i/K`, `z <- z + (1/K) * v` |
| Formulation parameters | Base id `SupraLabs/Supra2-IMG`, product Hub id `ntc-ai/supra-concept-sliders` |
| | Prompt card `configs/supra/prompts-supra.yaml` |
| | Ordinary LoRA release on `ntc-ai/supra-particle-sliders` |

`vendor/particle-sliders/conceptmod/textsliders/supra_model.py` is the DiT
backbone used by ordinary LoRA inference. It is not the particle game.
`backend.lock.json` records the core pin and that file's SHA-256. Do not
point training at a `PARTICLE_SLIDERS_ROOT` checkout.

There is no ComfyUI node in this repository. When one exists, its class name
stays here next to the Euler card.

## Train

CPU smoke (no Hub weights):

```bash
python scripts/train_lora_supra.py --dummy --steps 8 --device cpu \
  --save-dir outputs/lighting-supra-dummy
python scripts/train_lora_supra.py --print-card
```

A CUDA run loads the pinned Supra2-IMG checkpoint and feeds mean-pooled
Euler velocity edits into the same game:

```bash
python scripts/train_lora_supra.py --allow-hub --device cuda:0 \
  --save-dir outputs/lighting-supra
```

The published 1,600-step final-boss artifacts are ordinary LoRAs trained by
`scripts/train_final_boss.py`. That recipe is separate from this game.

## Research trainer still in particle-sliders

HyperGAN/particle-sliders still contains
`conceptmod/textsliders/train_lora_supra.py`. That file is the research UNI
trainer (`FakeSupraBackend`, trajectory / direct / cfg-delta losses in
`supra_slider.py`). It does not call `winning_formulation()`, and its live
path still stops before loading Flan-T5.

This pull request does not rewrite particle-sliders. Follow-up, in that
repository only: replace `train_lora_supra.py` with a short comment pointing
at `scripts/train_lora_supra.py` in HyperGAN/supra-particle-sliders. Leave
`supra_model.py` there for the research dummy DiT; this repo already vendors
that backbone for ordinary LoRA inference.

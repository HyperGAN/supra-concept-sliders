# ComfyUI

**TODO.** No ComfyUI plugin is wired for Supra2-IMG. `comfy_particle.py` exports empty node maps so the product layout matches anima-concept-sliders without registering a node that pretends to load weights. `validation/comfyui.json` records `plugin_wired: false`.

## Why it is blocked

Supra2-IMG is a small DiT with frozen Flan-T5-Base and SD-VAE-FT-MSE, sampled with Euler flow at 256², CFG 3.0, 50 steps. The Anima plugin wraps ComfyUI's Anima transformer and a particle branch. Those module names and that branch are not this model. Copying `AnimaParticleSlider` here would fail on load.

A working plugin needs all of the following:

1. A ComfyUI graph that actually runs Supra2-IMG (DiT, Flan-T5, SD-VAE).
2. A released slider file (particle adapter or, later, a distilled LoRA).
3. Projection names that match the Supra modules the trainer adapts: `ctx_proj`, `cross_attn.q`, `cross_attn.kv`, `cross_attn.proj` for the default `cross` card.

## What to use instead

Inference for the base model is the Hub `inference.py` (CFG 3.0, 50 steps). Slider training and the dummy CPU path are `train_lora_supra.py` in particle-sliders. Commands are in [REPRODUCE.md](REPRODUCE.md).

When a distilled LoRA exists and a Supra Comfy graph exists, the ordinary LoRA can use standard Load LoRA, with the strength on the diffusion model. That loader is not a substitute for the missing graph.

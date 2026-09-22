# Supra Concept Sliders

Product repo for **SupraLabs/Supra2-IMG** concept sliders, alongside
[Anima](https://github.com/HyperGAN/anima-particle-sliders) and
[Krea2](https://github.com/HyperGAN/krea2-particle-sliders).

## Final boss

A rank-16 LoRA trained on the same six final-boss subjects as the Krea2 run,
using short prose for Supra's 128-token context. Native 256px, 50 Euler steps,
CFG 3. The DiT is 104.1M parameters; the adapter trains 1.70M parameters.

```bash
bash scripts/train_final_boss_gpu1.sh
```

Weights and matched comparisons are written to `outputs/final-boss-supra/`.
The grid shows **Off → 0.5 → 1 → frozen positive teacher**, with the same seed.
The bridge guardian and fruit control are held out from training.

The completed local run produced a **6.8 MB** adapter in 400 updates: about
25 seconds for training/checkpointing and 99 seconds including loading,
teacher caching and verification on an RTX A6000. Sampling averaged **0.96 s
per 256px image**. The knight and cave warrior gain dark, spiked boss armor;
the held-out bridge effect is milder. Fruit remains recognizable with some
appearance changes. The local comparison is at `outputs/final-boss-supra/grid.png`.
Weights, generated images and logs are not included in this source checkout.

To continue the saved run until validation plateaus, use
`scripts/converge_final_boss.py`; the [recipe](docs/final-boss.md#continue-to-convergence)
documents the stopping rule and checkpoint selection.

See [the recipe, inference command and validation notes](docs/final-boss.md).
The adapter uses the included native Supra loader; ComfyUI loading is unverified.

The architecture and LoRA layers come from
[HyperGAN/particle-sliders](https://github.com/HyperGAN/particle-sliders), pinned
in [backend.lock.json](backend.lock.json). This repo supplies live text encoding,
training, checkpointing, VAE decoding and sample generation. Base weights stay
in the Hugging Face cache.

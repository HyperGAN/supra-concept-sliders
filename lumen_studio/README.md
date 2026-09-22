# Lumen studio

**TODO.** anima-concept-sliders ships a studio (`lumen_studio/`) that trains, mixes, and previews Anima particle sliders. That package is Anima-specific (Qwen text encoder, Anima transformer, the routed particle branch). It is not copied here.

Supra preview and training go through `train_lora_supra.py` in particle-sliders. See [REPRODUCE.md](../REPRODUCE.md). A Supra studio, if one is built, should call that trainer rather than re-implement the DiT.

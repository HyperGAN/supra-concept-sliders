# Reproduce the Supra catalog

This repo is the product side: prompt cards, model pin, and release identity. The DiT trainer is [HyperGAN/particle-sliders](https://github.com/HyperGAN/particle-sliders) pull request [130](https://github.com/HyperGAN/particle-sliders/pull/130), file `conceptmod/textsliders/train_lora_supra.py`. [`backend.lock.json`](backend.lock.json) pins the commit this scaffold was written against:

`03962e09e2770b240f254e7367086256549dea74` on branch `cursor/supra2-img-backend-aee7`.

Weights are not in either repo. A live run needs a local `model_final_ema.pt` from [SupraLabs/Supra2-IMG](https://huggingface.co/SupraLabs/Supra2-IMG) revision `10dec6e4b4b5d1c44fd1d7d3fe5e50137333da5b`. The dummy path never downloads it.

## Check out the backend

```bash
git clone https://github.com/HyperGAN/particle-sliders.git
cd particle-sliders
git fetch origin cursor/supra2-img-backend-aee7
git checkout 03962e09e2770b240f254e7367086256549dea74
export PARTICLE_SLIDERS_ROOT="$PWD"
```

Use that checkout's own environment for training. `requirements.txt` in this product repo is only PyYAML and pytest for catalog checks. It is not a CUDA lock.

Backend contract test, from the particle-sliders checkout:

```bash
PYTHONPATH=. pytest tests/test_supra_slider.py -q
```

## Catalog check (this repo)

```bash
python -m pip install -r requirements.txt
pytest -q
```

That test rebuilds the Sunlit prompts from `configs/supra/characters.json` and `configs/supra/definitions.json` and checks the YAML and JSON catalogs. It does not load Supra2-IMG.

## Dummy train

From this repo, with `PARTICLE_SLIDERS_ROOT` set:

```bash
DUMMY=1 ./scripts/train_supra.sh
```

That runs eight CPU steps on a tiny `SupraDiT` with the live LoRA suffixes. Prompts default to `data/prompts-supra.yaml`. `HF_HUB_OFFLINE=1` is set. The script never passes `--allow_hub`.

## Live train

Point `--checkpoint` at a local `model_final_ema.pt`. The trainer's live path attaches LoRA and trains the UNI card. Frozen Flan-T5 encode and VAE decode are not auto-wired by the backend PR; read `docs/supra-slider.md` in that checkout before expecting decoded images.

```bash
CHECKPOINT=/path/to/model_final_ema.pt DEVICE=cuda:0 ./scripts/train_supra.sh
```

The script expands to:

```bash
HF_HUB_OFFLINE=1 python conceptmod/textsliders/train_lora_supra.py \
  --name lighting-supra \
  --prompts_file /path/to/supra-concept-sliders/data/prompts-supra.yaml \
  --model_id SupraLabs/Supra2-IMG \
  --lora_targets cross --rank 16 --resolution 256 \
  --sample_steps 50 --cfg 3 \
  --lr 1e-4 --lm_target trajectory --traj_steps 4 \
  --sample_every 100 \
  --device cuda:0 --save_dir artifacts/lighting-supra \
  --checkpoint /path/to/model_final_ema.pt
```

Print the card and exit, from the backend checkout:

```bash
PYTHONPATH=. python conceptmod/textsliders/train_lora_supra.py --print_card
```

Keep dev prompts out of the train file. `data/prompts-supra-dev.yaml` is the held-out doorway row. `data/prompts-supra-canary.yaml` adds a moonlight minus line that the trainer does not teach.

## Samples, distill, Comfy

Not in this revision.

- Gallery contract: `data/release-samples.json` (seed 29001, 256², 50 Euler steps, CFG 3, Particle → Distill → Off). Image paths are null.
- Distill: [DISTILLATION.md](DISTILLATION.md).
- Comfy: [COMFYUI.md](COMFYUI.md).

`validation/` stores that status. It is not a passed release report.

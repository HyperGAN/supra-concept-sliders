# How the Supra sliders learn

Sunlit is a lighting slider for **SupraLabs/Supra2-IMG**. The base DiT, frozen Flan-T5-Base, and SD-VAE-FT-MSE stay frozen. The concept is neutral daylight to warm sun. Age is outside the catalog. Declared concept words (`warm`, `golden`, `sunlit`, `glow`) are not held. `attributes` are unused-token hold pins and are not caption prefixes.

The public card, once samples exist, will show a Particle adapter beside a distilled ordinary LoRA, the same Particle → Distill → Off story as [anima-concept-sliders](https://github.com/HyperGAN/anima-concept-sliders). That particle branch is not what the backend trains today. The opt-in trainer in particle-sliders pull request 130 follows the Anima trainer contract: UNI plus unused-token hold, on a LoRA. A later particle adapter should follow Anima's routed branch rather than a new adversarial recipe. This file records the card that `train_lora_supra.py` runs.

## Backbone

| field | value |
|---|---|
| hub id | `SupraLabs/Supra2-IMG` |
| checkpoint | `model_final_ema.pt` (local; not vendored) |
| params | 104,094,736 (~104.1M) |
| encoder | frozen `google/flan-t5-base`, context 128, `D_CTX=768` |
| VAE | `stabilityai/sd-vae-ft-mse`, scale `0.18215` |
| resolution | 256², latent 32², patch 2 |
| DiT | `D_MODEL=576`, `DEPTH=14`, `N_HEADS=9`, `HEAD_DIM=64`, `MLP_RATIO=4` |
| sampler | Euler flow, `t = i/K`, `z <- z + (1/K) * v` |
| sample CFG | 3.0, 50 steps: `v_u + cfg * (v_c - v_u)` |

Train-time UNI uses the unscaled direction `v(z, t, c) - v(z, t, '')`. The Euler grid runs `t` from 0 toward 1. That is the Hub schedule, not Anima's FlowMatch σ schedule.

## LoRA

Flan-T5 has no separate text conditioner. Caption signal enters through `ctx_proj` and cross-attention. `proj` exists on both self-attention and cross-attention, so targets are path suffixes.

| `--lora_targets` | trained | frozen |
|---|---|---|
| `cross` (default) | `ctx_proj`, `cross_attn.q`, `cross_attn.kv`, `cross_attn.proj` | Flan-T5, VAE, self-attn |
| `dit` | `self_attn.qkv`, `self_attn.proj` | Flan-T5, VAE, cross-attn, `ctx_proj` |
| `dit+cross` | both | Flan-T5, VAE |

Rank 16, alpha 16. The product card uses `cross`.

## UNI and unused-token hold

Same contract as the Anima trainer, on this backbone:

- student **+1** stays on the neutral / infer caption
- the **+** caption is the teacher only
- scale **0** stays on the neutral caption
- pinned `attributes` are unused-token hold bookkeeping
- `concept_words` are not held
- a minus caption is a canary only (`data/prompts-supra-canary.yaml`)

| scale | student | `--lm_target trajectory` (default) |
|---|---|---|
| **+1** | infer / neutral | K-step Hub Euler of the frozen **plus** prompt from the same `z_0` |
| **0** | infer / neutral | light identity against the frozen **neutral** trajectory (`--traj_identity_weight`, default 0.25) |
| **−1** | unscored canary | — |

```
MSE(x_student, x_plus) + λ_id * MSE(x_zero, x_neu)
```

`--lm_target direct` and `--lm_target cfg_delta` are the one-step recipes. `--teacher_gap_boost` (default 1, off) applies to those only. Anima `embed_struct` / `same_crop` and Music 3 `v9` are rejected by this trainer.

## Catalog

Captions are `{character}, {lighting}` from [`configs/supra/characters.json`](configs/supra/characters.json) and [`configs/supra/definitions.json`](configs/supra/definitions.json). Train characters cross train definitions into [`data/prompts-supra.yaml`](data/prompts-supra.yaml). The eval definition and the dev character stay in [`data/prompts-supra-dev.yaml`](data/prompts-supra-dev.yaml).

The first two train rows are the woman-at-a-window and man-at-a-table lines shipped with the backend stub. Two more train rows paraphrase the warm pole (`amber sunlight, warm glow`). One dev row paraphrases again (`honey-colored sun across the floor`) and is the reserved gallery prompt.

## Live card

| Setting | Value |
|---|---|
| Name | `lighting-supra` |
| Steps | 500 (dummy CI uses 8) |
| Adam learning rate | `1e-4` |
| Seed | 7 |
| Hold weight | 1 |
| Trajectory steps | 4 |
| Identity weight | 0.25 |
| Sample every | 100 updates |
| Sample scales | 0, 0.25, 0.5, 1 |
| Sample seed | 42 |
| Control prompt | a bowl of fruit on a table |
| Resolution | 256 × 256 |

These are the backend defaults, not a finished release recipe. No checkpoint has been selected. Strengths above the trained endpoint are unspecified until a gallery measures them.

Authoritative write-up of the trainer: [docs/supra-slider.md on particle-sliders PR 130](https://github.com/HyperGAN/particle-sliders/blob/cursor/supra2-img-backend-aee7/docs/supra-slider.md).

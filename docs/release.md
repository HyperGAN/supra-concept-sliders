# Reproduce the Final Boss release

Install a CUDA-enabled PyTorch build and `pip install -r requirements.txt`.
That file pins `particle-sliders-core` from HyperGAN/particle-sliders at
`4340e28bed388d50800c469525b460a108091da0`. The product imports
`particle_sliders.winning_formulation`. It does not install
`concept-slider-core` and it does not use `PARTICLE_SLIDERS_ROOT`.

The DiT used by this ordinary-LoRA release is the in-repo file
`vendor/particle-sliders/conceptmod/textsliders/supra_model.py`, with its
licenses. The runtime checks that file against `dit_backbone.sha256` in
`backend.lock.json` before loading. The concept-slider train entry is
`scripts/train_lora_supra.py`. See [shared-core.md](shared-core.md).

The measured machine used physical GPU 1, an RTX A6000, eight CPU threads,
bf16 forward computation and fp32 parameters. Exact measured package versions
and CUDA version are in the public `evidence/benchmark-1600.json` file.
The base weights were already downloaded; network download time is excluded.
Another workload used GPU 0 during measurement. Timings vary by environment.

## Train 1,600 steps

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_final_boss.py --allow-hub \
  --steps 1600 --save-every 400 --out outputs/final-boss-supra-1600
```

This starts a fresh adapter, creates all teacher targets, trains, saves four
checkpoints and an early preview, checks strength zero and reload, evaluates
fresh-seed velocities, and renders 32 final comparison images. The reported
1,600-step headline includes checkpoints and the early preview. The complete
run includes preparation and final verification. Each optimizer update and
the outer training timer synchronize CUDA. The evidence includes all 1,600
update timing records, not an extrapolation from a shorter run.

## Continue to validation convergence

The released converged checkpoint comes from the earlier seed-7, 400-step run
and its optimizer/RNG continuation. Reproduce that chain with:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_final_boss.py --allow-hub \
  --steps 400 --save-every 50 --out outputs/final-boss-supra
CUDA_VISIBLE_DEVICES=1 python scripts/converge_final_boss.py
```

Both runs use the same prompt recipe and rank-16 architecture. The convergence
run checks independent seeds every 400 steps, reduces the learning rate after
plateaus, stops at the minimum-rate plateau, and exports the best checkpoint.
It selected 28,000 and stopped at 28,800. See the complete selection rule in
[the training recipe](final-boss.md#continue-to-convergence). The 1,600-step
benchmark is a separate run and is not described as converged.

## Fit both distills and build comparisons

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/distill_final_boss.py \
  --teacher outputs/final-boss-supra-1600/final-boss-supra.safetensors \
  --out outputs/final-boss-distill-1600
CUDA_VISIBLE_DEVICES=1 python scripts/distill_final_boss.py \
  --teacher outputs/final-boss-supra-converged/final-boss-supra.safetensors \
  --out outputs/final-boss-distill-converged
pip install matplotlib
python scripts/plot_convergence.py outputs/final-boss-supra-converged
python scripts/build_release.py
python -m pytest -q tests
```

The staging script writes `artifacts/release-v0.1.0` and regenerates the GitHub
README from actual measurements. It copies four adapters, sidecar metadata,
48 sample PNGs with sidecars, comparison grids, the convergence chart, prompt
recipe, benchmark evidence and distillation reports. Generated files stay
outside Git; source is committed separately. There is no ComfyUI export claim.

After committing source, `python scripts/publish_release.py` validates the
staged files and creates source provenance, a source archive and SHA-256
manifest. `--publish` additionally uploads to the model repository and verifies
every remote file hash plus downloads all four weights for readback. Existing
remote content is preserved; an update requires `--expected-parent` matching
the inspected Hub commit. The source archive includes the pinned backend and
its notices but no base weights, caches, optimizer state or local environment.

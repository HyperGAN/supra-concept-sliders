"""Supra2-IMG train entry. The game is ``winning_formulation()``.

This module owns prompts, the Euler sample card, Hub ids, and the optimizer
step sizes on the stamp's model surface. It does not define routed particles
or a gradient penalty. Those come from ``stamp.bridge()``,
``stamp.regularizer()``, and ``stamp.losses()``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import torch
import yaml

from particle_sliders import FormulationGame, dummy_features, run_formulation_game

from supra.card import (
    BASE_MODEL_ID,
    CFG,
    COMFY_CLASS,
    CORE_COMMIT,
    EULER_SCHEDULE,
    PRODUCT_HUB_ID,
    REFERENCE_ITERATIONS,
    RELEASE_HUB_ID,
    RESOLUTION,
    SAMPLE_STEPS,
    SUPRA_ADV_BATCH,
    SUPRA_G_LR,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = ROOT / "configs/supra/prompts-supra.yaml"
_FOREIGN_BACKENDS = ("anima", "krea", "sana", "z-image", "zimage", "zit", "minimax")


def assert_supra_only(model_id: str) -> None:
    lowered = str(model_id).lower()
    for name in _FOREIGN_BACKENDS:
        if name in lowered:
            raise ValueError(f"this trainer is Supra-only; refused foreign backend {name!r}")


def load_prompts(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = yaml.safe_load(Path(path).read_text())
    rows = list(data.get("rows") or [])
    if not rows:
        raise ValueError(f"no prompt rows in {path}")
    for row in rows:
        for key in ("neutral", "positive"):
            if not str(row.get(key, "")).strip():
                raise ValueError(f"prompt row missing {key}")
    meta = {key: data[key] for key in ("plus_label", "concept_words", "recommended_range") if key in data}
    return rows, meta


def load_stamp():
    """Import the shared game and lock this product's surface against it."""
    from particle_sliders import winning_formulation

    stamp = winning_formulation()
    declared = {**stamp.as_dict(), "g_lr": SUPRA_G_LR, "adv_batch": SUPRA_ADV_BATCH}
    stamp.require(declared)
    return stamp, declared


def euler_card() -> dict[str, Any]:
    return {
        "resolution": RESOLUTION,
        "sample_steps": SAMPLE_STEPS,
        "cfg": CFG,
        "schedule": EULER_SCHEDULE,
        "sampler": "euler_flow",
    }


def product_card(stamp, declared: dict[str, Any], prompts_file: Path, *, steps: int, dummy: bool) -> dict[str, Any]:
    rows, meta = load_prompts(prompts_file)
    return {
        "name": "lighting-supra",
        "model_id": BASE_MODEL_ID,
        "product_hub_id": PRODUCT_HUB_ID,
        "release_hub_id": RELEASE_HUB_ID,
        "comfy_class": COMFY_CLASS,
        "core_commit": CORE_COMMIT,
        "core_import": "particle_sliders.winning_formulation",
        "architecture_id": stamp.architecture_id,
        "formulation_id": stamp.formulation_id,
        "formulation_provisional": stamp.formulation_provisional,
        "g_lr": declared["g_lr"],
        "d_lr": declared["d_lr"],
        "particle_lr": declared["particle_lr"],
        "adv_batch": declared["adv_batch"],
        "reference_iterations": REFERENCE_ITERATIONS,
        "steps": int(steps),
        "dummy": bool(dummy),
        "prompts_file": str(prompts_file),
        "prompt_rows": len(rows),
        "plus_label": meta.get("plus_label", ""),
        **euler_card(),
    }


# Shared D/G/VIC/optimizer step lives in particle-sliders-core.
# Back-compat aliases for tests/callers that still say SupraGame / run_game.
SupraGame = FormulationGame
run_game = run_formulation_game
_dummy_features = dummy_features


def live_velocity_features(runtime, rows: list[dict[str, Any]], rank: int, *, cfg: float = CFG):
    """Project a frozen Euler velocity edit into the stamp rank.

    The projection is product glue. The particle map applied to it is
    ``stamp.bridge()``, constructed by the caller.
    """
    projector = torch.nn.Linear(4, rank)

    def draw(step: int, batch: int) -> torch.Tensor:
        row = rows[(step - 1) % len(rows)]
        z = runtime.noise(1000 + step, n=1)
        t = ((step - 1) % SAMPLE_STEPS) / SAMPLE_STEPS
        with torch.no_grad():
            neutral = runtime.velocity(row["neutral"], z, t, scale=0.0, cfg=cfg)
            positive = runtime.velocity(row["positive"], z, t, scale=0.0, cfg=cfg)
        pooled = (positive - neutral).mean(dim=(2, 3))
        edit = projector(pooled.to(dtype=projector.weight.dtype))
        return edit.expand(batch, -1).contiguous()

    return projector, draw


    for step in range(1, int(steps) + 1):
        history.append(game.step(step, features_for_step(step)))
    if not history or not math.isfinite(history[-1]["g_loss"]):
        raise RuntimeError("Supra game produced no finite generator loss")
    return history


def train_dummy(prompts_file: Path, steps: int, seed: int, save_dir: Path) -> dict[str, Any]:
    stamp, declared = load_stamp()
    card = product_card(stamp, declared, prompts_file, steps=steps, dummy=True)
    batch = int(declared["adv_batch"])
    game = SupraGame(stamp, declared, seed=seed)
    history = run_game(game, steps, _dummy_features(game.rank, batch, seed))
    sidecar = {**card, "loss_last": history[-1], "history_tail": history[-min(4, len(history)):]}
    _write_sidecar(save_dir, sidecar)
    return sidecar


def train_live(prompts_file: Path, steps: int, seed: int, save_dir: Path, device: str, allow_hub: bool) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("live Supra2-IMG needs CUDA. Pass --dummy for the CPU game smoke.")
    from supra.runtime import SupraRuntime

    stamp, declared = load_stamp()
    rows, _meta = load_prompts(prompts_file)
    runtime = SupraRuntime(device, rank=int(stamp.spec["adapter_rank"]), allow_hub=allow_hub)
    projector, draw = live_velocity_features(runtime, rows, int(stamp.spec["adapter_rank"]))
    projector.to(runtime.device)
    batch = int(declared["adv_batch"])
    game = SupraGame(
        stamp, declared, seed=seed, device=runtime.device,
        extra_generator=list(projector.parameters()),
    )

    def features(step: int) -> torch.Tensor:
        return draw(step, batch)

    card = product_card(stamp, declared, prompts_file, steps=steps, dummy=False)
    history = run_game(game, steps, features)
    sidecar = {**card, "loss_last": history[-1], "device": str(runtime.device)}
    _write_sidecar(save_dir, sidecar)
    return sidecar


def _write_sidecar(save_dir: Path, sidecar: dict[str, Any]) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "supra-game.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    (save_dir / "euler-card.json").write_text(json.dumps(euler_card(), indent=2) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Supra2-IMG slider on the shared particle game.")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--model-id", default=BASE_MODEL_ID)
    parser.add_argument("--steps", type=int, default=REFERENCE_ITERATIONS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/lighting-supra"))
    parser.add_argument("--dummy", action="store_true", help="CPU game smoke. Does not load Supra2-IMG weights.")
    parser.add_argument("--allow-hub", action="store_true")
    parser.add_argument("--print-card", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    assert_supra_only(args.model_id)
    if args.model_id != BASE_MODEL_ID:
        raise ValueError(f"Supra2-IMG train entry only loads {BASE_MODEL_ID}")
    if int(args.steps) < 1:
        raise ValueError("steps must be positive")
    stamp, declared = load_stamp()
    if args.print_card:
        card = product_card(stamp, declared, args.prompts, steps=args.steps, dummy=args.dummy)
        print(json.dumps(card, indent=2))
        return card
    if args.dummy:
        sidecar = train_dummy(args.prompts, int(args.steps), int(args.seed), args.save_dir)
        print(json.dumps({key: sidecar[key] for key in ("formulation_id", "product_hub_id", "loss_last", "schedule")}, indent=2))
        return sidecar
    return train_live(args.prompts, int(args.steps), int(args.seed), args.save_dir, args.device, args.allow_hub)

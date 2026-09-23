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


class SupraGame:
    """One optimizer step of the shared game. Width is the stamp's adapter rank."""

    def __init__(self, stamp, declared: dict[str, Any], *, seed: int = 7,
                 device: str | torch.device = "cpu", extra_generator: list | None = None):
        torch.manual_seed(seed)
        self.stamp = stamp
        self.declared = declared
        self.device = torch.device(device)
        self.regularizer = stamp.regularizer()
        self.d_loss, self.g_loss, self.vic = stamp.losses()
        self.rank = int(stamp.spec["adapter_rank"])
        self.bridge = stamp.bridge().to(self.device)
        parts = int(stamp.spec["parts"])
        particle_dim = int(stamp.spec["particle_dim"])
        self.particles = torch.nn.Parameter(torch.randn(parts, particle_dim, device=self.device) * 0.02)
        bank = _paired_bank(self.rank, seed)
        self.critic = stamp.critic(bank["targets"].to(self.device), neutrals=bank["neutrals"].to(self.device))
        betas = tuple(float(x) for x in stamp.spec["betas"])
        generator_params = list(self.bridge.parameters()) + list(extra_generator or [])
        self.opt_g = torch.optim.Adam(
            [
                {"params": generator_params, "lr": float(declared["g_lr"])},
                {"params": [self.particles], "lr": float(declared["particle_lr"])},
            ],
            betas=betas,
        )
        self.opt_d = torch.optim.Adam(self.critic.parameters(), lr=float(declared["d_lr"]), betas=betas)
        self.edit_rms = float(self.critic.edit_rms)

    def step(self, index: int, features: torch.Tensor) -> dict[str, float]:
        if index < 1:
            raise ValueError("steps are numbered starting at 1")
        if features.ndim != 2 or features.shape[1] != self.rank:
            raise ValueError(f"features must be [batch, {self.rank}]")
        features = features.to(self.device)
        sigma = float(self.stamp.noise_std_at(index - 1, self.edit_rms))
        noise = torch.randn(features.shape[0], self.rank, device=self.device) * sigma
        self.critic.requires_grad_(True)
        self.opt_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake_detached = noise + self.bridge(features, self.particles)
        adv_d = self.d_loss(self.critic(noise), self.critic(fake_detached))
        penalty = self.regularizer(self.critic, noise, fake_detached, step=index)
        loss_d = adv_d + penalty
        if not torch.isfinite(loss_d):
            raise FloatingPointError(f"non-finite critic loss at step {index}")
        loss_d.backward()
        self.opt_d.step()

        self.critic.requires_grad_(False)
        self.opt_g.zero_grad(set_to_none=True)
        fake = noise.detach() + self.bridge(features, self.particles)
        with torch.no_grad():
            real_score = self.critic(noise.detach())
        adv_g = self.g_loss(real_score, self.critic(fake))
        take = int(self.stamp.spec["particle_vic_batch"])
        choice = torch.randperm(self.particles.shape[0], device=self.device)[:take]
        vic = self.vic(self.particles[choice])
        loss_g = adv_g + vic
        if not torch.isfinite(loss_g):
            raise FloatingPointError(f"non-finite generator loss at step {index}")
        loss_g.backward()
        self.opt_g.step()
        return {
            "step": float(index),
            "d_loss": float(loss_d.detach()),
            "g_loss": float(adv_g.detach()),
            "vic": float(vic.detach()),
            "sigma": sigma,
        }


def _paired_bank(rank: int, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed + 17)
    neutrals = torch.randn(32, rank, generator=generator)
    targets = neutrals + torch.randn(32, rank, generator=generator)
    return {"targets": targets, "neutrals": neutrals}


def _dummy_features(rank: int, batch: int, seed: int) -> Callable[[int], torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)

    def draw(step: int) -> torch.Tensor:
        del step
        return torch.randn(batch, rank, generator=generator)

    return draw


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


def run_game(game: SupraGame, steps: int, features_for_step: Callable[[int], torch.Tensor]) -> list[dict[str, float]]:
    history = []
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

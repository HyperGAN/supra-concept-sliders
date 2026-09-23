"""CPU smoke for the in-repo Supra train entry and the core pin."""
import json
from pathlib import Path

import pytest
import torch

from supra.card import CORE_COMMIT, CORE_REQUIREMENT, PRODUCT_HUB_ID, SAMPLE_STEPS
from supra.train import live_velocity_features, load_stamp, main

ROOT = Path(__file__).resolve().parents[1]


def test_requirements_and_lock_pin_particle_sliders_core():
    requirements = (ROOT / "requirements.txt").read_text()
    assert CORE_REQUIREMENT in requirements
    assert "concept-slider-core" not in requirements
    assert "concept_slider_core" not in requirements
    lock = json.loads((ROOT / "backend.lock.json").read_text())
    assert lock["package"] == "particle-sliders-core"
    assert lock["commit"] == CORE_COMMIT
    assert lock["import"] == "particle_sliders"
    assert lock["entrypoint"] == "winning_formulation"
    assert "concept-slider-core" not in lock["package"]
    dit = ROOT / lock["dit_backbone"]["path"]
    assert dit.is_file()
    import hashlib
    assert hashlib.sha256(dit.read_bytes()).hexdigest() == lock["dit_backbone"]["sha256"]


def test_product_sources_do_not_define_the_game():
    files = [ROOT / "supra/train.py", ROOT / "supra/card.py", ROOT / "scripts/train_lora_supra.py"]
    text = "\n".join(path.read_text() for path in files)
    assert "class RoutedMLP" not in text
    assert "class GradRegularizer" not in text
    assert "concept_slider_core" not in text
    assert "winning_formulation" in text


def test_require_locks_formulation_and_keeps_supra_surface():
    stamp, declared = load_stamp()
    assert stamp.formulation_id == "particle-gmix-1600-v2"
    assert stamp.architecture_id == "gmix"
    assert declared["g_lr"] == pytest.approx(1e-4)
    assert declared["adv_batch"] == 4
    assert stamp.spec["aux_weights"]["cover_weight"] == 0.0
    assert stamp.regularizer().__class__.__module__.startswith("particlegan")
    d_loss, g_loss, vic = stamp.losses()
    assert d_loss.__module__.startswith("particle_sliders")
    assert g_loss.__module__.startswith("particle_sliders")
    assert vic.__module__.startswith("particle_sliders")
    forked = {**stamp.as_dict(), "g_lr": 1e-4, "adv_batch": 4, "cover_weight": 1.0}
    with pytest.raises(ValueError, match="unknown"):
        stamp.require(forked)


def test_dummy_entry_runs_the_game(tmp_path):
    sidecar = main(["--dummy", "--steps", "4", "--device", "cpu", "--save-dir", str(tmp_path)])
    assert sidecar["product_hub_id"] == PRODUCT_HUB_ID
    assert sidecar["formulation_id"] == "particle-gmix-1600-v2"
    assert sidecar["sample_steps"] == SAMPLE_STEPS
    assert sidecar["schedule"].startswith("t=i/K")
    assert sidecar["comfy_class"] is None
    assert sidecar["core_commit"] == CORE_COMMIT
    assert torch.isfinite(torch.tensor(sidecar["loss_last"]["g_loss"]))
    assert (tmp_path / "supra-game.json").is_file()
    assert json.loads((tmp_path / "euler-card.json").read_text())["cfg"] == 3.0


def test_print_card_and_foreign_model_id(capsys):
    card = main(["--print-card", "--steps", "2"])
    assert card["model_id"] == "SupraLabs/Supra2-IMG"
    assert "particle-gmix-1600-v2" in capsys.readouterr().out
    with pytest.raises(ValueError, match="Supra-only"):
        main(["--dummy", "--steps", "1", "--model-id", "anima-beta"])


class _FakeRuntime:
    def noise(self, seed, n=1):
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(n, 4, 8, 8, generator=generator)

    def velocity(self, prompt, z, t, scale=0.0, cfg=3.0):
        del t, scale, cfg
        shift = 0.4 if "sunlit" in prompt else 0.0
        return torch.ones_like(z) * shift


def test_live_feature_hook_uses_euler_velocity_edit():
    rows = [
        {"neutral": "neutral daylight", "positive": "warm golden sunlit glow"},
    ]
    projector, draw = live_velocity_features(_FakeRuntime(), rows, rank=8, cfg=3.0)
    features = draw(1, 4)
    assert features.shape == (4, 8)
    assert projector.weight.requires_grad
    assert torch.isfinite(features).all()

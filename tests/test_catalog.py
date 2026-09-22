"""Catalog consistency for the Supra2-IMG product scaffold.

These checks do not load the DiT. Training lives in particle-sliders.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

AGE_WORDS = re.compile(
    r"\b(age|ages|aged|child|children|teen|teenager|young|old|older|elderly|baby|infant|minor)\b",
    re.IGNORECASE,
)
CONCEPT_WORDS = ("warm", "golden", "sunlit", "glow")


def _load_json(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _load_yaml(path: str):
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def _caption(character: dict, definition: dict, pole: str) -> str:
    lighting = definition["lighting"] if pole == "positive" else definition["neutral_lighting"]
    return f"{character['caption']}, {lighting}"


def _crossed(split_characters: str, split_definitions: str) -> list[dict]:
    characters = [c for c in _load_json("configs/supra/characters.json") if c["split"] == split_characters]
    definitions = [d for d in _load_json("configs/supra/definitions.json") if d["split"] == split_definitions]
    rows = []
    for definition in definitions:
        for character in characters:
            rows.append(
                {
                    "character": character,
                    "definition": definition,
                    "neutral": _caption(character, definition, "neutral"),
                    "positive": _caption(character, definition, "positive"),
                }
            )
    return rows


def test_model_lock_matches_hub_card():
    lock = _load_json("configs/supra/model.lock.json")
    assert lock["model"] == "SupraLabs/Supra2-IMG"
    assert lock["revision"] == "10dec6e4b4b5d1c44fd1d7d3fe5e50137333da5b"
    assert lock["params"] == 104094736
    assert lock["encoder"]["repo"] == "google/flan-t5-base"
    assert lock["encoder"]["frozen"] is True
    assert lock["encoder"]["ctx_len"] == 128
    assert lock["vae"]["scale"] == 0.18215
    assert lock["arch"]["resolution"] == 256
    assert lock["arch"]["latent_size"] == 32
    assert lock["arch"]["patch"] == 2
    assert lock["arch"]["d_model"] == 576
    assert lock["arch"]["depth"] == 14
    assert lock["defaults"]["steps"] == 50
    assert lock["defaults"]["cfg"] == 3.0
    assert lock["defaults"]["sampler"] == "euler_flow"
    assert lock["weights_vendored"] is False
    assert lock["checkpoint_sha256"] is None


def test_backend_pin_points_at_particle_sliders_trainer():
    pin = _load_json("backend.lock.json")
    assert pin["pull_request"] == 130
    assert pin["commit"] == "03962e09e2770b240f254e7367086256549dea74"
    assert pin["trainer"] == "conceptmod/textsliders/train_lora_supra.py"
    assert "train_lora_supra.py" in (ROOT / "REPRODUCE.md").read_text(encoding="utf-8")
    trainer_sources = list(ROOT.rglob("train_lora_supra.py"))
    assert trainer_sources == []


def test_train_and_dev_catalogs_match_cards():
    train_rows = _crossed("train", "train")
    dev_rows = _crossed("dev", "eval")
    assert [row["definition"]["id"] for row in train_rows] == [
        "sunlit-01",
        "sunlit-01",
        "sunlit-02",
        "sunlit-02",
    ]
    prompts = _load_yaml("data/prompts-supra.yaml")
    dev_prompts = _load_yaml("data/prompts-supra-dev.yaml")
    train_json = _load_json("data/train.json")
    dev_json = _load_json("data/dev.json")

    assert prompts["plus_label"] == "sunlit"
    assert prompts["concept_words"] == "warm, golden, sunlit, glow"
    assert len(prompts["rows"]) == len(train_rows) == len(train_json["rows"])
    assert len(dev_prompts["rows"]) == len(dev_rows) == len(dev_json["rows"])

    for expected, yaml_row, json_row in zip(train_rows, prompts["rows"], train_json["rows"]):
        assert yaml_row["target"] == yaml_row["neutral"] == expected["neutral"]
        assert yaml_row["positive"] == expected["positive"]
        assert json_row["neutral"] == expected["neutral"]
        assert json_row["positive"] == expected["positive"]
        assert json_row["target"] == expected["neutral"]
        assert yaml_row["attributes"] == expected["character"]["attributes"]
        assert yaml_row["resolution"] == 256
        assert yaml_row["guidance_scale"] == 3.0
        assert "negative" not in yaml_row

    dev_expected = dev_rows[0]
    assert dev_prompts["rows"][0]["neutral"] == dev_expected["neutral"]
    assert dev_json["rows"][0]["positive"] == dev_expected["positive"]
    assert dev_expected["neutral"] not in {row["neutral"] for row in prompts["rows"]}


def test_concept_words_are_not_attributes_and_age_is_absent():
    blob = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in (
            "data/prompts-supra.yaml",
            "data/prompts-supra-dev.yaml",
            "data/prompts-supra-canary.yaml",
            "configs/supra/definitions.json",
            "configs/supra/characters.json",
            "configs/supra/candidates/candlelit-v1.json",
        )
    )
    assert AGE_WORDS.search(blob) is None
    prompts = _load_yaml("data/prompts-supra.yaml")
    for row in prompts["rows"]:
        assert not set(CONCEPT_WORDS) & set(row["attributes"])
        neutral_tokens = set(re.findall(r"[a-z]+", row["neutral"].lower()))
        assert not set(CONCEPT_WORDS) & neutral_tokens
    assert "sunlit" in prompts["rows"][0]["positive"]
    assert "warm" in prompts["rows"][2]["positive"]


def test_canary_minus_is_not_the_train_file():
    canary = _load_yaml("data/prompts-supra-canary.yaml")
    assert canary["minus_label"] == "moonlight"
    assert "cold blue moonlight" in canary["rows"][0]["negative"]
    train = _load_json("data/train.json")
    assert all("negative" not in row for row in train["rows"])


def test_release_samples_are_reserved_and_empty():
    samples = _load_json("data/release-samples.json")
    assert samples["status"] == "pending"
    assert samples["comparison"] == ["particle", "distill", "off"]
    assert samples["width"] == 256
    assert samples["steps"] == 50
    assert samples["cfg"] == 3.0
    assert samples["seed"] == 29001
    case = samples["cases"][0]
    assert case["images"] is None
    assert case["prompt"] == _load_json("data/dev.json")["rows"][0]["neutral"]
    identity = _load_json("configs/supra/released-identity.json")
    assert identity["status"] == "scaffolding"
    assert identity["sliders"]["sunlit"]["particle_sha256"] is None


def test_readme_leads_with_scaffold_and_particle_story():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    head = "\n".join(readme.splitlines()[:8])
    assert "Scaffolding" in head
    assert "particle-sliders" in readme
    assert "pull/130" in readme or "#130" in readme or "pull request 130" in readme.lower() or "130" in head
    assert "Particle → Distill → Off" in readme
    assert "not released" in readme
    distill = (ROOT / "DISTILLATION.md").read_text(encoding="utf-8")
    comfy = (ROOT / "COMFYUI.md").read_text(encoding="utf-8")
    assert "TODO" in distill
    assert "TODO" in comfy


def test_comfy_stub_registers_no_nodes():
    import comfy_particle

    assert comfy_particle.NODE_CLASS_MAPPINGS == {}
    assert comfy_particle.NODE_DISPLAY_NAME_MAPPINGS == {}
    text = (ROOT / "comfy_particle.py").read_text(encoding="utf-8")
    assert "TODO" in text
    assert "import torch" not in text


def test_train_script_calls_backend_and_stays_offline():
    script = (ROOT / "scripts" / "train_supra.sh").read_text(encoding="utf-8")
    assert "train_lora_supra.py" in script
    assert "HF_HUB_OFFLINE" in script
    assert "--allow_hub" not in script
    assert "data/prompts-supra.yaml" in script
    assert (ROOT / "lumen_studio" / "README.md").read_text(encoding="utf-8").startswith("# Lumen studio")

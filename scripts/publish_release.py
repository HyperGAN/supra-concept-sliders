#!/usr/bin/env python3
"""Validate and optionally publish an immutable, checksummed HF model release."""
import argparse
import hashlib
import json
import math
import re
import subprocess
import zipfile
from pathlib import Path
from huggingface_hub import HfApi, ModelCard, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
REPO = "ntc-ai/supra-concept-sliders"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def files(folder):
    return [p for p in sorted(folder.rglob("*")) if p.is_file()
            and p.relative_to(folder).parts[0] not in (".cache", ".git", ".gitattributes")]


def write(path, data):
    path.write_text(json.dumps(data, indent=2)+"\n")


def validate(folder):
    import torch
    from PIL import Image
    from safetensors import safe_open
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    catalog = read(folder/"catalog.json")
    assert len(catalog["sliders"]) == 2
    benchmark = read(folder/"evidence/benchmark-1600.json")
    updates = [json.loads(line) for line in (folder/"evidence/benchmark-1600-updates.jsonl").read_text().splitlines()]
    assert [r["step"] for r in updates] == list(range(1,1601))
    assert abs(sum(r["seconds"] for r in updates)-benchmark["optimizer_update_seconds"]) < 1e-6
    assert benchmark["optimizer_update_seconds"] <= benchmark["training_seconds"] <= benchmark["total_wall_seconds"]
    for entry in catalog["sliders"]:
        report = read(folder/entry["evaluation"])
        assert all(report["checks"].values())
        assert report["teacher_sha256"] == entry["original_sha256"]
        assert report["adapter_sha256"] == entry["distill_sha256"]
        for kind, rank in (("original",16), ("distill",8)):
            path = folder/entry[kind]
            assert digest(path) == entry[kind+"_sha256"]
            with safe_open(path, framework="pt") as handle:
                meta = {k:json.loads(v) for k,v in handle.metadata().items()}
            assert meta["format"] == "supra-native-lora-v1" and meta["rank"] == meta["alpha"] == rank
            assert meta["model_revision"] == catalog["model_revision"]
            state = load_file(str(path))
            assert len(state) == 142
            assert sum(v.numel() for v in state.values()) == (1698816 if rank == 16 else 849408)
            assert all(torch.isfinite(t).all() for t in state.values())
        assert len(entry["samples"]) == 24
        groups = {}
        for item in entry["samples"]:
            path = folder/item["image"]
            assert Image.open(path).size == (256,256)
            meta = read(folder/item["metadata"])
            assert meta["prompt"] == item["prompt"] and meta["seed"] == item["seed"]
            expected = None if item["variant"] == "off" else entry[item["variant"]+"_sha256"]
            assert meta["adapter_sha256"] == expected
            groups.setdefault((item["name"],item["seed"]), []).append(meta)
        assert len(groups) == 8
        for group in groups.values():
            assert {r["variant"] for r in group} == {"original","distill","off"}
            for key in ("prompt","seed","steps","cfg","model_revision"):
                assert len({r[key] for r in group}) == 1
        assert math.isfinite(report["heldout_projection_relative_mse"])
    card = (folder/"README.md").read_text()
    for path in re.findall(r"https://huggingface.co/ntc-ai/supra-concept-sliders/resolve/main/([^\s)]+)", card):
        # These provenance files are produced after the initial content validation.
        assert (folder/path).is_file() or path in ("release-manifest.json","source-provenance.json"),path
    return catalog


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder", type=Path, default=ROOT/"artifacts/release-v0.1.0")
    p.add_argument("--publish", action="store_true")
    p.add_argument("--expected-parent")
    args = p.parse_args()
    folder = args.folder.resolve()
    catalog = validate(folder)
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip(), "Commit source first"
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    tracked = [s for s in subprocess.check_output(["git","ls-files","-z"],cwd=ROOT).decode().split("\0") if s]
    write(folder/"source-provenance.json", dict(repository="https://github.com/HyperGAN/supra-concept-sliders",
        commit=commit, backend=read(ROOT/"backend.lock.json"),
        files={name:dict(sha256=digest(ROOT/name),bytes=(ROOT/name).stat().st_size) for name in tracked}))
    with zipfile.ZipFile(folder/"source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in tracked:
            archive.write(ROOT/name,"supra-concept-sliders/"+name)
    write(folder/"release-manifest.json", dict(source_commit=commit,
        files={str(path.relative_to(folder)):dict(bytes=path.stat().st_size,sha256=digest(path))
               for path in files(folder) if path.name != "release-manifest.json"}))
    ModelCard.load(folder/"README.md").validate()
    print("Validated", len(files(folder)), "files; source", commit, flush=True)
    if not args.publish:
        return
    api = HfApi()
    api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
    before = api.model_info(REPO, files_metadata=True)
    existing = {r.rfilename:r for r in before.siblings if r.rfilename != ".gitattributes"}
    if existing:
        assert before.sha == args.expected_parent, "Remote content exists; inspect and supply expected parent"
        for name, remote in existing.items():
            path = folder/name
            assert path.is_file(), ("Unrecognized remote file",name)
            if name.startswith(("weights/","distilled/","samples/","assets/","evidence/")):
                assert remote.size == path.stat().st_size, name
                if remote.lfs:
                    assert remote.lfs.sha256 == digest(path), name
                else:
                    data=path.read_bytes()
                    assert remote.blob_id == hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest(), name
    result = api.upload_folder(repo_id=REPO, repo_type="model", folder_path=folder,
        parent_commit=before.sha, ignore_patterns=[".cache/**",".git/**",".gitattributes"],
        commit_message="Publish Final Boss: measured 1600-step LoRA, converged LoRA, rank-8 distills and matched samples")
    remote = {r.rfilename:r for r in api.model_info(REPO,revision=result.oid,files_metadata=True).siblings}
    for path in files(folder):
        name = str(path.relative_to(folder))
        r = remote[name]
        assert r.size == path.stat().st_size, name
        if r.lfs:
            assert r.lfs.sha256 == digest(path),name
        else:
            data = path.read_bytes()
            assert r.blob_id == hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest(),name
    readbacks = ["README.md"]+[entry[k] for entry in catalog["sliders"] for k in ("original","distill")]
    for name in readbacks:
        path = Path(hf_hub_download(REPO,name,revision=result.oid))
        assert digest(path) == digest(folder/name),name
    report = dict(repo=REPO,commit=result.oid,source_commit=commit,verified_files=len(files(folder)),
        readbacks=len(readbacks),previous_commit=before.sha)
    write(folder.parent/"publication.json",report)
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()

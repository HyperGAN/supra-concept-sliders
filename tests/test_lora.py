"""Exercise the actual backend's adapter with a small CPU DiT."""
import torch
from safetensors.torch import load_file, save_file

from supra.runtime import TARGETS, adapter_state, load_adapter_state, model_module


def test_frozen_base_zero_scale_and_export_roundtrip(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    mod = model_module()
    model = mod.SupraDiT(d_model=32, depth=2, n_heads=4, ctx_dim=16, num_tokens=4)
    x, t, ctx = torch.randn(2, 4, 4, 4), torch.rand(2), torch.randn(2, 8, 16)
    mask = torch.ones(2, 8)
    mask[:, 6:] = 0
    base = model(x, t, ctx, mask).detach()
    loras = mod.attach_supra_lora(model, rank=4, alpha=4, targets=TARGETS)
    assert torch.equal(base, model(x, t, ctx, mask))
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(2):
        opt.zero_grad()
        loss = (model(x, t, ctx, mask) - (base + 0.2)).square().mean()
        loss.backward()
        opt.step()
    trained = model(x, t, ctx, mask).detach()
    assert not torch.equal(base, trained)
    assert all(torch.equal(frozen[n], p) for n, p in model.named_parameters() if n in frozen)
    for lora in loras:
        lora.multiplier = 0
    assert torch.equal(base, model(x, t, ctx, mask))
    for lora in loras:
        lora.multiplier = 1
    path = tmp_path / "adapter.safetensors"
    save_file(adapter_state(model), str(path))
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.zero_()
    load_adapter_state(model, load_file(str(path)))
    assert torch.equal(trained, model(x, t, ctx, mask))


def test_incomplete_adapter_is_rejected():
    import pytest
    mod = model_module()
    model = mod.SupraDiT(d_model=32, depth=1, n_heads=4, ctx_dim=16, num_tokens=4)
    mod.attach_supra_lora(model, rank=4, alpha=4, targets=TARGETS)
    state = adapter_state(model)
    state.pop(next(iter(state)))
    with pytest.raises(ValueError, match="Adapter keys differ"):
        load_adapter_state(model, state)

import torch
from supra.distillation import compress_lora


def test_reduced_rank_fit_matches_direct_optimum_and_beats_weight_svd():
    torch.manual_seed(12)
    x = torch.randn(100, 12, dtype=torch.float64) * torch.logspace(-2, 2, 12)
    down, up = torch.randn(5, 12), torch.randn(9, 5)
    h = x @ down.double().T
    target = h @ up.double().T
    d, u = compress_lora(down, up, h.T @ h, rank=2)
    fitted = x @ d.double().T @ u.double().T
    _, s, _ = torch.linalg.svd(target, full_matrices=False)
    optimum = s[2:].square().sum()
    assert torch.allclose((target-fitted).square().sum(), optimum, rtol=1e-5)
    w = up.double() @ down.double()
    wu, ws, wv = torch.linalg.svd(w, full_matrices=False)
    naive = x @ ((wu[:, :2] * ws[:2]) @ wv[:2]).T
    assert (target-fitted).square().sum() < (target-naive).square().sum()


def test_full_rank_and_singular_calibration_covariance():
    torch.manual_seed(4)
    x = torch.randn(30, 8, dtype=torch.float64)
    down, up = torch.randn(4, 8), torch.randn(7, 4)
    down[-1] = down[0]
    h = x @ down.double().T
    d, u = compress_lora(down, up, h.T @ h, rank=4)
    assert torch.allclose(x @ d.double().T @ u.double().T,
                          h @ up.double().T, atol=1e-5, rtol=1e-5)

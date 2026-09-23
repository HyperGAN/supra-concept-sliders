"""Activation-weighted reduced-rank regression for ordinary LoRA teachers."""
import torch


@torch.no_grad()
def compress_lora(down, up, covariance, rank):
    """Minimize ||X D.T U.T - X D8.T U8.T||² with C=(X D.T).T(X D.T).

    The leading left singular vectors of U sqrt(C) span the optimal output
    subspace. Projecting U D into that subspace gives a rank-constrained fit
    without materializing the large input covariance or output activations.
    """
    if not 0 < rank <= down.shape[0]:
        raise ValueError("Student rank must be positive and no larger than teacher rank")
    d, u, c = down.double(), up.double(), covariance.double()
    values, vectors = torch.linalg.eigh((c + c.T) / 2)
    root = vectors * values.clamp_min(0).sqrt()[None, :]
    q, _, _ = torch.linalg.svd(u @ root, full_matrices=False)
    q = q[:, :rank]
    return (q.T @ u @ d).float(), q.float()


@torch.no_grad()
def projection_error(up, compressed_up, covariance):
    u, q, c = up.double(), compressed_up.double(), covariance.double()
    residual = u - q @ (q.T @ u)
    error = float(((residual @ c) * residual).sum().clamp_min(0))
    norm = float(((u @ c) * u).sum().clamp_min(0))
    return error, norm

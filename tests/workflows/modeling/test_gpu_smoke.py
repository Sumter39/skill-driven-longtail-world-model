from __future__ import annotations

import pytest
import torch

from skilldrive.models import ConditionalCVAE
from skilldrive.training.metrics import gaussian_kl_divergence


@pytest.mark.gpu
def test_full_default_cvae_cuda_amp_forward_backward_stays_below_memory_limit() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = ConditionalCVAE(dropout=0.0).to(device)
    batch_size = 2
    actor_count = 32
    history_steps = 50
    polyline_count = 128
    point_count = 20
    future_steps = 60
    batch = {
        "actor_history": torch.randn(
            batch_size, actor_count, history_steps, 6, device=device
        ),
        "actor_time_mask": torch.zeros(
            batch_size, actor_count, history_steps, dtype=torch.bool, device=device
        ),
        "actor_mask": torch.zeros(
            batch_size, actor_count, dtype=torch.bool, device=device
        ),
        "actor_type_id": torch.zeros(
            batch_size, actor_count, dtype=torch.long, device=device
        ),
        "actor_role_id": torch.zeros(
            batch_size, actor_count, dtype=torch.long, device=device
        ),
        "map_polylines": torch.randn(
            batch_size, polyline_count, point_count, 4, device=device
        ),
        "map_point_mask": torch.zeros(
            batch_size,
            polyline_count,
            point_count,
            dtype=torch.bool,
            device=device,
        ),
        "map_polyline_mask": torch.zeros(
            batch_size, polyline_count, dtype=torch.bool, device=device
        ),
        "map_type_id": torch.zeros(
            batch_size, polyline_count, dtype=torch.long, device=device
        ),
        "target_actor_index": torch.zeros(
            batch_size, dtype=torch.long, device=device
        ),
        "skill_id": torch.zeros(batch_size, dtype=torch.long, device=device),
        "skill_parameters": torch.zeros(batch_size, 106, device=device),
        "parameter_mask": torch.zeros(
            batch_size, 106, dtype=torch.bool, device=device
        ),
        "target_future": torch.randn(batch_size, future_steps, 2, device=device),
        "target_future_mask": torch.ones(
            batch_size, future_steps, dtype=torch.bool, device=device
        ),
    }
    batch["actor_time_mask"][:, :8] = True
    batch["actor_mask"][:, :8] = True
    batch["map_point_mask"][:, :16] = True
    batch["map_polyline_mask"][:, :16] = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator(device=device).manual_seed(2026)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model.forward_train(batch, generator=generator)
        reconstruction = torch.nn.functional.smooth_l1_loss(
            output.future_position_local,
            batch["target_future"],
        )
        kl = gaussian_kl_divergence(
            output.posterior_mean,
            output.posterior_logvar,
            output.prior_mean,
            output.prior_logvar,
        )
        loss = reconstruction + 0.1 * kl
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)

    assert torch.isfinite(loss)
    nonfinite_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    assert not nonfinite_gradients, nonfinite_gradients
    assert torch.cuda.max_memory_allocated(device) < 7 * 1024**3

import torch


def diffusion_mixup(
    x: torch.Tensor,
    weights: torch.Tensor,
    labels: torch.Tensor,
    diffusion_timesteps: int = 100,
    diffusion_t_min: int = 1,
    diffusion_t_max: int = 20,
    diffusion_beta_start: float = 0.0001,
    diffusion_beta_end: float = 0.02,
    diffusion_embedding_noise: float = 0.0,
    diffusion_fake_fraction: float = 1.0,
):
    """Create non-conditional diffusion-style extensions of speaker weights."""
    if diffusion_timesteps <= 0:
        raise ValueError("diffusion_timesteps must be positive")
    if diffusion_t_min < 0 or diffusion_t_max < diffusion_t_min:
        raise ValueError("diffusion timestep range is invalid")
    if diffusion_t_max >= diffusion_timesteps:
        raise ValueError("diffusion_t_max must be smaller than diffusion_timesteps")
    if diffusion_fake_fraction <= 0.0 or diffusion_fake_fraction > 1.0:
        raise ValueError("diffusion_fake_fraction must be in the range (0, 1]")

    device = x.device
    dtype = x.dtype
    labels_for_weights = labels.to(device=weights.device, dtype=torch.long)
    unique_labels_all, inverse_all = torch.unique(
        labels_for_weights,
        sorted=True,
        return_inverse=True,
    )
    unique_count = unique_labels_all.numel()
    synthetic_count = max(1, int(torch.ceil(torch.tensor(unique_count * diffusion_fake_fraction)).item()))
    if synthetic_count < unique_count:
        selected_unique_indices = torch.randperm(
            unique_count,
            device=weights.device,
        )[:synthetic_count].sort()[0]
    else:
        selected_unique_indices = torch.arange(unique_count, device=weights.device)

    selected_unique_labels = unique_labels_all.index_select(0, selected_unique_indices)
    selected_weights = weights.index_select(1, selected_unique_labels).to(device=device, dtype=dtype)

    label_map = torch.full(
        (unique_count,),
        -1,
        dtype=torch.long,
        device=weights.device,
    )
    label_map[selected_unique_indices] = torch.arange(
        synthetic_count,
        dtype=torch.long,
        device=weights.device,
    )
    mapped_labels = label_map.index_select(0, inverse_all)
    selected_rows = torch.nonzero(mapped_labels >= 0, as_tuple=False).view(-1).to(device=device)
    synthetic_labels = mapped_labels.index_select(0, selected_rows.to(device=weights.device)).to(device=device)
    synthetic_count = selected_weights.size(1)

    betas = torch.linspace(
        diffusion_beta_start,
        diffusion_beta_end,
        diffusion_timesteps,
        device=device,
        dtype=dtype,
    )
    alphas = 1.0 - betas
    alpha_bar_tail = torch.cumprod(alphas, dim=0)
    alpha_bars = torch.cat((torch.ones(1, device=device, dtype=dtype), alpha_bar_tail[:-1]))

    timesteps = torch.randint(
        diffusion_t_min,
        diffusion_t_max + 1,
        (synthetic_count,),
        device=device,
        dtype=torch.long,
    )
    alpha_bar_t = alpha_bars.index_select(0, timesteps).view(1, synthetic_count)
    epsilon = torch.randn_like(selected_weights)
    synthetic_weights = (
        torch.sqrt(alpha_bar_t) * selected_weights
        + torch.sqrt(1.0 - alpha_bar_t) * epsilon
    )
    synthetic_weights = synthetic_weights / torch.norm(
        synthetic_weights,
        p=2,
        dim=0,
        keepdim=True,
    ).clamp(min=1e-12)

    if diffusion_embedding_noise > 0:
        synthetic_embeddings = x.index_select(0, selected_rows)
        synthetic_embeddings = synthetic_embeddings + diffusion_embedding_noise * torch.randn_like(
            synthetic_embeddings
        )
    else:
        synthetic_embeddings = x.index_select(0, selected_rows)

    return synthetic_embeddings, synthetic_labels, synthetic_weights

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
):
    """Create non-conditional diffusion-style extensions of speaker weights."""
    if diffusion_timesteps <= 0:
        raise ValueError("diffusion_timesteps must be positive")
    if diffusion_t_min < 0 or diffusion_t_max < diffusion_t_min:
        raise ValueError("diffusion timestep range is invalid")
    if diffusion_t_max >= diffusion_timesteps:
        raise ValueError("diffusion_t_max must be smaller than diffusion_timesteps")

    batch_size = x.size(0)
    device = x.device
    dtype = x.dtype
    labels_for_weights = labels.to(device=weights.device, dtype=torch.long)
    selected_weights = weights.index_select(1, labels_for_weights).to(device=device, dtype=dtype)

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
        (batch_size,),
        device=device,
        dtype=torch.long,
    )
    alpha_bar_t = alpha_bars.index_select(0, timesteps).view(1, batch_size)
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
        synthetic_embeddings = x + diffusion_embedding_noise * torch.randn_like(x)
    else:
        synthetic_embeddings = x

    synthetic_labels = torch.arange(batch_size, dtype=torch.long, device=device)
    return synthetic_embeddings, synthetic_labels, synthetic_weights

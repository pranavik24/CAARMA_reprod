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

    device = x.device
    dtype = x.dtype
    labels_for_weights = labels.to(device=weights.device, dtype=torch.long)
    unique_labels, synthetic_labels = torch.unique(
        labels_for_weights,
        sorted=True,
        return_inverse=True,
    )
    selected_weights = weights.index_select(1, unique_labels).to(device=device, dtype=dtype)
    synthetic_labels = synthetic_labels.to(device=device)
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
        synthetic_embeddings = x + diffusion_embedding_noise * torch.randn_like(x)
    else:
        synthetic_embeddings = x

    return synthetic_embeddings, synthetic_labels, synthetic_weights

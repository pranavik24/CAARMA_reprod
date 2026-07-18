from criterion.amsoftmax_mix_gan import amsoftmax_gan

def build_criterion(config):
    if config['criterion'] == 'AMSoftmaxGAN':
        criterion = amsoftmax_gan(
            embedding_dim=int(config.get('embedding_dim', 192)),
            num_classes=int(config.get('num_spk', 1211)),
            margin=float(config.get('margin', 0.2)),
            scale=float(config.get('scale', 30)),
            synthetic_strategy=config.get("synthetic_strategy", "avg"),
            diffusion_timesteps=int(config.get("diffusion_timesteps", 100)),
            diffusion_t_min=int(config.get("diffusion_t_min", 1)),
            diffusion_t_max=int(config.get("diffusion_t_max", 20)),
            diffusion_beta_start=float(config.get("diffusion_beta_start", 0.0001)),
            diffusion_beta_end=float(config.get("diffusion_beta_end", 0.02)),
            diffusion_embedding_noise=float(config.get("diffusion_embedding_noise", 0.0)),
            diffusion_fake_fraction=float(config.get("diffusion_fake_fraction", 1.0)),
        )
    else:
        raise NotImplementedError

    return criterion

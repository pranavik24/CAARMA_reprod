from criterion.amsoftmax_mix_gan import amsoftmax_gan

def build_criterion(config):
    if config['criterion'] == 'AMSoftmaxGAN':
        criterion = amsoftmax_gan(
            embedding_dim=int(config.get('embedding_dim', 192)),
            num_classes=int(config.get('num_spk', 1211)),
            margin=float(config.get('margin', 0.2)),
            scale=float(config.get('scale', 30)),
        )
    else:
        raise NotImplementedError

    return criterion

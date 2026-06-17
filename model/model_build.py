from .ECAPA_TDNN import ecapa_tdnn, ecapa_tdnn_large
from .MFA_Conformer import conformer_cat
from .Raw_Net import RawNet3, Bottle2neck
from .ska_tdnn import SKA_MainModel

def build_model(config, device):
    embedding_dim = int(config.get('embedding_dim', 192))

    if config['model'] == 'ECAPA':
        model = ecapa_tdnn(n_mels=80, embedding_dim=embedding_dim, channel=512)
    
    elif config['model'] == 'MFA-CONFORMER':
        model = conformer_cat(n_mels=80, num_blocks=6, output_size=256, 
        embedding_dim=embedding_dim, input_layer="conv2d2", pos_enc_layer_type="rel_pos").to(device)
        
    elif config['model'] == 'ECAPA-LARGE':
        model = ecapa_tdnn_large(n_mels=80, embedding_dim=embedding_dim, channel=1024)
        
    elif config['model'] == 'RAWNET3':
        model = RawNet3(
        Bottle2neck, 
        model_scale=8, 
        context=True, 
        summed=True, 
        nOut=embedding_dim,
        encoder_type="ECA",
        log_sinc=True,
        norm_sinc="mean_std",
        out_bn=True,
        sinc_stride=10,
        )
        
    elif config['model'] == 'SKA_TDNN':
        model = SKA_MainModel(eca_c=1024, eca_s=8, log_input=True, num_mels=80, num_out=embedding_dim, pooling='CCSP')

    else: 
        raise NotImplementedError

    return model

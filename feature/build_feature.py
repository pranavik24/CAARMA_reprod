from .fbanks import Mel_Spectrogram


def build_feature(config):
    feature_name = str(config['features']).lower()
    if feature_name in {'fbank', 'fbanks', 'fbank_new'}:
        features = Mel_Spectrogram()        
    else:
        raise NotImplementedError(f"Unsupported feature type: {config['features']}")
    
    return features

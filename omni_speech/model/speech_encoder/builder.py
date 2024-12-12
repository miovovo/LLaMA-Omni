from .speech_encoder import WhisperWrappedEncoder
from .speech_encoder import ConformerWrappedEncoder

def build_speech_encoder(config):
    speech_encoder_type = getattr(config, 'speech_encoder_type', None)
    if "conformer" in speech_encoder_type.lower():
        return ConformerWrappedEncoder.load(config)

    raise ValueError(f'Unknown speech encoder: {speech_encoder_type}')

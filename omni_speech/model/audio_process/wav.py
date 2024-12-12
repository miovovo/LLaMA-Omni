import torchaudio
import torch.nn.functional as F

def preprocess_audio(file_path: str, target_sample_rate: int = 16000, n_mels: int = 80, sequence_length: int = 12345):
    """
    Preprocess an audio file for input to the Conformer model.

    Args:
        file_path (str): Path to the audio file.
        target_sample_rate (int): Target sampling rate for audio resampling.
        n_mels (int): Number of mel filter banks (feature dimension).
        sequence_length (int): Fixed length for padding/truncating sequences.

    Returns:
        Tuple[Tensor, int]: Preprocessed audio feature and its original length.
    """
    # Load audio
    waveform, sample_rate = torchaudio.load(file_path)

    # Resample if needed
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    # Convert to mel spectrogram
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_mels=n_mels,
        hop_length=400,
        n_fft=1024
    )
    features = mel_spectrogram(waveform).squeeze(0).transpose(0, 1)  # Shape: (time_frames, n_mels)

    # Original sequence length
    original_length = features.size(0)

    # Pad or truncate to the fixed length
    features = F.pad(features, (0, 0, 0, max(0, sequence_length - original_length)))
    features = features[:sequence_length]

    return features, original_length

import librosa
import librosa.filters
import math
import numpy as np
from scipy import signal
from hparams import hparams
from scipy.io import wavfile

# r9r9 preprocessing
import lws


def load_wav(path):
    return librosa.load(path, sr=hparams.sample_rate)[0]

def save_wav(wav, path):
    wav = wav * 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, hparams.sample_rate, wav.astype(np.int16))


def preemphasis(x):
    from nnmnkwii.preprocessing import preemphasis
    return preemphasis(x, hparams.preemphasis)


def inv_preemphasis(x):
    from nnmnkwii.preprocessing import inv_preemphasis
    return inv_preemphasis(x, hparams.preemphasis)


def spectrogram(y):
    D = _lws_processor().stft(preemphasis(y)).T
    S = _amp_to_db(np.abs(D)) - hparams.ref_level_db
    return _normalize(S)


def inv_spectrogram(spectrogram):
    '''Converts spectrogram to waveform using librosa'''
    S = _db_to_amp(_denormalize(spectrogram) + hparams.ref_level_db)  # Convert back to linear
    processor = _lws_processor()
    D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
    y = processor.istft(D).astype(np.float32)
    return inv_preemphasis(y)


def melspectrogram(y):
    D = _lws_processor().stft(preemphasis(y)).T
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hparams.ref_level_db
    if not hparams.allow_clipping_in_normalization:
        assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0
    return _normalize(S)


def _lws_processor():
    return lws.lws(hparams.fft_size, hparams.hop_size, mode="speech")


# Conversions:


_mel_basis = None


def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _build_mel_basis():
    if hparams.fmax is not None:
        assert hparams.fmax <= hparams.sample_rate // 2
    return librosa.filters.mel(hparams.sample_rate, hparams.fft_size,
                               fmin=hparams.fmin, fmax=hparams.fmax,
                               n_mels=hparams.num_mels)


def _amp_to_db(x):
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize(S):
    return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)


def _denormalize(S):
    return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db


# Fatcord's preprocessing
def quantize(x):
    """quantize audio signal

    """
    quant = (x + 1.) * (2**hparams.bits - 1) / 2
    return quant.astype(np.int)


# mulaw encoding and decoding
def encode_mu_law(x, mu) :
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def decode_mu_law(y, mu, from_labels=True) :
    # TODO : get rid of log2 - makes no sense
    if from_labels : y = label_2_float(y, math.log2(mu))
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


# testing
def test_everything():
    wav = np.random.randn(12000,)
    mel = melspectrogram(wav)
    spec = spectrogram(wav)
    quant = quantize(wav)
    print(wav.shape, mel.shape, spec.shape, quant.shape)
    print(quant.max(), quant.min(), mel.max(), mel.min(), spec.max(), spec.min())
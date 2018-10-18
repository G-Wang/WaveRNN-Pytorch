import numpy as np
import torch

def num_params(model) :
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    print('Trainable Parameters: %.3f million' % parameters)


# for mulaw encoding and decoding in torch tensors, modified from: https://github.com/pytorch/audio/blob/master/torchaudio/transforms.py
def mulaw_quantize(x, quantization_channels=256):
    """Encode signal based on mu-law companding.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_

    This algorithm assumes the signal has been scaled to between -1 and 1 and
    returns a signal encoded with values from 0 to quantization_channels - 1

    Args:
        quantization_channels (int): Number of channels. default: 256

    """
    mu = quantization_channels - 1
    if isinstance(x, np.ndarray):
        x_mu = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
        x_mu = ((x_mu + 1) / 2 * mu + 0.5).astype(int)
    elif isinstance(x, (torch.Tensor, torch.LongTensor)):

        if isinstance(x, torch.LongTensor):
            x = x.float()
        mu = torch.FloatTensor([mu])
        x_mu = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu)
        x_mu = ((x_mu + 1) / 2 * mu + 0.5).long()
    return x_mu


def inv_mulaw_quantize(x_mu, quantization_channels=256, cuda=False):
    """Decode mu-law encoded signal.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_

    This expects an input with values between 0 and quantization_channels - 1
    and returns a signal scaled between -1 and 1.

    Args:
        quantization_channels (int): Number of channels. default: 256

    """
    mu = quantization_channels - 1.
    if isinstance(x_mu, np.ndarray):
        x = ((x_mu) / mu) * 2 - 1.
        x = np.sign(x) * (np.exp(np.abs(x) * np.log1p(mu)) - 1.) / mu
    elif isinstance(x_mu, (torch.Tensor, torch.LongTensor)):
        if isinstance(x_mu, (torch.LongTensor, torch.cuda.LongTensor)):
            x_mu = x_mu.float()
        if cuda:
            mu = (torch.FloatTensor([mu])).cuda()
        else:
            mu = torch.FloatTensor([mu])
        x = ((x_mu) / mu) * 2 - 1.
        x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.) / mu
    return x


def test_inv_mulaw():
    wav = torch.rand(5, 5000)
    wav = wav.cuda()
    de_quant = inv_mulaw_quantize(wav, 512, True)
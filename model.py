import torch
from torch import nn
import torch.nn.functional as F
from hparams import hparams as hp
from torch.utils.data import DataLoader, Dataset
from distributions import *
from utils import num_params, mulaw_quantize, inv_mulaw_quantize

from tqdm import tqdm
import numpy as np

class ResBlock(nn.Module) :
    def __init__(self, dims) :
        super().__init__()
        self.conv1 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(dims)
        self.batch_norm2 = nn.BatchNorm1d(dims)
        
    def forward(self, x) :
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x + residual

class MelResNet(nn.Module) :
    def __init__(self, res_blocks, in_dims, compute_dims, res_out_dims) :
        super().__init__()
        self.conv_in = nn.Conv1d(in_dims, compute_dims, kernel_size=5, bias=False)
        self.batch_norm = nn.BatchNorm1d(compute_dims)
        self.layers = nn.ModuleList()
        for i in range(res_blocks) :
            self.layers.append(ResBlock(compute_dims))
        self.conv_out = nn.Conv1d(compute_dims, res_out_dims, kernel_size=1)
        
    def forward(self, x) :
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        for f in self.layers : x = f(x)
        x = self.conv_out(x)
        return x

class Stretch2d(nn.Module) :
    def __init__(self, x_scale, y_scale) :
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        
    def forward(self, x) :
        b, c, h, w = x.size()
        x = x.unsqueeze(-1).unsqueeze(3)
        x = x.repeat(1, 1, 1, self.y_scale, 1, self.x_scale)
        return x.view(b, c, h * self.y_scale, w * self.x_scale)

class UpsampleNetwork(nn.Module) :
    def __init__(self, feat_dims, upsample_scales, compute_dims, 
                 res_blocks, res_out_dims, pad) :
        super().__init__()
        total_scale = np.cumproduct(upsample_scales)[-1]
        self.indent = pad * total_scale
        self.resnet = MelResNet(res_blocks, feat_dims, compute_dims, res_out_dims)
        self.resnet_stretch = Stretch2d(total_scale, 1)
        self.up_layers = nn.ModuleList()
        for scale in upsample_scales :
            k_size = (1, scale * 2 + 1)
            padding = (0, scale)
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
            conv.weight.data.fill_(1. / k_size[1])
            self.up_layers.append(stretch)
            self.up_layers.append(conv)
    
    def forward(self, m) :
        aux = self.resnet(m).unsqueeze(1)
        aux = self.resnet_stretch(aux)
        aux = aux.squeeze(1)
        m = m.unsqueeze(1)
        for f in self.up_layers : m = f(m)
        m = m.squeeze(1)[:, :, self.indent:-self.indent]
        return m.transpose(1, 2), aux.transpose(1, 2)


class Model(nn.Module) :
    def __init__(self, rnn_dims, fc_dims, bits, pad, upsample_factors,
                 feat_dims, compute_dims, res_out_dims, res_blocks):
        super().__init__()
        if hp.input_type == 'raw':
            self.n_classes = 2
        elif hp.input_type == 'mixture':
            # mixture requires multiple of 3, default at 10 component mixture, i.e 3 x 10 = 30
            self.n_classes = 30
        elif hp.input_type == 'mulaw':
            self.n_classes = hp.mulaw_quantize_channels
        elif hp.input_type == 'bits':
            self.n_classes = 2**bits
        else:
            raise ValueError("input_type: {hp.input_type} not supported")
        self.rnn_dims = rnn_dims
        self.aux_dims = res_out_dims // 4
        self.upsample = UpsampleNetwork(feat_dims, upsample_factors, compute_dims, 
                                        res_blocks, res_out_dims, pad)
        self.I = nn.Linear(feat_dims + self.aux_dims + 1, rnn_dims)
        self.rnn1 = nn.GRU(rnn_dims, rnn_dims, batch_first=True)
        self.rnn2 = nn.GRU(rnn_dims + self.aux_dims, rnn_dims, batch_first=True)
        self.fc1 = nn.Linear(rnn_dims + self.aux_dims, fc_dims)
        self.fc2 = nn.Linear(fc_dims + self.aux_dims, fc_dims)
        self.fc3 = nn.Linear(fc_dims, self.n_classes)
        num_params(self)
    
    def forward(self, x, mels) :
        bsize = x.size(0)
        h1 = torch.zeros(1, bsize, self.rnn_dims).cuda()
        h2 = torch.zeros(1, bsize, self.rnn_dims).cuda()
        mels, aux = self.upsample(mels)
        
        aux_idx = [self.aux_dims * i for i in range(5)]
        a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
        a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
        a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
        a4 = aux[:, :, aux_idx[3]:aux_idx[4]]
        
        x = torch.cat([x.unsqueeze(-1), mels, a1], dim=2)
        x = self.I(x)
        res = x
        x, _ = self.rnn1(x, h1)
        
        x = x + res
        res = x
        x = torch.cat([x, a2], dim=2)
        x, _ = self.rnn2(x, h2)
        
        x = x + res
        x = torch.cat([x, a3], dim=2)
        x = F.relu(self.fc1(x))
        
        x = torch.cat([x, a4], dim=2)
        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        if hp.input_type == 'raw':
            return x
        elif hp.input_type == 'mixture':
            return x
        elif hp.input_type == 'bits' or hp.input_type == 'mulaw':
            return F.log_softmax(x, dim=-1)
        else:
            raise ValueError("input_type: {hp.input_type} not supported")


    def preview_upsampling(self, mels) :
        mels, aux = self.upsample(mels)
        return mels, aux
    
    def generate(self, mels) :
        self.eval()
        output = []
        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)
        
        with torch.no_grad() :
            x = torch.zeros(1, 1).cuda()
            h1 = torch.zeros(1, self.rnn_dims).cuda()
            h2 = torch.zeros(1, self.rnn_dims).cuda()
            
            mels = torch.FloatTensor(mels).cuda().unsqueeze(0)
            mels, aux = self.upsample(mels)
            
            aux_idx = [self.aux_dims * i for i in range(5)]
            a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
            a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
            a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
            a4 = aux[:, :, aux_idx[3]:aux_idx[4]]
            
            seq_len = mels.size(1)
            
            for i in tqdm(range(seq_len)) :

                m_t = mels[:, i, :]
                a1_t = a1[:, i, :]
                a2_t = a2[:, i, :]
                a3_t = a3[:, i, :]
                a4_t = a4[:, i, :]
                
                x = torch.cat([x, m_t, a1_t], dim=1)
                x = self.I(x)
                h1 = rnn1(x, h1)
                
                x = x + h1
                inp = torch.cat([x, a2_t], dim=1)
                h2 = rnn2(inp, h2)
                
                x = x + h2
                x = torch.cat([x, a3_t], dim=1)
                x = F.relu(self.fc1(x))
                
                x = torch.cat([x, a4_t], dim=1)
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                if hp.input_type == 'raw':
                    if hp.distribution == 'beta':
                        sample = sample_from_beta_dist(x.unsqueeze(0))
                    elif hp.distribution == 'gaussian':
                        sample = sample_from_gaussian(x.unsqueeze(0))
                elif hp.input_type == 'mixture':
                    sample = sample_from_discretized_mix_logistic(x.unsqueeze(-1),hp.log_scale_min)
                elif hp.input_type == 'bits':
                    posterior = F.softmax(x, dim=1).view(-1)
                    distrib = torch.distributions.Categorical(posterior)
                    sample = 2 * distrib.sample().float() / (self.n_classes - 1.) - 1.
                elif hp.input_type == 'mulaw':
                    posterior = F.softmax(x, dim=1).view(-1)
                    distrib = torch.distributions.Categorical(posterior)
                    sample = inv_mulaw_quantize(distrib.sample(), hp.mulaw_quantize_channels, True)
                output.append(sample.view(-1))
                x = torch.FloatTensor([[sample]]).cuda()
        output = torch.stack(output).cpu().numpy()
        self.train()
        return output


    def batch_generate(self, mels) :
        """mel should be of shape [batch_size x 80 x mel_length]
        """
        self.eval()
        output = []
        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)
        b_size = mels.shape[0]
        assert len(mels.shape) == 3, "mels should have shape [batch_size x 80 x mel_length]"
        
        with torch.no_grad() :
            x = torch.zeros(b_size, 1).cuda()
            h1 = torch.zeros(b_size, self.rnn_dims).cuda()
            h2 = torch.zeros(b_size, self.rnn_dims).cuda()
            
            mels = torch.FloatTensor(mels).cuda()
            mels, aux = self.upsample(mels)
            
            aux_idx = [self.aux_dims * i for i in range(5)]
            a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
            a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
            a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
            a4 = aux[:, :, aux_idx[3]:aux_idx[4]]
            
            seq_len = mels.size(1)
            
            for i in tqdm(range(seq_len)) :

                m_t = mels[:, i, :]
                a1_t = a1[:, i, :]
                a2_t = a2[:, i, :]
                a3_t = a3[:, i, :]
                a4_t = a4[:, i, :]
                
                x = torch.cat([x, m_t, a1_t], dim=1)
                x = self.I(x)
                h1 = rnn1(x, h1)
                
                x = x + h1
                inp = torch.cat([x, a2_t], dim=1)
                h2 = rnn2(inp, h2)
                
                x = x + h2
                x = torch.cat([x, a3_t], dim=1)
                x = F.relu(self.fc1(x))
                
                x = torch.cat([x, a4_t], dim=1)
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                if hp.input_type == 'raw':
                    sample = sample_from_beta_dist(x.unsqueeze(0))
                elif hp.input_type == 'mixture':
                    sample = sample_from_discretized_mix_logistic(x.unsqueeze(-1),hp.log_scale_min)
                elif hp.input_type == 'bits':
                    posterior = F.softmax(x, dim=1).view(b_size, -1)
                    distrib = torch.distributions.Categorical(posterior)
                    sample = 2 * distrib.sample().float() / (self.n_classes - 1.) - 1.
                elif hp.input_type == 'mulaw':
                    posterior = F.softmax(x, dim=1).view(b_size, -1)
                    distrib = torch.distributions.Categorical(posterior)
                    print(type(distrib.sample()))
                    sample = inv_mulaw_quantize(distrib.sample(), hp.mulaw_quantize_channels, True)
                output.append(sample.view(-1))
                x = sample.view(b_size,1)
        output = torch.stack(output).cpu().numpy()
        self.train()
        # output is a batch of wav segments of shape [batch_size x seq_len]
        # will need to merge into one wav of size [batch_size * seq_len]
        assert output.shape[1] == b_size
        output = (output.swapaxes(1,0)).reshape(-1)
        return output
    
    def get_gru_cell(self, gru) :
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell


def build_model():
    """build model with hparams settings

    """
    if hp.input_type == 'raw':
        print('building model with Beta distribution output')
    elif hp.input_type == 'mixture':
        print("building model with mixture of logistic output")
    elif hp.input_type == 'bits':
        print("building model with quantized bit audio")
    elif hp.input_type == 'mulaw':
        print("building model with quantized mulaw encoding")
    else:
        raise ValueError('input_type provided not supported')
    model = Model(hp.rnn_dims, hp.fc_dims, hp.bits,
        hp.pad, hp.upsample_factors, hp.num_mels,
        hp.compute_dims, hp.res_out_dims, hp.res_blocks)

    return model 

def no_test_build_model():
    model = Model(hp.rnn_dims, hp.fc_dims, hp.bits,
        hp.pad, hp.upsample_factors, hp.num_mels,
        hp.compute_dims, hp.res_out_dims, hp.res_blocks).cuda()
    print(vars(model))


def test_batch_generate():
    model = Model(hp.rnn_dims, hp.fc_dims, hp.bits,
        hp.pad, hp.upsample_factors, hp.num_mels,
        hp.compute_dims, hp.res_out_dims, hp.res_blocks).cuda()
    print(vars(model))
    batch_mel = torch.rand(3, 80, 100)
    output = model.batch_generate(batch_mel)
    print(output.shape)
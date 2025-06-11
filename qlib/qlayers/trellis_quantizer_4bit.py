import os
from dataclasses import dataclass
import torch
import torch.nn as nn
from tqdm import tqdm
import math
from qlib.qlayers.kernel_decompress import decode_compressed #DecodeKernelAG
from dataclasses import dataclass
from qlib.qlayers.trellis_quantizer import TrellisQuantizer, TrellisQuantizerParams, quantlut_sym_2d


def get_positive_lowbit_codebook(base_codebook_size, values_bits, bound):
    """Generate a symmetric low-bit codebook"""
    sample_values = int(base_codebook_size * 1.5)
    scale = bound / ((2**(values_bits-1)) - 0.5)

    quantiles = torch.special.ndtr(scale * (torch.arange(2**(values_bits-1))))
    quantiles_padded = torch.tensor(list(quantiles) + [1])
    freq = (quantiles_padded[1:] - quantiles_padded[:-1]).unsqueeze(0)
    freq_2d = freq.T @ freq

    counts = (freq_2d * sample_values / freq_2d.sum())
    counts = counts.round()
    counts = counts.flatten()

    unique_values = (torch.arange(2**(values_bits-1)) + 0.5)
    unique_cb_h = unique_values.repeat(len(unique_values), 1)
    unique_cb_v = unique_cb_h.T
    unique_cb_2d = torch.stack([unique_cb_v, unique_cb_h], dim=0)

    unique_cb = unique_cb_2d.reshape(2, -1).T

    cb = []
    for i, c in enumerate(counts):
        cb += int(c) * [unique_cb[i],]
        
    cb = torch.stack(cb)
    n_to_remove = len(cb)- base_codebook_size
    torch.manual_seed(0)
    cb = cb[torch.randperm(len(cb))][n_to_remove:]

    return cb, scale


class TrellisQuantizer4bit(TrellisQuantizer):
    def __init__(self,
                 params: TrellisQuantizerParams):
        super(TrellisQuantizer4bit, self).__init__()
        self.idx_dtype = torch.int32

        self.L = params.L
        self.K = params.K
        self.V = params.V
        self.T = params.T
        self.tlut_bits = params.tlut_bits
        self.decode_mode = params.decode_mode

        # Adaptive batch sizing
        if params.viterbi_bs == 'auto':
            self.viterbi_bs = min(2**(24 - self.L), 256)
        else:
            self.viterbi_bs = params.viterbi_bs

        assert self.decode_mode == 'LowBitSym'
        assert self.V == 2
        assert self.tlut_bits in [10, 12] 
        
        # Compressed codebook bitwidth
        if self.tlut_bits==10:
            values_bits = 4
        else:
            values_bits = 5

        bound = 3.0 # three standard deviations of N(0, 1)

        fname = f'/home/msst/repo/Quantization/ml/weights/TrellisQuantizer4bit_codebook_v{values_bits}_l{self.tlut_bits}.pt'
        if not os.path.exists(fname):
            tlut, codebook_scale = get_positive_lowbit_codebook(2**self.tlut_bits, values_bits=values_bits, bound=bound)
            torch.save(tlut, fname)
        else:
            tlut = torch.load(fname, weights_only=True)
        self.register_buffer('tlut', tlut)
        self.register_buffer('codebook_scale', codebook_scale)
        self.register_buffer(
            'lut',
            quantlut_sym_2d(self.tlut, self.L, self.tlut_bits).contiguous())


    def quantize(self, X, batch_size='auto', **kwargs):
        X_shape = X.shape
        assert self.T == 256 # 256 needed for fast kernerl
        
        X = X.reshape(-1, self.T) / codebook_scale

        # Fisrt fase
        roll_X = torch.roll(X, self.T // (2 * self.V) * self.V, 1)
        state = self.quantize_seq(roll_X, overlap=None, batch_size=batch_size)
        overlap = state[:, self.T // (2 * self.V)] >> self.K * self.V
        # Second fase
        state = self.quantize_seq(X, overlap=overlap, batch_size=batch_size)
        
        hatX = self.recons(state).reshape(X_shape)
        return hatX.contiguous().to(X.device), state.contiguous().to(X.device)

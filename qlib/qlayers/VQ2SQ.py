import math
from functools import cache
import random 
from tqdm import tqdm
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
import os

from qlib.utils.incoherence_preprocessing.incoherence_process_functions import incoherence_process, incoherence_preprocess


def get_positive_lowbit_codebook(base_codebook_size, values_bits, bound):
    sample_values = int(base_codebook_size * 1.5)
    scale = bound / ((2**(values_bits-1)) - 0.5)

    quantiles = torch.special.ndtr(scale * (torch.arange(2**(values_bits-1))))
    quantiles_padded = torch.tensor(list(quantiles) + [1])
    freq = (quantiles_padded[1:] - quantiles_padded[:-1]).unsqueeze(0)
    freq_2d = freq.T @ freq

    counts = (freq_2d * sample_values / freq_2d.sum())
    counts = counts.round()
    #counts = counts.to(torch.int) + 1
    counts = counts.flatten()

    unique_values = scale * (torch.arange(2**(values_bits-1)) + 0.5)
    unique_cb_h = unique_values.repeat(len(unique_values), 1)
    unique_cb_v = unique_cb_h.T
    unique_cb_2d = torch.stack([unique_cb_v, unique_cb_h], dim=0)

    unique_cb = unique_cb_2d.reshape(2, -1).T

    cb = []
    for i, c in enumerate(counts):
        cb += int(c) * [unique_cb[i],]
        
    cb = torch.stack(cb)
    n_to_remove = len(cb)- base_codebook_size
    cb = cb[torch.randperm(len(cb))][n_to_remove:]

    return cb, scale


def decode_1mad(x):
    x = x.to(torch.int64)
    x = x & ((1 << 32) - 1)
    x = x * 34038481 + 76625530
    x = x & ((1 << 32) - 1)
    y = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)
    y = y - 510
    y = y.to(torch.float32)
    y = y / 147.800537109375
    return y


def quantlut_sym(tlut, L, nbits):
    with torch.no_grad():
        lut = torch.arange(1 << L, device=tlut.device)
        lut = (lut + 1) * lut
        sflp = 1 - ((lut >> 15) & 1) * 2
        lut = (lut >> (16 - nbits - 1)) & ((1 << nbits) - 1)
    lut = tlut[lut]
    lut[:, 0] = lut[:, 0] * sflp
    return lut


def quantlut_sym_2d(tlut, L, nbits):
    with torch.no_grad():
        lut = torch.arange(1 << L, device=tlut.device)
        lut = (lut + 1) * lut
        sflp0 = 1 - ((lut >> 15) & 1) * 2
        sflp1 = 1 - ((lut >> 7) & 1) * 2
        lut = (lut >> (16 - nbits - 1)) & ((1 << nbits) - 1)
    lut = tlut[lut] * torch.stack([sflp0, sflp1]).T
    return lut


class bitshift_codebook(nn.Module):
    def __init__(self,
                 L=16,
                 K=2,
                 V=2,
                 decode_mode='1mad',
                 tlut_bits=10,
                 tlut=None):
        super(bitshift_codebook, self).__init__()
        self.idx_dtype = torch.int32
        self.opt_scale = 1

        self.L = L
        self.K = K
        self.V = V
        self.decode_mode = decode_mode

        if decode_mode == '1mad':
            assert V == 1
            self.register_buffer('lut',
                                 decode_1mad(torch.arange(2**L)).unsqueeze(0))
        
        elif decode_mode == 'quantlut_sym':
            if tlut is None:
                tlut_bits = 9
                assert tlut_bits > 0
                if V == 2:
                    fname = f'/tmp/kmeans_{tlut_bits}_{V}.pt'
                    if not os.path.exists(fname):
                        tlut = torch.randn(2**tlut_bits, V)
                        import scipy
                        data = torch.randn(1 << 20, 2)
                        clusters = scipy.cluster.vq.kmeans(data, tlut)
                        tlut = torch.tensor(clusters[0])
                        tlut = (tlut /
                                tlut.std(unbiased=False)) * 0.9682458365518543
                        torch.save(tlut, fname)
                    else:
                        tlut = torch.load(fname)
                else:
                    raise Exception
                self.register_buffer('tlut', tlut)
                self.register_buffer(
                    'lut',
                    quantlut_sym(self.tlut, L, tlut_bits).T.contiguous())

        elif decode_mode == 'quantlut_sym_2d':
            assert tlut_bits > 0
            tlut = get_positive_lowbit_codebook(2**tlut_bits, values_bits=4, bound=3.0)[0]
            self.register_buffer('tlut', tlut)
            self.register_buffer(
                'lut',
                quantlut_sym_2d(self.tlut, L, tlut_bits).T.contiguous())

        else:
            raise Exception

        self.fakeinf = torch.tensor(torch.inf)

        self.register_buffer('sumdelta',
                             torch.arange(2**(K * V)) << (L - K * V))
        self.sumdelta = self.sumdelta.view(1, 1, -1)

        self.register_buffer('state', torch.arange(2**L).unsqueeze(0))
        self.register_buffer('state_cand',
                             (self.state >>
                              (K * V))[0, ::2**(K * V)].unsqueeze(-1) +
                             self.sumdelta)
        self.register_buffer('recons_state', self.recons(self.state))

        self.version = 0

    def recons(self, encoded, **kwargs):
        return self.lut[:, encoded.int().to(self.lut.device)].to(encoded.device)

    @torch.compile
    def update(self, cost, thing):
        state_err = (self.recons_state -
                     thing.unsqueeze(-1)).square().sum(dim=0)
        cand_cost = torch.gather(
            cost.unsqueeze(-2).expand(-1, self.state_cand.shape[1], -1), -1,
            self.state_cand.expand(len(cost), -1, 2**(self.K * self.V)))
        best = torch.min(cand_cost, dim=-1)
        cost = state_err + best.values.unsqueeze(-1).expand(
            -1, -1, 2**(self.K * self.V)).reshape(state_err.shape)
        prev_state = torch.gather(
            self.state_cand.expand(thing.shape[1], -1, -1), -1,
            best.indices.unsqueeze(-1))[..., 0]
        return prev_state, cost

    def viterbi(self, X, overlap=None):
        T, B = X.shape
        assert T % self.V == 0
        # cost is (B, 2**L)
        cost = (self.recons_state -
                X[:self.V].unsqueeze(-1)).square().sum(dim=0)

        if overlap is not None:
            mask = torch.ones(B, 2**self.L, device=X.device) * self.fakeinf
            allow = (overlap <<
                     (self.K * self.V)).unsqueeze(-1) + torch.arange(
                         2**(self.K * self.V)).to(X.device).view(1, 1, -1)
            mask.scatter_(1, allow[0], 0)
            cost = torch.min(cost + mask, self.fakeinf)

        from_state = torch.zeros(T // self.V,
                                 B,
                                 2**(self.L - self.K * self.V),
                                 dtype=self.state.dtype,
                                 device=self.state.device)

        for i in range(1, T // self.V):
            from_state[i], cost = self.update(cost,
                                              X[i * self.V:(i + 1) * self.V])

        if overlap is not None:
            mask = torch.ones(B, 2**self.L, device=X.device) * self.fakeinf
            allow = (overlap.unsqueeze(-1) + self.sumdelta.unsqueeze(0))
            mask.scatter_(1, allow[0, 0], 0)
            cost = torch.min(cost + mask, self.fakeinf)

        final_state = torch.zeros(T // self.V,
                                  B,
                                  dtype=self.idx_dtype,
                                  device=X.device)
        final_state[T // self.V - 1] = torch.argmin(cost, dim=-1)
        for i in range(T // self.V - 1, 0, -1):
            final_state[i - 1] = torch.gather(
                from_state[i], -1,
                (final_state[i].to(torch.int64).unsqueeze(-1)) >>
                (self.K * self.V))[..., 0]
        return final_state

    def quantize_seq(self, X, overlap=None, batch_size='auto', **kwargs):
        T, NO = X.shape
        if batch_size == 'auto':
            bs = min(2**(24 - self.L), NO)
        else:
            bs = batch_size
        pad_amt = math.ceil(NO / bs) * bs - NO
        X = torch.nn.functional.pad(X, (0, pad_amt))
        T, N = X.shape
        X = X.reshape(T, N // bs, bs).transpose(0, 1).contiguous()
        if overlap is not None:
            overlap = torch.nn.functional.pad(overlap, (0, pad_amt))
            overlap = overlap.reshape(N // bs, bs)

        Qidxs = torch.zeros(N // bs,
                            T // self.V,
                            bs,
                            dtype=self.idx_dtype,
                            device=X.device)
        for i in tqdm(range(len(X))):
            b_overlap = None if overlap is None else overlap[i]
            Qidxs[i] = self.viterbi(X[i], overlap=b_overlap)
        Qidxs = Qidxs.transpose(0, 1).reshape(T // self.V, N)[:, :NO]
        return Qidxs

    def quantize(self, X, batch_size='auto', **kwargs):
        X = X.T.contiguous().to(torch.float16)
        T = X.shape[0]
        roll_X = torch.roll(X, T // (2 * self.V) * self.V, 0)
        state = self.quantize_seq(roll_X, overlap=None, batch_size=batch_size)
        overlap = state[T // (2 * self.V)] >> self.K * self.V
        state = self.quantize_seq(X, overlap=overlap, batch_size=batch_size)
        hatX = self.recons(state).transpose(0, 1).reshape(X.shape)
        return hatX.T.contiguous().to(X.device), state.T.contiguous().to(X.device)

    def pack_trellis(self, trellis):
        # T is really T // self.V here
        B, T = trellis.shape
        bf = torch.zeros(B,
                         T * self.K * self.V + self.L - self.K * self.V,
                         dtype=bool,
                         device=trellis.device)
        bf[:, :self.L] = (trellis[:, 0].unsqueeze(-1) & (2**torch.arange(
            self.L, device=trellis.device).flip(dims=(-1, ))).unsqueeze(0)) > 0
        K_mask = 2**torch.arange(
            self.K * self.V,
            device=trellis.device).flip(dims=(-1, )).unsqueeze(0)
        for i in range(1, T):
            assert ((trellis[:, i - 1] &
                     ((1 << (self.L - self.K * self.V)) - 1)) == (
                         trellis[:, i] >> (self.K * self.V))).all()
            bf[:,
               (self.L +
                (i - 1) * self.K * self.V):(self.L + i * self.K * self.V)] = (
                    (trellis[:, i] &
                     ((1 <<
                       (self.K * self.V)) - 1)).unsqueeze(-1) & K_mask) > 0

        bf = bf[:, :-(self.L - self.K * self.V)]
        pad_amt = math.ceil(
            T * self.K * self.V / 16) * 16 - T * self.K * self.V
        bf = torch.nn.functional.pad(bf, (0, pad_amt)).reshape(
            -1, (T * self.K * self.V + pad_amt) // 16, 16)

        uint_mask = (2**torch.arange(
            16, dtype=torch.int32,
            device=bf.device)).flip(dims=(-1, )).unsqueeze(0).unsqueeze(0)
        bf_sum = (bf.to(torch.int32) * uint_mask).sum(dim=-1)
        return bf_sum.to(torch.uint16)


    def unpack_trellis(self, packed, T):
        packed = packed.view(torch.uint16).to(torch.int32)
        uint_mask = (2**torch.arange(
            16, dtype=torch.int32,
            device=packed.device)).flip(dims=(-1, )).unsqueeze(0).unsqueeze(0)
        bf = (packed.unsqueeze(-1) & uint_mask) > 0
        pad_amt = math.ceil(T * self.K / 16) * 16 - T * self.K
        bf = bf.reshape(-1, (T * self.K + pad_amt))[:, :T * self.K]
        bf = torch.concat([bf, bf[:, :self.L - self.K * self.V]], dim=-1)
        L_mask = (2**torch.arange(
            self.L, dtype=torch.int32,
            device=packed.device).flip(dims=(-1, ))).unsqueeze(0)
        K_mask = (2**torch.arange(
            self.K * self.V, dtype=torch.int32,
            device=packed.device).flip(dims=(-1, ))).unsqueeze(0)
        trellis = torch.zeros(bf.shape[0],
                              T // self.V,
                              dtype=torch.int32,
                              device=bf.device)
        trellis[:, 0] = (bf[:, :self.L].int() * L_mask).sum(dim=-1)
        for i in range(1, T // self.V):
            trellis[:, i] = ((trellis[:, i-1] << (self.K*self.V)) & ((1 << self.L) - 1)) + \
                (bf[:, self.L + (i-1)*self.K*self.V : self.L + i*self.K*self.V].int() * K_mask).sum(dim=-1)

        return trellis


class TrellisLinear(torch.nn.Module):
    def __init__(self,
                 T=256,
                 L=16,
                 V=1,
                 K=2,
                 tlut_bits=10,
                 decode_mode='1mad',
                 incoh_proc_mode='qtip',
                 viterby_batch_size=256,
                 init_device='cuda:0'):
        super().__init__()
        self.T = T
        self.L = L
        self.V = V
        self.K = K
        self.incoh_proc_mode = incoh_proc_mode
        self.viterby_batch_size = viterby_batch_size
        self.init_device = init_device

        self.codebook = bitshift_codebook(L=L, K=K, V=V, decode_mode=decode_mode, tlut_bits=tlut_bits)
        self.input_quantizer = torch.nn.Identity()


    @torch.no_grad()
    def wrap_module(self, module, *args, **kwargs):
        orig_device = module.weight.device
        self.w_shape = module.weight.shape
        self.codebook = self.codebook.to(self.init_device)
        w = module.weight.to(self.init_device)
        
        
        # Incoherence processing
        if self.incoh_proc_mode == 'qtip':
            w, SU, SV = incoherence_preprocess(w)
            self.SU = torch.nn.Parameter(SU).to(self.init_device)
            self.SV = torch.nn.Parameter(SV).to(self.init_device)
        elif self.incoh_proc_mode == 'skip':
            self.SU = torch.nn.Parameter(torch.tensor(1.0)).to(self.init_device)
            self.SV = torch.nn.Parameter(torch.tensor(1.0)).to(self.init_device)
        else:
            raise RuntimeError
        

        # Scale to Normal(0,1)
        w_std = w.std()
        w = w / w_std
        self.SU *= torch.sqrt(w_std)
        self.SV *= torch.sqrt(w_std)

        # Quantize
        w = w.reshape(-1, self.T).to(self.init_device)
        _, states = self.codebook.quantize(w, batch_size=256)
        self.trellis = self.codebook.pack_trellis(states)

        return deepcopy(self).to(orig_device)


    @property
    def weight(self):
        states = self.codebook.unpack_trellis(self.trellis, T=self.T)
        w = self.codebook.recons(states.T).transpose(0, 1).reshape(-1, states.shape[0]).T
        w = w.reshape(self.w_shape)

        if self.incoh_proc_mode == 'qtip':
            w = incoherence_process(w.float(), self.SU, self.SV)
        elif self.incoh_proc_mode == 'skip':
            w = w * self.SU * self.SV
        else:
            raise RuntimeError
        return w


    def forward(self, x):
        x = self.input_quantizer(x)
        w = self.weight

        return F.linear(weight=w, input=x)


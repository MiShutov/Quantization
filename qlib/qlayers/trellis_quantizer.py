import os
from dataclasses import dataclass
import torch
import torch.nn as nn
from tqdm import tqdm
import math
from qlib.qlayers.kernel_decompress import decode_compressed #DecodeKernelAG
from dataclasses import dataclass
from microxcaling.mx.mx_ops import (
    _get_format_params, _reshape_to_blocks, 
    _shared_exponents, _undo_reshape_to_blocks, quantize_mx_op
)


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


def decode_1mad(x):
    """Special 1MAD decoding function"""
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
    """2D quantized lookup table with sign flipping"""
    with torch.no_grad():
        lut = torch.arange(1 << L, device=tlut.device)
        lut = (lut + 1) * lut
        sflp0 = 1 - ((lut >> 15) & 1) * 2
        sflp1 = 1 - ((lut >> 13) & 1) * 2        
        lut = (lut >> (16 - nbits - 1)) & ((1 << nbits) - 1)
    lut = tlut[lut] * torch.stack([sflp0, sflp1]).T
    return lut


# fname = f'/home/msst/repo/Quantization/ml/weights/LowBitSym_v4_l10.pt'
# _LOW_BIT_LUT_CACHED = torch.load(fname, weights_only=True).to(torch.float16).cuda().contiguous()
# _EXPANDED_LUT_CACHED = quantlut_sym_2d(_LOW_BIT_LUT_CACHED, 16, 10).to(torch.float16).cuda().contiguous()


@dataclass
class TrellisQuantizerParams:
    T: int = 256,
    L: int = 16,
    V: int = 2,
    K: int = 2,
    decode_mode: str = "LowBitSym",
    tlut_bits: int = 10,
    viterbi_bs: int = "auto"


class TrellisQuantizer(nn.Module):
    def __init__(self,
                 params: TrellisQuantizerParams,
                 **kwargs):
        super(TrellisQuantizer, self).__init__()
        self.idx_dtype = torch.int32

        self.L = params.L
        self.K = params.K
        self.V = params.V
        self.T = params.T
        self.tlut_bits = params.tlut_bits
        self.decode_mode = params.decode_mode

        self.mx_mode = False

        # Adaptive batch sizing
        if params.viterbi_bs == 'auto':
            self.viterbi_bs = min(2**(24 - self.L), 256)
        else:
            self.viterbi_bs = params.viterbi_bs

        if self.decode_mode == '1mad':
            assert self.V == 1
            self.register_buffer('lut',
                               decode_1mad(torch.arange(2**self.L)).unsqueeze(-1))
        
        elif self.decode_mode == 'Rand2d':
            assert self.V == 2
            fname = f'/home/msst/repo/Quantization/ml/weights/Rand2d_{self.L}.pt'
            if not os.path.exists(fname):
                lut = torch.randn(2**self.L, 2)
                torch.save(lut, fname)
            else:
                lut = torch.load(fname, weights_only=True)
            self.register_buffer('lut', lut)

        elif self.decode_mode == 'LowBitSym':
            assert self.V == 2
            assert self.tlut_bits in [10, 12] 
            
            if self.tlut_bits==10:
                values_bits = 4
            else:
                values_bits = 5
            
            bound = 3.0 # 3 standard deviation

            fname = f'/home/msst/repo/Quantization/ml/weights/LowBitSym_v{values_bits}_l{self.tlut_bits}.pt'
            if not os.path.exists(fname):
                tlut, scale = get_positive_lowbit_codebook(2**self.tlut_bits, values_bits=values_bits, bound=bound)
                torch.save({"tlut": tlut, "scale" : scale}, fname)
            else:
                #tlut = torch.load(fname, weights_only=True)
                cb_dict = torch.load(fname, weights_only=True)
                tlut = cb_dict['tlut']
                scale = cb_dict['scale']
            self.codebook_scale = scale
            self.register_buffer('tlut', tlut)
            self.register_buffer(
                'lut',
                quantlut_sym_2d(self.tlut, self.L, self.tlut_bits).contiguous())

        elif self.decode_mode == 'QuantlutSym':
            if tlut is None:
                assert self.tlut_bits > 0
                assert self.V == 2
                fname = f'/home/msst/repo/Quantization/ml/weights/QuantlutSym_{self.tlut_bits}_{self.V}.pt'
                if not os.path.exists(fname):
                    tlut = torch.randn(2**self.tlut_bits, self.V)
                    import scipy
                    data = torch.randn(1 << 20, 2)
                    clusters = scipy.cluster.vq.kmeans(data.to(torch.float32), tlut.to(torch.float32))
                    tlut = torch.tensor(clusters[0])
                    tlut = (tlut /
                            tlut.std(unbiased=False)) * 0.9682458365518543
                    torch.save(tlut, fname)
                else:
                    tlut = torch.load(fname, weights_only=True)

                self.register_buffer('tlut', tlut)
                self.register_buffer(
                    'lut',
                    quantlut_sym(self.tlut, self.L, self.tlut_bits).contiguous())


        elif self.decode_mode == 'load_tlut':
            self.register_buffer('tlut', kwargs["tlut"])
            self.register_buffer(
                'lut',
                quantlut_sym_2d(self.tlut, self.L, self.tlut_bits).contiguous())

        elif self.decode_mode.startswith('mx_'):
            self.mx_mode = True
            self.w_elem_format = self.decode_mode.split('mx_')[1]
        else:
            raise Exception


    def init_lut_mx(self, w, shared_exp, mx_specs):
        axes=[-1]
        qis_weight = quantize_mx_op(
            w,
            mx_specs,
            elem_format=mx_specs['w_elem_format'],
            axes=axes,
            round=mx_specs["round_mx_output"],
        )

        _qis_weight, axes, orig_shape, padded_shape = _reshape_to_blocks(qis_weight, axes, mx_specs["block_size"])
        _qis_weight_scaled = _qis_weight / 2**shared_exp

        with torch.no_grad():
            uniq_pos = torch.unique(_qis_weight_scaled.abs())

        base_codebook_size = 2**self.tlut_bits
        sample_values = int(base_codebook_size * 1.5)

        counts = torch.zeros_like(uniq_pos).to(torch.int32)
        for i, v in enumerate(uniq_pos):
            counts[i] = (_qis_weight_scaled.abs().cpu()==v.cpu()).sum()
            # if v == 0.0:
            #     counts[i] /= 2

        freq = (counts / counts.sum()).unsqueeze(0)


        freq_2d = freq.T @ freq
        counts = (freq_2d * sample_values / freq_2d.sum())
        counts = counts.round()
        counts = counts.flatten()

        unique_cb_h = uniq_pos.repeat(len(uniq_pos), 1)
        unique_cb_v = unique_cb_h.T
        unique_cb_2d = torch.stack([unique_cb_v, unique_cb_h], dim=0)

        unique_cb = unique_cb_2d.reshape(2, -1).T

        cb = []
        for i, c in enumerate(counts):
            cb += int(c) * [unique_cb[i],]
            
        cb = torch.stack(cb)
        n_to_remove = len(cb)- base_codebook_size
        torch.manual_seed(0)
        tlut = cb[torch.randperm(len(cb))][n_to_remove:]
        self.register_buffer('tlut', tlut)
        self.register_buffer(
            'lut',
            quantlut_sym_2d(self.tlut, self.L, self.tlut_bits).contiguous())


    def recons(self, encoded, **kwargs):
        """Reconstruct values from encoded states"""
        return self.lut[encoded.int().to(self.lut.device)].to(encoded.device)

    @torch.compile
    def update(self, cost, orig_seq_part, state_candidates):
        B = orig_seq_part.shape[0]  # batch size
        R = 2 ** (self.L - self.K * self.V)  # reduced state size
        D = 2 ** (self.K * self.V)  # delta size
        S = 2 ** self.L  # total states

        # Calculate state reconstruction error (B, S)
        state_err = (self.lut - orig_seq_part.unsqueeze(1)).square().sum(dim=-1)

        # Reshape cost to (B, 1, S) for gathering
        cost_expanded = cost.view(B, 1, S).expand(-1, R, -1) 

        # Prepare candidate indices (B, R, D)
        candidates = state_candidates.expand(B, R, D)

        # Gather candidate costs (B, R, D)
        cand_cost = torch.gather(
            input=cost_expanded, # (B, R, S)
            dim=-1,
            index=candidates  # (B, R, D)
        )

        # Find best candidate for each reduced state (B, R)
        best = torch.min(cand_cost, dim=-1)

        # Update cost (B, S)
        cost = state_err + best.values.view(B, R, 1).expand(-1, -1, D).reshape(B, S)

        # Get previous states (B, R)
        prev_state = torch.gather(
            input=candidates,
            dim=-1,
            index=best.indices.unsqueeze(-1)
        )[..., 0]

        return prev_state, cost


    def viterbi(self, X, overlap=None):
        """Optimized Viterbi decoding with time-major storage"""

        # State transition buffers
        sumdelta =  (torch.arange(2**(self.K * self.V), device=X.device) << (self.L - self.K * self.V)).view(1, 1, -1)
      
        # State candidates: maps (reduced_state, delta) -> full_state
        # Shape: (1, 2^(L-K*V), 2^(K*V))
        state_candidates = (torch.arange(2**self.L, device=X.device).unsqueeze(0) >> (self.K * self.V))[0, ::2**(self.K * self.V)].unsqueeze(-1) + sumdelta

        B = X.shape[0]
        T_v = self.T // self.V
        fakeinf = torch.tensor(torch.inf)

        # Forward pass
        cost = (self.lut - X[:, :self.V].unsqueeze(1)).square().sum(dim=-1)
        
        if overlap is not None:
            mask = torch.ones(B, 2**self.L, device=X.device) * fakeinf
            allow = (overlap << (self.K * self.V)).unsqueeze(-1) + torch.arange(
                         2**(self.K * self.V)).to(X.device).view(1, 1, -1)
            mask.scatter_(1, allow[0], 0)
            cost = torch.min(cost + mask, fakeinf)

        # Time-major storage for efficient backtrace
        from_state = torch.zeros(T_v, B, 2**(self.L - self.K * self.V), 
                               dtype=torch.long, device=X.device)
        
        for i in range(1, T_v):
            obs = X[:, i*self.V:(i+1)*self.V]
            prev_state, cost = self.update(cost, obs, state_candidates)
            from_state[i] = prev_state

        if overlap is not None:
            mask = torch.ones(B, 2**self.L, device=X.device) * fakeinf
            allow = (overlap.unsqueeze(-1) + sumdelta.unsqueeze(0))
            mask.scatter_(1, allow[0, 0], 0)
            cost = torch.min(cost + mask, fakeinf)

        # Backtrace
        final_state = torch.zeros(T_v, B, dtype=self.idx_dtype, device=X.device)
        final_state[T_v - 1] = torch.argmin(cost, dim=-1)
        
        for i in range(T_v - 1, 0, -1):
            reduced_idx = (final_state[i] >> (self.K * self.V)).long().unsqueeze(1)
            final_state[i-1] = torch.gather(from_state[i], 1, reduced_idx).squeeze(1)
            
        return final_state.transpose(0, 1)  # Return as (B, T_v)


    def quantize_seq(self, X, overlap=None, **kwargs):
        """Quantize sequence with batch processing"""
        n_seq, T = X.shape
        batch_padding_len = math.ceil(n_seq / self.viterbi_bs) * self.viterbi_bs - n_seq
        X = torch.nn.functional.pad(X.T, (0, batch_padding_len)).T

        n_seq_padded = X.shape[0]
        X = X.reshape(n_seq_padded // self.viterbi_bs, self.viterbi_bs, T).contiguous()
        if overlap is not None:
            overlap = torch.nn.functional.pad(overlap, (0, batch_padding_len))
            overlap = overlap.reshape(n_seq_padded // self.viterbi_bs, self.viterbi_bs)

        Qidxs = torch.zeros(n_seq_padded // self.viterbi_bs,
                          self.viterbi_bs,
                          T // self.V,
                          dtype=self.idx_dtype,
                          device=X.device)
        for i in tqdm(range(len(X))):
            overlap_batch = None if overlap is None else overlap[i]
            Qidxs[i] = self.viterbi(X[i], overlap=overlap_batch)
        Qidxs = Qidxs.reshape(n_seq_padded, T // self.V)[:n_seq]
        return Qidxs


    def quantize(self, X, batch_size='auto', **kwargs):
        X_shape = X.shape
        assert self.T == 256
        

        X = X.reshape(-1, self.T)
        if hasattr(self, "codebook_scale"):
            X = X / self.codebook_scale

        # Fisrt fase
        roll_X = torch.roll(X, self.T // (2 * self.V) * self.V, 1)
        state = self.quantize_seq(roll_X, overlap=None, batch_size=batch_size)
        overlap = state[:, self.T // (2 * self.V)] >> self.K * self.V
        # Second fase
        state = self.quantize_seq(X, overlap=overlap, batch_size=batch_size)
        
        hatX = self.recons(state).reshape(X_shape)
        return hatX.contiguous().to(X.device), state.contiguous().to(X.device)


    def pack_trellis(self, trellis):
        # T is really T // self.V here
        B, T = trellis.shape
        assert T != self.T
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


    def unpack_trellis(self, packed):
        packed = packed.view(torch.uint16).to(torch.int32)
        uint_mask = (2**torch.arange(
            16, dtype=torch.int32,
            device=packed.device)).flip(dims=(-1, )).unsqueeze(0).unsqueeze(0)
        bf = (packed.unsqueeze(-1) & uint_mask) > 0
        pad_amt = math.ceil(self.T * self.K / 16) * 16 - self.T * self.K
        bf = bf.reshape(-1, (self.T * self.K + pad_amt))[:, :self.T * self.K]
        bf = torch.concat([bf, bf[:, :self.L - self.K * self.V]], dim=-1)
        L_mask = (2**torch.arange(
            self.L, dtype=torch.int32,
            device=packed.device).flip(dims=(-1, ))).unsqueeze(0)
        K_mask = (2**torch.arange(
            self.K * self.V, dtype=torch.int32,
            device=packed.device).flip(dims=(-1, ))).unsqueeze(0)
        trellis = torch.zeros(bf.shape[0],
                              self.T // self.V,
                              dtype=torch.int32,
                              device=bf.device)
        trellis[:, 0] = (bf[:, :self.L].int() * L_mask).sum(dim=-1)
        for i in range(1, self.T // self.V):
            trellis[:, i] = ((trellis[:, i-1] << (self.K*self.V)) & ((1 << self.L) - 1)) + \
                (bf[:, self.L + (i-1)*self.K*self.V : self.L + i*self.K*self.V].int() * K_mask).sum(dim=-1)

        return trellis


    def reconstruct_weight(self, packed_trellis, w_shape):
        unpacked_trellis = self.unpack_trellis(packed_trellis)
        
        w_reco = self.recons(unpacked_trellis).reshape(w_shape)
        return w_reco

    
    def reconstruct_weight_fast(self, packed_trellis, w_shape):
        assert self.decode_mode=='LowBitSym'
        assert self.L == 16
        return decode_compressed(
            L=self.L,
            K=self.K,
            V=self.V,
            m=w_shape[0],
            n=w_shape[1],
            compressed=packed_trellis.view(-1),
            expanded_lut=self.lut 
        )

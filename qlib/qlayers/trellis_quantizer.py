import os
import torch
import torch.nn as nn
from tqdm import tqdm
import math
from qlib.qlayers.kernel_decompress import decode_compressed


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


def quantlut_sym_2d(tlut, L, nbits):
    """2D quantized lookup table with sign flipping"""
    with torch.no_grad():
        lut = torch.arange(1 << L, device=tlut.device)
        lut = (lut + 1) * lut
        sflp0 = 1 - ((lut >> 15) & 1) * 2
        sflp1 = 1 - ((lut >> 7) & 1) * 2
        lut = (lut >> (16 - nbits - 1)) & ((1 << nbits) - 1)
    lut = tlut[lut] * torch.stack([sflp0, sflp1]).T
    return lut


class trellis_quantizer(nn.Module):
    def __init__(self,
                 L=16,
                 K=2,
                 V=2,
                 T=256,
                 decode_mode='1mad',
                 tlut_bits=10,
                 tlut=None,
                 viterby_bs='auto'):
        super(trellis_quantizer, self).__init__()
        self.idx_dtype = torch.int32

        self.L = L
        self.K = K
        self.V = V
        self.T = T
        self.tlut_bits = tlut_bits
        self.decode_mode = decode_mode
        
        # Adaptive batch sizing
        if viterby_bs == 'auto':
            self.viterby_bs = min(2**(24 - self.L), 256)
        else:
            self.viterby_bs = viterby_bs

        if decode_mode == '1mad':
            assert V == 1
            self.register_buffer('lut',
                               decode_1mad(torch.arange(2**L)).unsqueeze(-1))
        
        elif decode_mode == 'LowBitSym':
            assert self.V == 2
            assert tlut_bits > 0
            tlut, scale = get_positive_lowbit_codebook(2**tlut_bits, values_bits=4, bound=3.0)
            self.register_buffer('tlut', tlut)
            self.register_buffer(
                'lut',
                quantlut_sym_2d(self.tlut, L, tlut_bits).contiguous())

        else:
            raise Exception

        # State transition buffers
        self.register_buffer('sumdelta', (torch.arange(2**(K * V)) << (L - K * V)).view(1, 1, -1))
        
        # State candidates: maps (reduced_state, delta) -> full_state
        # Shape: (1, 2^(L-K*V), 2^(K*V))
        self.register_buffer('state_candidates',
                           (torch.arange(2**L).unsqueeze(0) >> (K * V))[0, ::2**(K * V)].unsqueeze(-1) + self.sumdelta)
        

    def recons(self, encoded, **kwargs):
        """Reconstruct values from encoded states"""
        return self.lut[encoded.int().to(self.lut.device)].to(encoded.device)

    @torch.compile
    def update(self, cost, orig_seq_part):
        B = orig_seq_part.shape[0]  # batch size
        R = 2 ** (self.L - self.K * self.V)  # reduced state size
        D = 2 ** (self.K * self.V)  # delta size
        S = 2 ** self.L  # total states

        # Calculate state reconstruction error (B, S)
        state_err = (self.lut - orig_seq_part.unsqueeze(1)).square().sum(dim=-1)

        # Reshape cost to (B, 1, S) for gathering
        cost_expanded = cost.view(B, 1, S).expand(-1, R, -1) 

        # Prepare candidate indices (B, R, D)
        candidates = self.state_candidates.expand(B, R, D)

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
            prev_state, cost = self.update(cost, obs)
            from_state[i] = prev_state

        if overlap is not None:
            mask = torch.ones(B, 2**self.L, device=X.device) * fakeinf
            allow = (overlap.unsqueeze(-1) + self.sumdelta.unsqueeze(0))
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
        batch_padding_len = math.ceil(n_seq / self.viterby_bs) * self.viterby_bs - n_seq
        X = torch.nn.functional.pad(X.T, (0, batch_padding_len)).T

        n_seq_padded = X.shape[0]
        X = X.reshape(n_seq_padded // self.viterby_bs, self.viterby_bs, T).contiguous()
        if overlap is not None:
            overlap = torch.nn.functional.pad(overlap, (0, batch_padding_len))
            overlap = overlap.reshape(n_seq_padded // self.viterby_bs, self.viterby_bs)

        Qidxs = torch.zeros(n_seq_padded // self.viterby_bs,
                          self.viterby_bs,
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
        # # Set vector as 16 x 16 patch
        # patch_size = 16
        # X = X.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
        # X = X.reshape(-1, self.T).contiguous().to(torch.float16)
        # X = X.contiguous().view(-1, patch_size, patch_size)
        # X = X.contiguous().view(-1, patch_size * patch_size)

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

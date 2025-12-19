import math
import os
from enum import Enum

import torch
import torch.nn as nn


def big_index_to_small_index(big_index, small_index_size):
    big_index = (big_index + 1) * big_index
    return (big_index >> (16 - small_index_size - 1)) & ((1 << small_index_size) - 1)


def custom_permute(x):
    x = x * 34038481 + 76625530
    return x >> 9 & 0xFFFF


def custom_permute_fast(x, a=34038481):
    """
    x - is int16 value == 16 arbitrary bits
    """
    # int multiplication: int16 x int32 -> int32 accumulator
    x = x * a
    # get bits from 9 to 25 -> new int16 value (pseudorandom value ~= random permutation)
    return x >> 9 & 0xFFFF


def fp4_s1e2m1_subnormal(x4):
    s = (x4 >> 3) & 0b1
    e = (x4 >> 1) & 0b11
    m = x4 & 0b1
    value = torch.where(e == 0, 0.5 * m.float(), (1.0 + 0.5 * m.float()) * (2.0 ** (e.float() - 1)))
    value = value * torch.where(s == 0, 1.0, -1.0)
    return value


def int4_s1m3(x4):
    s = (x4 >> 3) & 0b1
    m = x4 & 0b111
    value = m.float() / 4.0  # exponent shift = -2
    value = torch.where(s == 0, value, -value)
    return value


def int4_to_mx_value(x, mx_decoder):
    x = x.int() & 0b1111
    return mx_decoder(x)


def int8_to_mx_pair(x, mx_decoder):
    x = x.int() & 0xFF
    x1 = (x >> 4) & 0b1111
    x2 = x & 0b1111
    return mx_decoder(torch.stack([x1, x2], dim=-1))


def int16_to_mx_four(x, mx_decoder):
    x = x.int() & 0xFFFF
    x1 = x & 0b1111
    x2 = (x >> 4) & 0b1111
    x3 = (x >> 8) & 0b1111
    x4 = (x >> 12) & 0b1111
    return mx_decoder(torch.stack([x4, x3, x2, x1], dim=-1))

def int16_to_fp8_pair(x):
    x = x.int() & 0xFFFF # check int16

    mask = 0b10011110
    x1 = x & mask
    x2 = (x >> 8) & mask
    return torch.stack([x2, x1], dim=-1).to(torch.uint8).view(torch.float8_e4m3fn).float()


def int8_to_fp8(x):
    x = x.int() & 0xFF # check uint8

    mask = 0b10011110
    x = x & mask

    return x.to(torch.uint8).view(torch.float8_e4m3fn).float()


class TrellisQuantizerValues(Enum):
    QTIP_HYB = 1
    LUTFREE_FP4 = 2
    LUTFREE_FP8 = 3


class CodebookLUTfreeFP8:
    def __init__(self, L, V):
        assert L == 16
        assert V in [1, 2, 4]
        self.L = L
        self.V = V

    def get_training_lut(self):
        if self.V == 1:
            table = big_index_to_small_index(torch.arange(1 << self.L), small_index_size=8)
            training_lut = int8_to_fp8(table).unsqueeze(-1)

        elif self.V == 2:
            table = custom_permute_fast(torch.arange(1 << self.L))
            training_lut = int16_to_fp8_pair(table)

        elif self.V == 4:
            table1 = custom_permute_fast(torch.arange(1 << self.L))
            table2 = custom_permute_fast(torch.arange(1 << self.L), a=8675309)
            training_lut = torch.cat([int16_to_fp8_pair(table1),  int16_to_fp8_pair(table2)], dim=-1)

        return training_lut / training_lut.std()


class CodebookLUTfreeFP4:
    def __init__(self, L, V):
        assert L == 16
        assert V in [1, 2, 4]
        self.L = L
        self.V = V

    def get_training_lut(self):
        if self.V == 1:
            table = big_index_to_small_index(torch.arange(1 << self.L), small_index_size=4)
            training_lut = int4_to_mx_value(table, fp4_s1e2m1_subnormal).unsqueeze(-1)
        elif self.V == 2:
            table = big_index_to_small_index(torch.arange(1 << self.L), small_index_size=8)
            training_lut = int8_to_mx_pair(table, fp4_s1e2m1_subnormal)
        elif self.V == 4:
            table = custom_permute_fast(torch.arange(1 << self.L))
            training_lut = int16_to_mx_four(table, fp4_s1e2m1_subnormal)

        return training_lut / training_lut.std()


class CodebookQTIPhyb(nn.Module):
    def __init__(self, L, V):
        super().__init__()
        assert L == 16
        assert V == 2
        self.L = L
        self.V = V
        self.lut_vector_bits = 9

        pretrained_model_path = os.environ.get("PRETRAINED_MODEL_PATH")
        if pretrained_model_path is None:
            raise EnvironmentError("Please set environment variable PRETRAINED_MODEL_PATH!")

        path_to_init_lut = os.path.join(os.path.dirname(pretrained_model_path), "kmeans", "tlut_QTIP_hyb_V2.pth")

        if not os.path.exists(path_to_init_lut):
            import scipy

            data = torch.randn(1 << 20, 2)
            lut = torch.randn(2**self.lut_vector_bits, V)
            clusters = scipy.cluster.vq.kmeans(data, lut)
            lut = torch.tensor(clusters[0])
            lut = (lut / lut.std(unbiased=False)) * 0.9682458365518543
            torch.save(lut, path_to_init_lut)
        else:
            lut = torch.load(path_to_init_lut)

        self.lut = torch.nn.Parameter(lut, requires_grad=True)

    def get_training_lut(self):
        with torch.no_grad():
            table = torch.arange(1 << self.L, device=self.lut.device)
            table = (table + 1) * table
            sflp = 1 - ((table >> 15) & 1) * 2
            table = (table >> (16 - self.lut_vector_bits - 1)) & ((1 << self.lut_vector_bits) - 1)
        training_lut = self.lut[table]
        training_lut[:, 0] = training_lut[:, 0] * sflp
        return training_lut


class TrellisQuantizer(nn.Module):
    """
    Trellis Quantizer for weight compression. Compressed weight == trellis compressed bits in uint16
    """
    def __init__(
        self,
        values_type: str,
        T: int = 256,
        L: int = 16,
        V: int = 2,
        K: int = 2,
        viterbi_batch_size=4096,
        use_kernel=False,
    ):
        super().__init__()

        self.values_type = TrellisQuantizerValues[values_type]
        self.idx_dtype = torch.int32
        self.T = T
        self.L = L
        self.K = K
        self.V = V
        self.viterbi_bs = viterbi_batch_size

        if self.values_type == TrellisQuantizerValues.QTIP_HYB:
            self.codebook = CodebookQTIPhyb(L=self.L, V=self.V)
        elif self.values_type == TrellisQuantizerValues.LUTFREE_FP4:
            self.codebook = CodebookLUTfreeFP4(L=self.L, V=self.V)
        elif self.values_type == TrellisQuantizerValues.LUTFREE_FP8:
            self.codebook = CodebookLUTfreeFP8(L=self.L, V=self.V)
        else:
            raise

        self.use_kernel = use_kernel

    def get_compressed_weight_params(self, weight_shape):
        """
        return (compressed_weight_shape, compressed_weight_dtype)
        """
        self.weight_shape = weight_shape

        trellis_shape = [self.weight_shape[0] * self.weight_shape[1] // self.T, self.T * self.K // 16]
        return trellis_shape, torch.uint16

    @torch.compile
    def update(self, training_lut, cost, orig_seq_part, state_candidates):
        B = orig_seq_part.shape[0]  # batch size
        R = 2 ** (self.L - self.K * self.V)  # reduced state size
        D = 2 ** (self.K * self.V)  # delta size
        S = 2 ** self.L  # total states

        # Gather candidate costs (B, R, D)
        cand_cost = torch.gather(
            input=cost.view(B, 1, S).expand(-1, R, -1), 
            dim=-1, 
            index=state_candidates.expand(B, R, D)
        )

        # Find best candidate for each reduced state (B, R)
        best = torch.min(cand_cost, dim=-1)

        # Calculate state reconstruction error (B, S)
        state_err = (training_lut - orig_seq_part.unsqueeze(1)).square().sum(dim=-1)

        # Update cost (B, S)
        cost = state_err + best.values.view(B, R, 1).expand(-1, -1, D).reshape(B, S)

        # Get previous states (B, R)
        prev_state = torch.gather(
            input=state_candidates.expand(B, R, D), 
            dim=-1, 
            index=best.indices.unsqueeze(-1)
        )[..., 0]

        return prev_state, cost

    def viterbi(self, training_lut, X, overlap=None):
        """Optimized Viterbi decoding with time-major storage"""

        # State transition buffers
        sumdelta = (torch.arange(2 ** (self.K * self.V), device=X.device) << (self.L - self.K * self.V)).view(1, 1, -1)

        # State candidates: maps (reduced_state, delta) -> full_state
        # Shape: (1, 2^(L-K*V), 2^(K*V))
        state_candidates = (torch.arange(2**self.L, device=X.device).unsqueeze(0) >> (self.K * self.V))[
            0, :: 2 ** (self.K * self.V)
        ].unsqueeze(-1) + sumdelta

        B = X.shape[0]
        T_v = self.T // self.V
        fakeinf = torch.tensor(torch.inf)

        # Forward pass
        cost = (training_lut - X[:, : self.V].unsqueeze(1)).square().sum(dim=-1)

        if overlap is not None:
            mask = torch.ones(B, 2**self.L, device=X.device) * fakeinf
            allow = (overlap << (self.K * self.V)).unsqueeze(-1) + torch.arange(2 ** (self.K * self.V)).to(
                X.device
            ).view(1, 1, -1)
            mask.scatter_(1, allow[0], 0)
            cost = torch.min(cost + mask, fakeinf)

        # Time-major storage for efficient backtrace
        from_state = torch.zeros(T_v, B, 2 ** (self.L - self.K * self.V), dtype=torch.long, device=X.device)

        for i in range(1, T_v):
            obs = X[:, i * self.V : (i + 1) * self.V]
            prev_state, cost = self.update(
                training_lut.to(torch.float32),
                cost.to(torch.float32),
                obs.to(torch.float32),
                state_candidates,
            )
            from_state[i] = prev_state

        if overlap is not None:
            mask = torch.ones(B, 2**self.L, device=X.device) * fakeinf
            allow = overlap.unsqueeze(-1) + sumdelta.unsqueeze(0)
            mask.scatter_(1, allow[0, 0], 0)
            cost = torch.min(cost + mask, fakeinf)

        # Backtrace
        final_state = torch.zeros(T_v, B, dtype=self.idx_dtype, device=X.device)
        final_state[T_v - 1] = torch.argmin(cost, dim=-1)

        for i in range(T_v - 1, 0, -1):
            reduced_idx = (final_state[i] >> (self.K * self.V)).long().unsqueeze(1)
            final_state[i - 1] = torch.gather(from_state[i], 1, reduced_idx).squeeze(1)

        return final_state.transpose(0, 1)  # Return as (B, T_v)

    def quantize_seq(self, training_lut, X, overlap=None, **kwargs):
        """Quantize sequence with batch processing"""
        n_seq, T = X.shape
        batch_padding_len = math.ceil(n_seq / self.viterbi_bs) * self.viterbi_bs - n_seq
        X = torch.nn.functional.pad(X.T, (0, batch_padding_len)).T

        n_seq_padded = X.shape[0]
        X = X.reshape(n_seq_padded // self.viterbi_bs, self.viterbi_bs, T).contiguous()
        if overlap is not None:
            overlap = torch.nn.functional.pad(overlap, (0, batch_padding_len))
            overlap = overlap.reshape(n_seq_padded // self.viterbi_bs, self.viterbi_bs)

        Qidxs = torch.zeros(
            n_seq_padded // self.viterbi_bs, self.viterbi_bs, T // self.V, dtype=self.idx_dtype, device=X.device
        )
        for i in range(len(X)):
            overlap_batch = None if overlap is None else overlap[i]
            Qidxs[i] = self.viterbi(training_lut, X[i], overlap=overlap_batch)
        Qidxs = Qidxs.reshape(n_seq_padded, T // self.V)[:n_seq]
        return Qidxs

    def quantize(self, X, return_reco=False, **kwargs):
        training_lut = self.codebook.get_training_lut().to(X.device)
        if self.use_kernel:
            m, n = X.shape
            assert self.L == 16
            assert self.T == 256
            X = weight_swizzling(X)
        else:
            X = X.reshape(-1, self.T)

        # Fisrt fase
        roll_X = torch.roll(X, self.T // (2 * self.V) * self.V, 1)
        state = self.quantize_seq(training_lut, roll_X, overlap=None)
        overlap = state[:, self.T // (2 * self.V)] >> self.K * self.V

        # Second fase
        state = self.quantize_seq(training_lut, X, overlap=overlap)

        # state = self.quantize_seq(training_lut, X)

        if return_reco:
            return training_lut[state.int().to(training_lut.device)].to(state.device)

        trellis = self.pack_states(state)
        if self.use_kernel:
            trellis = trellis_swizzling(trellis, m, n, self.K)
        return trellis

    def dequantize(self, trellis, **kwargs):
        training_lut = self.codebook.get_training_lut().to(trellis.device)
        if self.use_kernel:
            return decode_compressed(
                self.L, self.K, self.V, self.weight_shape[0], self.weight_shape[1], trellis.view(-1), training_lut
            )
        else:
            states = self.unpack_states(trellis)
            w_reco = training_lut[states.int().to(training_lut.device)].to(states.device)
            return w_reco.reshape(self.weight_shape)

    def pack_states(self, states):
        B, T = states.shape
        bf = torch.zeros(B, T * self.K * self.V + self.L - self.K * self.V, dtype=bool, device=states.device)
        bf[:, : self.L] = (
            states[:, 0].unsqueeze(-1)
            & (2 ** torch.arange(self.L, device=states.device).flip(dims=(-1,))).unsqueeze(0)
        ) > 0
        K_mask = 2 ** torch.arange(self.K * self.V, device=states.device).flip(dims=(-1,)).unsqueeze(0)
        for i in range(1, T):
            assert (
                (states[:, i - 1] & ((1 << (self.L - self.K * self.V)) - 1)) == (states[:, i] >> (self.K * self.V))
            ).all()
            bf[:, (self.L + (i - 1) * self.K * self.V) : (self.L + i * self.K * self.V)] = (
                (states[:, i] & ((1 << (self.K * self.V)) - 1)).unsqueeze(-1) & K_mask
            ) > 0

        bf = bf[:, : -(self.L - self.K * self.V)]
        pad_amt = math.ceil(T * self.K * self.V / 16) * 16 - T * self.K * self.V
        bf = torch.nn.functional.pad(bf, (0, pad_amt)).reshape(-1, (T * self.K * self.V + pad_amt) // 16, 16)

        uint_mask = (
            (2 ** torch.arange(16, dtype=torch.int32, device=bf.device)).flip(dims=(-1,)).unsqueeze(0).unsqueeze(0)
        )
        bf_sum = (bf.to(torch.int32) * uint_mask).sum(dim=-1)
        return bf_sum.to(torch.uint16)


    def unpack_states(self, states):
        packed = states.view(torch.uint16).to(torch.int32)
        uint_mask = (
            (2 ** torch.arange(16, dtype=torch.int32, device=packed.device)).flip(dims=(-1,)).unsqueeze(0).unsqueeze(0)
        )
        bf = (packed.unsqueeze(-1) & uint_mask) > 0
        pad_amt = math.ceil(self.T * self.K / 16) * 16 - self.T * self.K
        bf = bf.reshape(-1, (self.T * self.K + pad_amt))[:, : self.T * self.K]
        bf = torch.concat([bf, bf[:, : self.L - self.K * self.V]], dim=-1)
        L_mask = (2 ** torch.arange(self.L, dtype=torch.int32, device=packed.device).flip(dims=(-1,))).unsqueeze(0)
        K_mask = (
            2 ** torch.arange(self.K * self.V, dtype=torch.int32, device=packed.device).flip(dims=(-1,))
        ).unsqueeze(0)
        states = torch.zeros(bf.shape[0], self.T // self.V, dtype=torch.int32, device=bf.device)
        states[:, 0] = (bf[:, : self.L].int() * L_mask).sum(dim=-1)
        for i in range(1, self.T // self.V):
            states[:, i] = ((states[:, i - 1] << (self.K * self.V)) & ((1 << self.L) - 1)) + (
                bf[:, self.L + (i - 1) * self.K * self.V : self.L + i * self.K * self.V].int() * K_mask
            ).sum(dim=-1)

        return states


    def ste_forward(self, x, update_compressed_weight=False, compressed_weight_ptr=None):
        """
        compressed_weight_ptr is a pointer to torch.nn.Parameter with comporessed weight data
        """
        if (update_compressed_weight == False) and (compressed_weight_ptr is None):
            trellis = self.quantize(x)
            x_q = self.dequantize(trellis)
            x_q_ste = x_q + (x - x.detach())
            return x_q_ste
        elif (update_compressed_weight == True) and (compressed_weight_ptr is not None):
            trellis = self.quantize(x)
            compressed_weight_ptr.copy_(trellis)
            x_q = self.dequantize(trellis)
            x_q_ste = x_q + (x - x.detach())
            return x_q_ste
        elif (update_compressed_weight == False) and (compressed_weight_ptr is not None):
            x_q = self.dequantize(compressed_weight_ptr.data)
            x_q_ste = x_q + (x - x.detach())
            return x_q_ste
        else:  # If update_compressed_weight==True, compressed_weight_ptr is None
            raise RuntimeError("Can't update compressed weight without compressed_weight_ptr!")
# import os
# import torch
# import torch.nn as nn
# from tqdm import tqdm
# import math


# def get_positive_lowbit_codebook(base_codebook_size, values_bits, bound):
#     sample_values = int(base_codebook_size * 1.5)
#     scale = bound / ((2**(values_bits-1)) - 0.5)

#     quantiles = torch.special.ndtr(scale * (torch.arange(2**(values_bits-1))))
#     quantiles_padded = torch.tensor(list(quantiles) + [1])
#     freq = (quantiles_padded[1:] - quantiles_padded[:-1]).unsqueeze(0)
#     freq_2d = freq.T @ freq

#     counts = (freq_2d * sample_values / freq_2d.sum())
#     counts = counts.round()
#     #counts = counts.to(torch.int) + 1
#     counts = counts.flatten()

#     unique_values = scale * (torch.arange(2**(values_bits-1)) + 0.5)
#     unique_cb_h = unique_values.repeat(len(unique_values), 1)
#     unique_cb_v = unique_cb_h.T
#     unique_cb_2d = torch.stack([unique_cb_v, unique_cb_h], dim=0)

#     unique_cb = unique_cb_2d.reshape(2, -1).T

#     cb = []
#     for i, c in enumerate(counts):
#         cb += int(c) * [unique_cb[i],]
        
#     cb = torch.stack(cb)
#     n_to_remove = len(cb)- base_codebook_size
#     cb = cb[torch.randperm(len(cb))][n_to_remove:]

#     return cb, scale


# def decode_1mad(x):
#     x = x.to(torch.int64)
#     x = x & ((1 << 32) - 1)
#     x = x * 34038481 + 76625530
#     x = x & ((1 << 32) - 1)
#     y = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)
#     y = y - 510
#     y = y.to(torch.float32)
#     y = y / 147.800537109375
#     return y


# def quantlut_sym(tlut, L, nbits):
#     with torch.no_grad():
#         lut = torch.arange(1 << L, device=tlut.device)
#         lut = (lut + 1) * lut
#         sflp = 1 - ((lut >> 15) & 1) * 2
#         lut = (lut >> (16 - nbits - 1)) & ((1 << nbits) - 1)
#     lut = tlut[lut]
#     lut[:, 0] = lut[:, 0] * sflp
#     return lut


# def quantlut_sym_2d(tlut, L, nbits):
#     with torch.no_grad():
#         lut = torch.arange(1 << L, device=tlut.device)
#         lut = (lut + 1) * lut
#         sflp0 = 1 - ((lut >> 15) & 1) * 2
#         sflp1 = 1 - ((lut >> 7) & 1) * 2
#         lut = (lut >> (16 - nbits - 1)) & ((1 << nbits) - 1)
#     lut = tlut[lut] * torch.stack([sflp0, sflp1]).T
#     return lut


# class trellis_quantizer(nn.Module):
#     def __init__(self,
#                  L=16,
#                  K=2,
#                  V=2,
#                  decode_mode='1mad',
#                  tlut_bits=10,
#                  tlut=None,
#                  viterby_bs=1024):
#         super(trellis_quantizer, self).__init__()
#         self.idx_dtype = torch.int32
#         self.opt_scale = 1

#         self.L = L
#         self.K = K
#         self.V = V
#         self.decode_mode = decode_mode
#         self.viterby_bs = viterby_bs

#         if decode_mode == '1mad':
#             assert V == 1
#             self.register_buffer('lut',
#                                  decode_1mad(torch.arange(2**L)).unsqueeze(-1))
        
#         elif decode_mode == 'quantlut_sym':
#             if tlut is None:
#                 tlut_bits = 9
#                 assert tlut_bits > 0
#                 if V == 2:
#                     fname = f'/tmp/kmeans_{tlut_bits}_{V}.pt'
#                     if not os.path.exists(fname):
#                         tlut = torch.randn(2**tlut_bits, V)
#                         import scipy
#                         data = torch.randn(1 << 20, 2)
#                         clusters = scipy.cluster.vq.kmeans(data, tlut)
#                         tlut = torch.tensor(clusters[0])
#                         tlut = (tlut /
#                                 tlut.std(unbiased=False)) * 0.9682458365518543
#                         torch.save(tlut, fname)
#                     else:
#                         tlut = torch.load(fname)
#                 else:
#                     raise Exception
#                 self.register_buffer('tlut', tlut)
#                 self.register_buffer(
#                     'lut',
#                     quantlut_sym(self.tlut, L, tlut_bits).T.contiguous())

#         elif decode_mode == 'LowBitSym':
#             assert tlut_bits > 0
#             tlut = get_positive_lowbit_codebook(2**tlut_bits, values_bits=4, bound=3.0)[0]
#             self.register_buffer('tlut', tlut)
#             self.register_buffer(
#                 'lut',
#                 quantlut_sym_2d(self.tlut, L, tlut_bits).contiguous())

#         else:
#             raise Exception

#         self.fakeinf = torch.tensor(torch.inf)

#         self.register_buffer('sumdelta',
#                              torch.arange(2**(K * V)) << (L - K * V))
#         self.sumdelta = self.sumdelta.view(1, 1, -1)
        
#         self.register_buffer('state', torch.arange(2**L).unsqueeze(0))
        
#         self.register_buffer('state_candidates',
#                              (self.state >>
#                               (K * V))[0, ::2**(K * V)].unsqueeze(-1) +
#                              self.sumdelta) # who can go to this state
#         self.register_buffer('recons_state', self.recons(self.state))


#     def recons(self, encoded, **kwargs):
#         return self.lut[encoded.int().to(self.lut.device)].to(encoded.device)

 
#     #@torch.compile
#     def update(self, cost, orig_seq_part):
#         """
#         Viterbi update step: Computes new path costs and backtrace pointers
#         Args:
#             cost: Accumulated cost from previous timestep 
#                    Shape: (B, 2^(L-K*V)) = (batch, reduced_states)
#             orig_seq_part: Current observation vector 
#                            Shape: (B, V) = (batch, values_per_step)
#         Returns:
#             prev_state: Best previous full state for backtrace 
#                         Shape: (B, 2^(L-K*V)) = (batch, reduced_states)
#             new_cost: Updated path costs
#                       Shape: (B, 2^(L-K*V)) = (batch, reduced_states)
#         """
#         B = cost.shape[0]  # Batch size
#         S_red = self.state_candidates.shape[1]  # 2^(L-K*V) = reduced states
#         D = self.state_candidates.shape[2]      # 2^(K*V)   = transitions per state
        
#         # 1. Compute reconstruction error for current timestep
#         # recons_state: (1, 2^L, V)
#         # orig_seq_part: (B, V) -> (B, 1, V)
#         # state_err: (B, 2^L)
#         state_err = (self.recons_state - orig_seq_part.unsqueeze(1)).square().sum(dim=-1)

#         # 2. Get reduced state indices for candidates (shift to fit cost tensor)
#         # state_candidates: (1, S_red, D) = full state indices
#         # index_reduced: (1, S_red, D) = reduced state indices
#         index_reduced = self.state_candidates >> (self.K * self.V)
        
#         # 3. Gather previous costs for candidate states
#         # cost_expanded: (B, S_red, S_red) = cost tensor prepared for gathering
#         cost_expanded = cost.unsqueeze(1).expand(-1, S_red, -1)
#         # cost_of_candidates: (B, S_red, D) = cost for each candidate transition
#         cost_of_candidates = torch.gather(
#             cost_expanded, 
#             dim=2, 
#             index=index_reduced.expand(B, -1, -1).long()
#         )

#         # 4. Add reconstruction error to path costs
#         # Gather error for each candidate state: (B, S_red, D)
#         state_err_candidates = torch.gather(
#             state_err, 
#             dim=1, 
#             index=self.state_candidates.expand(B, -1, -1).view(B, -1).long()
#         ).view(B, S_red, D)
        
#         # Total cost for each candidate: (B, S_red, D)
#         total_cost_candidates = cost_of_candidates + state_err_candidates
        
#         # 5. Find best transition (min cost per reduced state)
#         # best_values: (B, S_red), best_indices: (B, S_red)
#         best_values, best_indices = torch.min(total_cost_candidates, dim=-1)
        
#         # 6. Get previous full states for backtrace
#         # prev_state: (B, S_red)
#         prev_state = torch.gather(
#             self.state_candidates.expand(B, -1, -1),
#             dim=2,
#             index=best_indices.unsqueeze(-1).long()
#         ).squeeze(-1)

#         return prev_state, best_values

#     def viterbi(self, X, overlap=None):
#         """Viterbi decoding for optimal sequence quantization
#         Args:
#             X: Input sequence, shape (B, T) = (batch, timesteps)
#         Returns:
#             final_state: Quantized state sequence, shape (B, T//V)
#         """
#         B, T = X.shape
#         assert T % self.V == 0, "Sequence length must be multiple of V"
        
#         # 1. Initialize cost matrix
#         # First observation: X[:, :V] shape (B, V)
#         # recons_state: (1, 2^L, V)
#         # cost: (B, 2^L) = initial reconstruction error
#         cost = (self.recons_state - X[:, :self.V].unsqueeze(1)).square().sum(dim=-1)
        
#         # 2. Forward pass: Update costs and store back pointers
#         # from_state: stores best previous FULL state for each reduced state
#         # Shape: (B, 2^(L-K*V), T//V)
#         from_state = torch.zeros(B, 2**(self.L - self.K * self.V), T // self.V,
#                                  dtype=self.state.dtype,
#                                  device=self.state.device)

#         # Process each timestep
#         for i in range(1, T // self.V):
#             # Get current observation segment
#             obs = X[:, i * self.V:(i + 1) * self.V]  # (B, V)
            
#             # Update cost matrix and get back pointers
#             prev_state, cost = self.update(cost, obs)  # both (B, 2^(L-K*V))
            
#             # Store back pointers (FULL state indices)
#             from_state[:, :, i] = prev_state

#         # 3. Backtrace: Find optimal path
#         # final_state: will store FULL state indices at each timestep
#         final_state = torch.zeros(B, T // self.V,
#                                   dtype=self.idx_dtype,
#                                   device=X.device)
        
#         # Start with lowest cost state at end
#         final_state[:, -1] = torch.argmin(cost, dim=1)
        
#         # Backwards traversal
#         for i in range(T // self.V - 1, 0, -1):
#             # Get reduced state for current full state
#             # (high bits = previous state pointer)
#             reduced_state = (final_state[:, i] >> (self.K * self.V))
            
#             # Gather best previous FULL state from from_state
#             final_state[:, i - 1] = torch.gather(
#                 from_state[:, :, i],  # (B, S_red)
#                 dim=1,
#                 index=reduced_state.unsqueeze(1).long()
#             ).squeeze(1)

#         return final_state

#     def quantize_seq(self, X, overlap=None, **kwargs):
#         n_seq, T = X.shape
#         batch_padding_len = math.ceil(n_seq / self.viterby_bs) * self.viterby_bs - n_seq
#         X = torch.nn.functional.pad(X.T, (0, batch_padding_len)).T

#         n_seq_padded = X.shape[0]
#         X = X.reshape(n_seq_padded // self.viterby_bs, self.viterby_bs, T).contiguous()
#         if overlap is not None:
#             overlap = torch.nn.functional.pad(overlap.T, (0, batch_padding_len)).T
#             overlap = overlap.reshape(n_seq_padded // self.viterby_bs, self.viterby_bs)

#         Qidxs = torch.zeros(n_seq_padded // self.viterby_bs,
#                             self.viterby_bs,
#                             T // self.V,
#                             dtype=self.idx_dtype,
#                             device=X.device)
#         for i in tqdm(range(len(X))):
#             overlap_batch = None if overlap is None else overlap[i]
#             Qidxs[i] = self.viterbi(X[i], overlap=overlap_batch)
#         Qidxs = Qidxs.reshape(n_seq_padded, T // self.V)[:n_seq]
#         return Qidxs

#     def quantize(self, X, batch_size='auto', **kwargs):
#         X = X.contiguous().to(torch.float16)
#         T = X.shape[-1]
#         roll_X = torch.roll(X, T // (2 * self.V) * self.V, 1)
#         state = self.quantize_seq(roll_X, overlap=None, batch_size=batch_size)
#         #overlap = state[T // (2 * self.V)] >> self.K * self.V
#         #state = self.quantize_seq(X, overlap=overlap, batch_size=batch_size)
#         print(state.shape)
#         hatX = self.recons(state).transpose(0, 1).reshape(X.shape)
#         return hatX.contiguous().to(X.device), state.contiguous().to(X.device)

import os
import torch
import torch.nn as nn
from tqdm import tqdm
import math


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
    """Quantized lookup table with sign flipping"""
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
        sflp1 = 1 - ((lut >> 7) & 1) * 2
        lut = (lut >> (16 - nbits - 1)) & ((1 << nbits) - 1)
    lut = tlut[lut] * torch.stack([sflp0, sflp1]).T
    return lut


class trellis_quantizer(nn.Module):
    def __init__(self,
                 L=16,
                 K=2,
                 V=2,
                 decode_mode='1mad',
                 tlut_bits=10,
                 tlut=None,
                 viterby_bs=1024):
        super(trellis_quantizer, self).__init__()
        self.idx_dtype = torch.int32
        self.opt_scale = 1

        self.L = L
        self.K = K
        self.V = V
        self.decode_mode = decode_mode
        self.viterby_bs = viterby_bs

        if decode_mode == '1mad':
            assert V == 1
            self.register_buffer('lut',
                               decode_1mad(torch.arange(2**L)).unsqueeze(-1))
        
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
                        tlut = (tlut / tlut.std(unbiased=False)) * 0.9682458365518543
                        torch.save(tlut, fname)
                    else:
                        tlut = torch.load(fname)
                else:
                    raise Exception
                self.register_buffer('tlut', tlut)
                self.register_buffer(
                    'lut',
                    quantlut_sym(self.tlut, L, tlut_bits).T.contiguous())

        elif decode_mode == 'LowBitSym':
            assert self.V == 2
            assert tlut_bits > 0
            tlut = get_positive_lowbit_codebook(2**tlut_bits, values_bits=4, bound=3.0)[0]
            self.register_buffer('tlut', tlut)
            self.register_buffer(
                'lut',
                quantlut_sym_2d(self.tlut, L, tlut_bits).contiguous())

        else:
            raise Exception

        self.fakeinf = torch.tensor(torch.inf)

        # State transition buffers
        self.register_buffer('sumdelta',
                           torch.arange(2**(K * V)) << (L - K * V))
        self.sumdelta = self.sumdelta.view(1, 1, -1)
        
        self.register_buffer('state', torch.arange(2**L).unsqueeze(0))  # (1, 2^L)
        
        # State candidates: maps (reduced_state, delta) -> full_state
        # Shape: (1, 2^(L-K*V), 2^(K*V))
        self.register_buffer('state_candidates',
                           (self.state >> (K * V))[0, ::2**(K * V)].unsqueeze(-1) + self.sumdelta)
        
        # Reconstruction values for all states
        self.register_buffer('recons_state', self.recons(self.state))  # (1, 2^L, V)

        # Add buffer for state reduction
        self.register_buffer('reduced_state_size', torch.tensor(2**(L - K * V)))

    def recons(self, encoded, **kwargs):
        """Reconstruct values from encoded states"""
        return self.lut[encoded.int().to(self.lut.device)].to(encoded.device)

    @torch.compile
    def update(self, cost, orig_seq_part):
        """
        Viterbi update step
        Args:
            cost: (B, 2^(L-K*V)) - reduced state costs from previous step
            orig_seq_part: (B, V) - current observation vector
        Returns:
            prev_state: (B, 2^(L-K*V)) - best previous full states
            new_cost: (B, 2^(L-K*V)) - new reduced state costs
        """
        B = cost.shape[0]  # Batch size
        S_red = self.reduced_state_size.item()  # 2^(L-K*V)
        D = 2**(self.K * self.V)  # 2^(K*V)

        # 1. Compute reconstruction error for current timestep
        # state_err: (B, 2^L)
        state_err = (self.recons_state - orig_seq_part.unsqueeze(1)).square().sum(dim=-1)

        # 2. Map state errors to candidate states
        # Reshape state_candidates to match state_err dimensions
        # state_candidates: (1, S_red, D) -> (B, S_red*D)
        candidates_flat = self.state_candidates.expand(B, -1, -1).reshape(B, -1)
        
        # state_err_candidates: (B, S_red*D)
        state_err_candidates = torch.gather(
            state_err, 
            dim=1,
            index=candidates_flat.long()
        ).view(B, S_red, D)  # Reshape back to (B, S_red, D)

        # 3. Gather previous costs for candidate transitions
        # cost_expanded: (B, S_red, S_red)
        cost_expanded = cost.unsqueeze(1).expand(-1, S_red, -1)
        
        # Get reduced state indices for candidates (high bits)
        # index_reduced: (1, S_red, D) -> (B, S_red, D)
        index_reduced = (self.state_candidates >> (self.K * self.V)).expand(B, -1, -1)
        
        # cost_of_candidates: (B, S_red, D)
        cost_of_candidates = torch.gather(
            cost_expanded, 
            dim=2, 
            index=index_reduced.long()
        )

        # 4. Total cost = previous path cost + reconstruction error
        total_cost_candidates = cost_of_candidates + state_err_candidates

        # 5. Find best transition (min cost per candidate group)
        best_values, best_indices = torch.min(total_cost_candidates, dim=-1)

        # 6. Get best previous full states for backtrace
        prev_state = torch.gather(
            self.state_candidates.expand(B, -1, -1),
            dim=2,
            index=best_indices.unsqueeze(-1).long()
        ).squeeze(-1)

        return prev_state, best_values

    def viterbi(self, X, overlap=None):
        """Viterbi decoding for optimal sequence quantization"""
        B, T = X.shape
        T_v = T // self.V  # Number of V-step segments
        S_red = self.reduced_state_size.item()  # 2^(L-K*V)
        D = 2**(self.K * self.V)  # 2^(K*V)

        # 1. Initialize cost matrix for first segment
        cost_full = (self.recons_state - X[:, :self.V].unsqueeze(1)).square().sum(dim=-1)
        
        # 2. Reduce first state: group into S_red groups and take min
        cost_reduced, low_bits = torch.min(
            cost_full.view(B, S_red, D), 
            dim=2
        )
        
        # Store initial full states: (B, S_red)
        full_states = (torch.arange(S_red, device=X.device).view(1, -1) * D + low_bits)
        
        # 3. Backtrace storage: (B, S_red, T_v)
        from_state = torch.zeros(B, S_red, T_v, 
                               dtype=torch.long, 
                               device=X.device)
        from_state[:, :, 0] = full_states
        
        # 4. Forward pass for subsequent timesteps
        for i in range(1, T_v):
            obs = X[:, i*self.V:(i+1)*self.V]  # (B, V)
            prev_state, cost_reduced = self.update(cost_reduced, obs)
            from_state[:, :, i] = prev_state

        # 5. Backtrace: find optimal path
        final_state = torch.zeros(B, T_v, dtype=torch.long, device=X.device)
        final_state[:, -1] = from_state[
            torch.arange(B), 
            torch.argmin(cost_reduced, dim=1),
            -1
        ]
        
        # Trace backwards
        for i in range(T_v-2, -1, -1):
            reduced = final_state[:, i+1] >> (self.K * self.V)
            final_state[:, i] = from_state[
                torch.arange(B),
                reduced,
                i
            ]
            
        return final_state

    def quantize_seq(self, X, overlap=None, **kwargs):
        """Quantize sequence with batch processing"""
        n_seq, T = X.shape
        batch_padding_len = math.ceil(n_seq / self.viterby_bs) * self.viterby_bs - n_seq
        X = torch.nn.functional.pad(X.T, (0, batch_padding_len)).T

        n_seq_padded = X.shape[0]
        X = X.reshape(n_seq_padded // self.viterby_bs, self.viterby_bs, T).contiguous()
        if overlap is not None:
            overlap = torch.nn.functional.pad(overlap.T, (0, batch_padding_len)).T
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
        """Main quantization method"""
        X = X.contiguous().to(torch.float16)
        T = X.shape[-1]
        #roll_X = torch.roll(X, T // (2 * self.V) * self.V, 1)
        #state = self.quantize_seq(roll_X, overlap=None, batch_size=batch_size)
        #overlap = state[T // (2 * self.V)] >> self.K * self.V


        state = self.quantize_seq(X, overlap=None, batch_size=batch_size)
        print(state.shape)
        
        hatX = self.recons(state).reshape(X.shape)
        return hatX.contiguous().to(X.device), state.contiguous().to(X.device)
    


import torch
import math

# @torch.compile
# def decode_compressed(L, K, V, m, n, compressed, expanded_lut):
#     V_flag = int(math.log2(V))
#     if compressed.dtype != torch.uint16:
#         compressed = compressed.view(torch.uint16)

#     assert compressed.shape == (K * m * n // 16, )

#     BLOCK_SIZE = 16 * 16
#     BITS_PER_BLOCK = K * 16 * 16  # K bits * f16 mma tile A size

#     # unswizzle interleaved blocks

#     compressed = (compressed.view(torch.uint8) \
#                   .reshape(m // 16 // 2, n // 16 // 2, BLOCK_SIZE // 8, 2, 2, K) \
#                   .permute(0, -2, 1, -3, 2, -1) \
#                   .flip((-1, )) \
#                   .reshape(m // 16, n // 16, BITS_PER_BLOCK // 16, 2)
#                   .flip((-1, )) \
#                   .view(torch.uint16) \
#                   .reshape(m // 16, n // 16, BITS_PER_BLOCK // 16))
#     # decode block

#     assert L <= 16

#     blocked = compressed.reshape(K * m * n // BITS_PER_BLOCK,
#                                  BITS_PER_BLOCK // 16, 1)
#     blocked_roll = torch.roll(blocked.to(torch.int32), -1,
#                               -2).to(blocked.dtype)
#     blocked32 = torch.cat((blocked_roll, blocked),
#                           dim=-1).reshape(blocked.shape[0],
#                                           -1).contiguous().view(torch.uint32)
#     # blocked32 is 16bits[-1]||16bits[0] 16bits[0]||16bits[1] ... 16bits[-2]||16bits[-1]

#     expanded32 = blocked32.reshape(*blocked32.shape,
#                                    1).expand(*blocked32.shape,
#                                              16).view(torch.int32)
#     shifts = (torch.arange(0, 16, dtype=torch.int32,
#                            device=blocked.device)).to(torch.int32).reshape(
#                                1, 1, -1).expand(expanded32.shape)
#     shifted = expanded32 >> (16 - shifts)
#     indices = torch.bitwise_and(
#         shifted.reshape(shifted.shape[0], -1)[:, 16 - L::K << V_flag], (1 << L) - 1)

#     # decode lut
#     mma_swizzled = expanded_lut[indices]
    
#     decompressed = (
#         mma_swizzled.reshape(m // 16, n // 16, 8, 4, 2, 2, 2) \
#             .permute(0, -2, 2, 1, -3, 3,-1) \
#             .reshape(m, n))
#     return decompressed

import torch
import math

@torch.compile
def decode_compressed(L, K, V, m, n, compressed, expanded_lut):
    V_flag = int(math.log2(V))
    if compressed.dtype != torch.uint16:
        compressed = compressed.view(torch.uint16)

    BITS_PER_BLOCK = K * 16 * 16  # K bits * f16 mma tile A size
    
    assert compressed.shape[0] == K * m * n // 16
    
    # Skip unswizzling - use natural block order
    compressed = compressed.reshape(m//16, n//16, K*16)
    
    # Continue with decoding
    blocked = compressed.reshape(K * m * n // BITS_PER_BLOCK,
                                 BITS_PER_BLOCK // 16, 1)
    blocked_roll = torch.roll(blocked.to(torch.int32), -1, -2).to(blocked.dtype)
    blocked32 = torch.cat((blocked_roll, blocked),
                          dim=-1).reshape(blocked.shape[0],
                                          -1).contiguous().view(torch.uint32)
    expanded32 = blocked32.reshape(*blocked32.shape, 1).expand(*blocked32.shape,
                                                            16).view(torch.int32)
    shifts = (torch.arange(0, 16, dtype=torch.int32,
                           device=blocked.device)).to(torch.int32).reshape(
                               1, 1, -1).expand(expanded32.shape)
    shifted = expanded32 >> (16 - shifts)
    indices = torch.bitwise_and(
        shifted.reshape(shifted.shape[0], -1)[:, 16 - L::K << V_flag], (1 << L) - 1)

    # Decode LUT
    mma_swizzled = expanded_lut[indices]
    
    # Deswizzle to 16x16 blocks
    decompressed = mma_swizzled.reshape(m // 16, n // 16, 16, 16)
    return decompressed.reshape(m, n)


# class DecodeKernelAG(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, L, K, V, m, n, compressed, lut):
#         ctx.save_for_backward(compressed, lut)
#         ctx.L = L
#         ctx.K = K
#         ctx.V = V
#         ctx.m = m
#         ctx.n = n

#         hatW = decode_compressed(L, K, V, m, n, compressed, lut)
#         return hatW

#     @staticmethod
#     def backward(ctx, grad_output):
#         # Not sure what the gradient of hatW should be used for; returning None for all
#         return (None,) * 7

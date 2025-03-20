import torch
import torch.nn as nn

def get_composite_kernel(base_kernel: torch.Tensor, levels: int) -> torch.Tensor:
    composite_kernel = base_kernel
    for _ in range(1, levels):
        composite_kernel = composite_kernel.view(-1, 1, 2 ** _, 2 ** _)
        composite_kernel = torch.einsum('aibc,djef->aidjbcef', composite_kernel, base_kernel)
        composite_kernel = composite_kernel.reshape(-1, 1, 2 ** (_ + 1), 2 ** (_ + 1))
    return composite_kernel


HAAR_KERNEL = torch.tensor([
    [[[1, 1], 
      [1, 1]]],
    [[[1, -1], 
      [1, -1]]],
    [[[1, 1], 
      [-1, -1]]],
    [[[1, -1], 
      [-1, 1]]]
], dtype=torch.float)



class HaarWavelet(nn.Module):
    def __init__(self, level=2, device='cpu', dtype=torch.float32):
        super().__init__()
        self.level = level
        
        self.forward_conv = torch.nn.Conv2d(1, 4**level, 2**level, 2**level, bias=False)
        #self.forward_conv.weight.requires_grad = False
        self.forward_conv.weight.data = get_composite_kernel(HAAR_KERNEL, level).to(device).to(dtype)

        self.inverse_conv = torch.nn.ConvTranspose2d(4**level, 1, 2**level, 2**level, bias=False)
        #self.inverse_conv.weight.requires_grad = False
        self.inverse_conv.weight.data = get_composite_kernel(HAAR_KERNEL, level).to(device).to(dtype)

        self.scale = 1 / 4**level

    def forward(self, x):
        return self.forward_conv(x.unsqueeze(0)) * self.scale
    

    def inverse(self, coeffs):
        return self.inverse_conv(coeffs).squeeze(0)
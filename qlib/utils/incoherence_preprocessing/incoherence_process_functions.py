# adapted from https://github.com/Cornell-RelaxML/quip-sharp

import torch

from qlib.utils.incoherence_preprocessing.matmul_had import (
	matmul_hadU_cuda, matmul_hadU, matmul_hadUt_cuda, matmul_hadUt
)


def incoherence_preprocess(W):
    dtype_ = torch.float32
    device = W.device
    (m, n) = W.shape

    use_func = matmul_hadUt_cuda if W.is_cuda else matmul_hadUt

    Wr = W

    # randomized hadamard transformation on W
    SU = (torch.randn(n, device=device).sign() + 1e-5).sign().to(dtype_)
    SV = (torch.randn(m, device=device).sign() + 1e-5).sign().to(dtype_)
    Wr = use_func(use_func(W.T * SV).T * SU)

    SV = SV.cpu()
    SU = SU.cpu()

    Wr = Wr.to(device)

    return Wr, SU, SV

def incoherence_process(hatWr, SU, SV):
    device = hatWr.device

    use_func = matmul_hadU_cuda if hatWr.is_cuda else matmul_hadU

    # reverse hadamard transformation
    hatWr = (use_func(
        (use_func(hatWr) * SU.to(device)).T) * SV.to(device)).T

    assert torch.isfinite(hatWr).all()
    return hatWr


def incoherence_preprocess_lukashevich(W):
    dtype_ = torch.float32
    device = W.device
    (m, n) = W.shape

    use_func = matmul_hadUt_cuda if W.is_cuda else matmul_hadUt

    # randomized hadamard transformation on W
    SU = (torch.randn(n, device=device).sign() + 1e-5).sign().to(dtype_)
    Wr = use_func(W * SU)

    SU = SU.cpu()
    Wr = Wr.to(device)

    return Wr, SU


def incoherence_process_lukashevich(hatWr, SU):
    device = hatWr.device

    use_func = matmul_hadU_cuda if hatWr.is_cuda else matmul_hadU

    # reverse hadamard transformation
    hatWr = use_func(hatWr) * SU.to(device)

    assert torch.isfinite(hatWr).all()
    return hatWr
import torch


def batch_gathering(codebooks, indices, matrix_shape=None):
    assert len(codebooks.shape) == 3
    assert len(indices.shape) == 2
    assert indices.shape[0] == codebooks.shape[0]
    
    N = codebooks.shape[0]

    batch_indices = (
        torch.arange(N)
        .view(N, 1)
    ).to(indices.device)

    result = codebooks[batch_indices, indices, :]
    if matrix_shape is not None:
        result = result.reshape(N, *matrix_shape)
    return result

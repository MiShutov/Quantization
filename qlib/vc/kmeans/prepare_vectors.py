import torch

@torch.no_grad()
def prepare_vectors_for_kmeans_grad(
        matrix,
        grad,
        device,
        vec_dim=8,
        scale_type='L2', # 'NONE', "L2", "STD"
        weights_type="PER_VECTOR", #'PER_COORD'
        matrix_block_size=-1
    ):

    matrix = matrix.to(device)
    grad = grad.to(device)

    # scale matrix
    if scale_type == 'L2':
        scales = torch.linalg.norm(matrix, axis=1)
    elif scale_type == 'STD':
        scales = matrix.std(axis=1)
    elif scale_type == 'NONE':
        scales = torch.ones_like(matrix[:, 0])
    else:
        raise RuntimeError(f'prepare_vectors::error: uknown scale_type <{scale_type}>')
    scales = scales.to(device)
    matrix = matrix/scales.unsqueeze(1)

    # scale grads
    grad = grad * (scales.unsqueeze(-1)**2)
    grad/=grad.mean()

    # prepere matrix blocks if needed
    if matrix_block_size==-1:
        vectors = matrix.reshape(-1, vec_dim)
        vector_grads = grad.reshape(-1, vec_dim)
    else:
        print(matrix.shape)
        vectors = matrix.reshape(-1, matrix_block_size, vec_dim)
        vector_grads = grad.reshape(-1, matrix_block_size, vec_dim)

    # apply weight type
    if weights_type=='PER_VECTOR':
        weights = torch.linalg.norm(vector_grads, axis=-1).unsqueeze(-1)
    elif weights_type=='PER_COORD':
        weights = vector_grads
    else:
        raise RuntimeError(f'prepare_vectors::error: uknown weights_type <{weights_type}>')

    return {
        'vectors' : vectors,
        'vector_weights' : weights,
        'output_scales': scales
    }

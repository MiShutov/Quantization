import os
import torch
from qlib.vector_quantization.kmeans.auto_kmeans import AutoKmeans
from tqdm import trange

from dataclasses import dataclass, asdict
from typing import Literal

SAVING_TEMPLATE = 'cb{cb_size}_vecdim{vecdim}_weight{weighting}_scale{scale}_dist{dist}_blocksize{matrix_block_size}_iters{num_iters}'

@dataclass
class KmeasJobParams:
    layer_name: str
    path_to_save: str
    vector_dim: int
    codebook_size: int
    num_iters: int
    matrix: torch.Tensor
    hess: torch.Tensor = None
    scale_type: Literal["OUTL2", "OUTSTD"] = None
    weighting_type: Literal["PERCOORD", "PERVECTOR"] = None
    distance_type: Literal["MSE", "COS", "WEIGHTED_MSE"] = 'MSE'
    init_type: Literal["TOPK", "RANDOM"] = 'RANDOM'
    batch_size: int = 2**14
    matrix_block_size: int = None
    eps: float = 1e-8


def save_result(indices, codebook, scales, params):
    metadata = asdict(params)
    del metadata['matrix']
    del metadata['hess']
    del metadata['path_to_save']
    del metadata['batch_size']

    save_chekpoint = {
        'indices': indices,
        'codebook': codebook,
        'scales': scales,
        'metadata': metadata
    }
    
    path2save = os.path.join(
        params.path_to_save,
        SAVING_TEMPLATE.format(
            cb_size=params.codebook_size,
            vecdim=params.vector_dim,
            weighting=params.weighting_type,
            scale=params.scale_type,
            dist=params.distance_type,
            matrix_block_size=params.matrix_block_size,
            num_iters=params.num_iters
        )
    )

    os.makedirs(path2save, exist_ok=True)
    torch.save(save_chekpoint, f'{path2save}/{params.layer_name}.pth')


def run_kmeans(cluster_computer, prepared_vectors_data):
    vectors = prepared_vectors_data['vectors']
    weights = prepared_vectors_data['weights']
    
    if vectors.ndim == 2:
        indices, centroids = cluster_computer.run(
            vectors=vectors,
            weights=weights,
        )
        return indices.cpu(), centroids.cpu()

    if vectors.ndim == 3:
        n_blocks = vectors.shape[0]

        indices_list = []
        centroids_list = []

        for i_block in trange(n_blocks):
            indxs, centroids = cluster_computer.run(
                vectors=vectors[i_block],
                weights=weights[i_block],
            )
            indices_list.append(indxs.cpu())
            centroids_list.append(centroids.cpu())

        return torch.stack(indices_list).cpu(), torch.stack(centroids_list).cpu()


def reconstruct_matrix(indices, codebook, scales, matrix_shape):
    if codebook.ndim == 2:
        matrix = codebook[indices].reshape(matrix_shape)
    
    elif codebook.ndim == 3:
        n_blocks = indices.shape[0]
        all_vectors = []
        for i_block in range(n_blocks):
            indices_part = indices[i_block]
            codebook_part = codebook[i_block]
            all_vectors.append(codebook_part[indices_part])
        matrix = torch.stack(all_vectors).reshape(matrix_shape)

    if scales is not None:
        matrix *= scales
    
    return matrix

@torch.no_grad()
def kmeas_job(
    params: KmeasJobParams, 
    device,
    ):
    #torch.cuda.set_device(device)

    cluster_computer = AutoKmeans(
            vector_dim=params.vector_dim,
            scale_type=params.scale_type,
            weighting_type=params.weighting_type,
            distance_type=params.distance_type,
            num_centroids=params.codebook_size,
            matrix_block_size=params.matrix_block_size,
            init_type=params.init_type,
            num_iters=params.num_iters,
            batch_size=params.batch_size,
            eps=params.eps,
            device=device,
        )

    prepared_vectors_data = cluster_computer.prepare_vectors(
        matrix=params.matrix,
        hess=params.hess
    )

    indices, codebook = run_kmeans(cluster_computer, prepared_vectors_data)

    save_result(indices, codebook, prepared_vectors_data['scales'], params)
    
    # compute mse
    matrix_quantized = reconstruct_matrix(indices, codebook, prepared_vectors_data['scales'], params.matrix.shape)
    error = ((matrix_quantized-params.matrix)**2).mean().item()

    return {
        'layer_name': params.layer_name,
        'mse' : error,
    }

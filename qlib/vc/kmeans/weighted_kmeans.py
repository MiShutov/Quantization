import torch
from qlib.utils import free_unused_memory

@torch.no_grad()
def weighted_kmeans_optimized(
        num_clusters: int, 
        vectors: torch.tensor,
        weights: torch.tensor, 
        num_iterations: int = 100, 
        batch_size: int = 2**16, 
        device='cpu',
        weights_type='PER_VECTOR', #'PER_COORD' #
    ):

    vectors = vectors.to(device)
    weights = weights.to(device)
    
    # set n_batches
    n_vectors = vectors.shape[0]
    if batch_size==-1:
        n_batches=1
        batch_size=n_vectors
    else:
        n_batches = n_vectors//batch_size
    assert n_batches*batch_size == n_vectors
    
    # # Randomly initialize cluster centers #TODO: implement k-means++ algo
    # import random
    # centroids = vectors[random.sample(range(n_vectors), num_clusters)].clone()

    # smart initialization
    centroids = vectors[torch.topk(torch.linalg.norm(weights, axis=1), num_clusters).indices].clone()

    for _ in range(num_iterations):
        all_cluster_assignments = []
        for batch_idx in range(n_batches):
            batch = vectors[batch_idx*batch_size:(batch_idx+1)*batch_size]
            distances = torch.cdist(batch, centroids)  # Shape: (N, num_clusters)
            batch_assignments = torch.argmin(distances, dim=1)
            all_cluster_assignments.append(batch_assignments)
        cluster_assignments = torch.cat(all_cluster_assignments, axis=-1).to(torch.int32)
        
        # Update cluster centers
        for i in range(num_clusters):
            if (cluster_assignments == i).any():
                cluster_vectors = vectors[cluster_assignments == i]
                cluster_vectors_weights = weights[cluster_assignments == i]
                if weights_type=='PER_VECTOR':
                    centroids[i] = (cluster_vectors * cluster_vectors_weights).mean(dim=0)/(cluster_vectors_weights.mean())
                elif weights_type=='PER_COORD':
                    centroids[i] = (cluster_vectors * cluster_vectors_weights).mean(dim=0)/(cluster_vectors_weights.mean(dim=0))

    free_unused_memory()
    return cluster_assignments, centroids

import torch
from qlib.utils import free_unused_memory


class AutoCentroid:
    def __init__(
            self,
            vec_dim=8,
            scale_type='L2',
            weights_type='PER_VECTOR', #'PER_COORD' #
            distance_type='MSE', # 'WEIGHTED_MSE' #
            num_clusters=None,
            num_iterations=None,
            batch_size=2**16,
        ):
        self.vec_dim = vec_dim
        self.scale_type = scale_type
        self.weights_type = weights_type
        self.distance_type = distance_type
        
        # optional
        self.batch_size = batch_size
        self.num_clusters = num_clusters
        self.num_iterations = num_iterations


    @torch.no_grad()
    def prepare_vectors(
            self,
            matrix,
            grad,
            device,
            matrix_block_size=-1
        ):

        matrix = matrix.to(device)
        grad = grad.to(device)

        # scale matrix
        if self.scale_type == 'L2':
            scales = torch.linalg.norm(matrix, axis=1)
        elif self.scale_type == 'STD':
            scales = matrix.std(axis=1)
        elif self.scale_type == 'NONE':
            scales = torch.ones_like(matrix[:, 0])
        else:
            raise RuntimeError(f'prepare_vectors::error: uknown scale_type <{self.scale_type}>')
        scales = scales.to(device)
        matrix = matrix/scales.unsqueeze(1)

        # scale grads
        grad = grad * (scales.unsqueeze(-1)**2)
        grad/=grad.mean()

        # prepere matrix blocks if needed
        if matrix_block_size==-1:
            vectors = matrix.reshape(1, -1, self.vec_dim)
            vector_grads = grad.reshape(1, -1, self.vec_dim)
        else:
            if matrix.numel() % (matrix_block_size * self.vec_dim) != 0:
                vectors = matrix.reshape(1, -1, self.vec_dim)
                vector_grads = grad.reshape(1, -1, self.vec_dim)
            else:
                vectors = matrix.reshape(-1, matrix_block_size, self.vec_dim)
                vector_grads = grad.reshape(-1, matrix_block_size, self.vec_dim)
        

        # apply weight type
        if self.weights_type=='PER_VECTOR':
            weights = torch.linalg.norm(vector_grads, axis=-1).unsqueeze(-1)
        elif self.weights_type=='PER_COORD':
            weights = vector_grads
        else:
            raise RuntimeError(f'prepare_vectors::error: uknown weights_type <{self.weights_type}>')

        return {
            'vectors' : vectors,
            'vector_weights' : weights,
            'output_scales': scales
        }

    @torch.no_grad()
    def _set_n_batches(self, vectors, batch_size=2**16):
        n_vectors = vectors.shape[0]
        if batch_size==-1:
            n_batches=1
            batch_size=n_vectors
        else:
            n_batches = n_vectors//batch_size
        
        assert n_batches*batch_size == n_vectors
        return n_batches

    @torch.no_grad()
    def _init_centroids(self, vectors, weights):
        # # Randomly initialize cluster centers #TODO: implement k-means++ algo
        # import random
        # centroids = vectors[random.sample(range(vectors.shape[0]), num_clusters)].clone()

        # smart initialization
        top_k_weights = torch.topk(torch.linalg.norm(weights, axis=1), self.num_clusters)
        centroids = vectors[top_k_weights.indices].clone()
        centroid_weigths = top_k_weights.values
        return centroids, centroid_weigths

    @torch.no_grad()
    def _compute_assigments(
            self,
            vectors, 
            centroids,
            batch_size, 
            n_batches, 
            centroid_weigths=None
        ):
        all_cluster_assignments = []
        for batch_idx in range(n_batches):
            batch = vectors[batch_idx*batch_size:(batch_idx+1)*batch_size]
            if self.distance_type=='MSE':
                distances = torch.cdist(batch, centroids)  # Shape: (N, num_clusters)
            elif self.distance_type=='WEIGHTED_MSE':
                distances = torch.cdist(batch, centroids)
                distances = distances * centroid_weigths
            elif self.distance_type=='COS':
                normed_vectors = batch / (batch.norm(dim=1, keepdim=True) + 1e-8)
                normed_centroids = centroids / (centroids.norm(dim=1, keepdim=True) + 1e-8)
                distances = 1 - torch.mm(normed_vectors, normed_centroids.T)
            else:
                raise RuntimeError(f'prepare_vectors::error: uknown distance_type <{self.distance_type}>')

            batch_assignments = torch.argmin(distances, dim=1)
            all_cluster_assignments.append(batch_assignments)
        cluster_assignments = torch.cat(all_cluster_assignments, axis=-1).to(torch.int32)
        return cluster_assignments
    
    @torch.no_grad()
    def _update_centroids( 
            self,
            cluster_assignments, 
            vectors, 
            weights, 
            centroids,
            centroid_weigths, 
        ):
        for i in range(self.num_clusters):
            if (cluster_assignments == i).any():
                cluster_vectors = vectors[cluster_assignments == i]
                cluster_vectors_weights = weights[cluster_assignments == i]
                #centroid_weigths[i] = torch.log(1+cluster_vectors_weights.sum())
                centroid_weigths[i] = torch.max(cluster_vectors_weights)
                if self.weights_type=='PER_VECTOR':
                    centroids[i] = (cluster_vectors * cluster_vectors_weights).mean(dim=0)/(cluster_vectors_weights.mean())
                elif self.weights_type=='PER_COORD':
                    centroids[i] = (cluster_vectors * cluster_vectors_weights).mean(dim=0)/(cluster_vectors_weights.mean(dim=0))

    @torch.no_grad()
    def run(
            self,
            vectors: torch.tensor,
            weights: torch.tensor,
            device='cpu',
            batch_size=None,
            num_iterations=None,
        ):
        batch_size=batch_size if batch_size is not None else self.batch_size
        num_iterations=num_iterations if num_iterations is not None else self.num_iterations

        assert batch_size is not None
        assert num_iterations is not None

        # set device
        vectors = vectors.to(device)
        weights = weights.to(device)
        
        # set n_batches
        n_batches = self._set_n_batches(vectors=vectors, batch_size=batch_size)

        # set initial centroids
        centroids, centroid_weigths = self._init_centroids(vectors=vectors, weights=weights)

        # main loop
        for _ in range(num_iterations):
            # compute cluster assignments
            cluster_assignments = self._compute_assigments(
                vectors=vectors, 
                centroids=centroids,
                batch_size=batch_size,
                n_batches=n_batches,
                centroid_weigths=centroid_weigths
            )
            # update cluster centers
            self._update_centroids(
                cluster_assignments=cluster_assignments, 
                vectors=vectors, 
                weights=weights, 
                centroids=centroids,
                centroid_weigths=centroid_weigths
            )
        free_unused_memory()
        return cluster_assignments, centroids

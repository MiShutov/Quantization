import torch
import warnings

class AutoKmeans:
    def __init__(
            self,
            device='cpu',
            vector_dim=8,
            scale_type='OUTL2',
            weighting_type='PERVECTOR', #'PERCOORD' #
            distance_type='MSE', # 'WEIGHTED_MSE' #
            init_type='RANDOM',
            num_centroids=None,
            num_iters=None,
            batch_size=2**16,
            eps=1e-8,
            matrix_block_size=None,
        ):
        self.device = device
        self.vector_dim = vector_dim
        self.scale_type = scale_type
        self.weighting_type = weighting_type
        self.distance_type = distance_type
        self.init_type = init_type
        # optional
        self.matrix_block_size = matrix_block_size
        self.batch_size = batch_size
        self.num_centroids = num_centroids
        self.num_iters = num_iters
        self.eps = eps


    @torch.no_grad()
    def prepare_vectors(
            self,
            matrix,
            hess,
        ):

        matrix = matrix.to(self.device)
        hess = hess.to(self.device)

        # scale matrix
        if self.scale_type == 'OUTL2':
            scales = torch.linalg.norm(matrix, axis=1).unsqueeze(-1)
        elif self.scale_type == 'STD':
            scales = matrix.std(axis=1).unsqueeze(-1)
        elif self.scale_type is None:
            scales = None
        else:
            raise RuntimeError(f'prepare_vectors::error: uknown scale_type <{self.scale_type}>')
        
        if scales is not None:
            scales = scales.to(self.device)
            matrix = matrix / scales
            scales = scales.to('cpu')

        # TODO: scale hess (if FISHER HESS)
        # hess = hess * (scales**2)

        # for numerical stability
        hess /= hess.mean()

        # prepere matrix blocks if needed
        if self.matrix_block_size is None:
            vectors = matrix.reshape(-1, self.vector_dim)
            weights = hess.reshape(-1, self.vector_dim)
        else:
            impossible_to_divide_flag = matrix.numel() % (self.matrix_block_size * self.vector_dim) != 0
            too_small_flag = matrix.numel() // (self.matrix_block_size * self.vector_dim) < 2
            
            if impossible_to_divide_flag or too_small_flag:
                warnings.warn(
                    f"AutoKmeans::prepare_vectors: matrix_block_size={self.matrix_block_size} is incompatible with matrix shape={matrix.shape} and vector_dim={self.vector_dim}. Running kmeans without dividing matrix on blocks.",
                    category=UserWarning,
                    stacklevel=2,
                )
                vectors = matrix.reshape(-1, self.vector_dim)
                weights = hess.reshape(-1, self.vector_dim)
            else:
                vectors = matrix.reshape(-1, self.matrix_block_size, self.vector_dim)
                weights = hess.reshape(-1, self.matrix_block_size, self.vector_dim)
        

        # apply weighting type
        if self.weighting_type=='PERVECTOR':
            weights = torch.linalg.norm(weights, axis=-1).unsqueeze(-1)
        elif self.weighting_type=='PERCOORD':
            weights = weights
        elif self.weighting_type is None:
            weights = None
        else:
            raise RuntimeError(f'prepare_vectors::error: uknown weighting_type <{self.weighting_type}>')

        return {
            'vectors' : vectors,
            'weights' : weights,
            'scales': scales
        }

    @torch.no_grad()
    def _set_n_batches(self, vectors, batch_size):
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
        if self.init_type == 'RANDOM':
            import random
            init_indices = random.sample(range(vectors.shape[0]), self.num_centroids)
            init_centroids = vectors[init_indices].clone()
        elif self.init_type == 'TOPK':
            init_indices = torch.topk(torch.linalg.norm(weights, axis=1), self.num_centroids).indices
            init_centroids = vectors[init_indices].clone()
        else:
            raise RuntimeError(f'_init_centroids::error: uknown init_type <{self.init_type}>')


        if self.distance_type == 'WEIGHTED_MSE':
            centroid_weigths = torch.linalg.norm(weights[init_indices], axis=1)
            return {
                "centroids": init_centroids, 
                "centroid_weigths": centroid_weigths
            }
        return init_centroids


    @torch.compile
    @torch.no_grad()
    def _compute_assigments(
            self,
            vectors, 
            centroids,
            batch_size, 
            n_batches
        ):
        all_cluster_assignments = []
        for batch_idx in range(n_batches):
            batch = vectors[batch_idx*batch_size:(batch_idx+1)*batch_size]
            if self.distance_type=='MSE':
                distances = torch.cdist(batch, centroids)
            elif self.distance_type=='WEIGHTED_MSE':
                distances = torch.cdist(batch, centroids["centroids"])
                distances = distances * centroids['centroid_weigths']
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
    
    @torch.compile
    @torch.no_grad()
    def _update_centroids( 
            self,
            cluster_assignments, 
            vectors, 
            weights, 
            centroids
        ):
        for i in range(self.num_centroids):
            if (cluster_assignments == i).any():
                cluster_vectors = vectors[cluster_assignments == i]
                cluster_vectors_weights = weights[cluster_assignments == i]
                
                if self.distance_type == 'WEIGHTED_MSE':
                    centroids['centroid_weigths'][i] = torch.max(cluster_vectors_weights)
                    # centroids['centroid_weigths'][i] = torch.log(1+cluster_vectors_weights.sum())

                if self.weighting_type=='PERVECTOR':
                    centroids[i] = (cluster_vectors * cluster_vectors_weights).mean(dim=0)/(cluster_vectors_weights.mean() + self.eps)
                elif self.weighting_type=='PERCOORD':
                    centroids[i] = (cluster_vectors * cluster_vectors_weights).mean(dim=0)/(cluster_vectors_weights.mean(dim=0) + self.eps)
                else:
                    raise RuntimeError(f'prepare_vectors::error: uknown weighting_type <{self.weighting_type}>')

    @torch.no_grad()
    def run(
            self,
            vectors: torch.tensor,
            weights: torch.tensor,
            batch_size=None,
            num_iters=None,
        ):
        batch_size=batch_size if batch_size is not None else self.batch_size
        num_iters=num_iters if num_iters is not None else self.num_iters

        assert batch_size is not None
        assert num_iters is not None

        # set device
        vectors = vectors.to(self.device)
        weights = weights.to(self.device)
        
        # set n_batches
        n_batches = self._set_n_batches(vectors=vectors, batch_size=batch_size)

        # set initial centroids
        centroids = self._init_centroids(vectors=vectors, weights=weights)
    
        # main loop
        for _ in range(num_iters):
            # compute cluster assignments
            cluster_assignments = self._compute_assigments(
                vectors=vectors, 
                centroids=centroids,
                batch_size=batch_size,
                n_batches=n_batches
            )
            # update cluster centers
            self._update_centroids(
                cluster_assignments=cluster_assignments, 
                vectors=vectors, 
                weights=weights, 
                centroids=centroids
            )
        # free_unused_memory()
        return cluster_assignments, centroids

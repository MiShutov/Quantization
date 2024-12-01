import math
import torch
import torch.nn as nn

from cuml.cluster import KMeans
from cuml.common.device_selection import using_device_type

from qlib.quantizers.quantizer import Quantizer


class VectorQuantizer(Quantizer):
    def __init__(self, codebook_size, group_size, scaler=None):
        super().__init__(
            group_size=group_size,
            bit_width=math.ceil(math.log2(codebook_size))
        )
        self.codebook_size = codebook_size
        self.codebook = nn.Embedding(num_embeddings=self.codebook_size,
                                     embedding_dim=self.group_size)
        self.idxs = nn.Parameter(torch.tensor(1), requires_grad=False)
        self.scaler = scaler


    @torch.no_grad()
    def _initialize(self, x) -> None:
        vectors = self.regroup(x)
        
        cuml_kmeans = KMeans(
            n_clusters=self.codebook_size,
            max_iter=100,#300,
            tol=1e-4,
            init='scalable-k-means++',
            oversampling_factor=1,
            n_init=1,
            max_samples_per_batch=32768
        )

        # TODO: add switcher
        with using_device_type('gpu'):
            idxs = torch.tensor(cuml_kmeans.fit_predict(vectors.cpu().numpy()))
        cluster_centers = torch.tensor(cuml_kmeans.cluster_centers_)
        
        self.codebook.weight.data = cluster_centers.to(x.device)
        self.idxs.data = idxs.to(x.device)
        self._initialized = True


    def quantize(self, x):
        x_shape = x.shape
        if self.scaler is not None:
            x = self.scaler.scale(x)

        if not self._initialized:
            self._initialize(x.detach())

        #if self.additions is not None:
        #    x = x + self.additions

        vectors = self.codebook(self.idxs)
        x_q = vectors.reshape(x_shape)

        if self.scaler is not None:
            x_q = self.scaler.unscale(x_q)

        return x_q
    
    
    def forward(self, x):
        if not self._quantize:
            return x
        return self.quantize(x)

import math
import torch
import torch.nn as nn

import faiss
#from cuml.cluster import KMeans
#from cuml.common.device_selection import using_device_type

from qlib.quantizers.quantizer import Quantizer

NITER = 25

class VectorQuantizer(Quantizer):
    def __init__(self,
                 codebook_size, 
                 group_size, 
                 scaler=None,
                 with_additions=False,
                 with_reassings=False):
        super().__init__(
            group_size=group_size,
            bit_width=math.ceil(math.log2(codebook_size))
        )
        self.codebook_size = codebook_size
        self.scaler = scaler
        self.with_additions = with_additions
        self.with_reassings = with_reassings
        self.faiss_index = None

    @torch.no_grad()
    def configure(self, module):
        module_weight_shape = self.regroup(module.weight).shape
        self.codebook = nn.Embedding(num_embeddings=self.codebook_size,
                                     embedding_dim=self.group_size)
        index_dtype = torch.int32 if self.codebook_size > 2**15 else torch.int16
        self.idxs = nn.Parameter(torch.empty(module_weight_shape[0], 1, dtype=index_dtype), requires_grad=False)
        if self.with_additions:
            self.additions = nn.Parameter(
                torch.zeros(module_weight_shape, dtype=torch.float32), requires_grad=True
            )
        if self.scaler is not None:
            self.scaler.configure(module.weight)

    
    @torch.no_grad()
    def _set_faiss_index(self, centroids) -> None:
        if self.codebook_size > 2**13: #> 2**12: 
            nlist = int(self.codebook_size**0.5)
            index_IVF = faiss.IndexIVFFlat(faiss.IndexFlatL2(self.group_size), self.group_size, nlist)
            index_IVF.train(centroids)
            index_IVF.nprobe = int(nlist**0.5)
            self.faiss_index = index_IVF
        else:
            self.faiss_index = faiss.IndexFlatL2(self.group_size)

    @torch.no_grad()
    def _initialize(self, x) -> None:
        vectors = self.regroup(x.detach()).cpu()
        kmeans = faiss.Kmeans(
            d=self.group_size,
            k=self.codebook_size, 
            niter=NITER, 
            verbose=False, 
            update_index=False,
            gpu=True,
            min_points_per_centroid=1,
            max_points_per_centroid=vectors.shape[0]//self.codebook_size
        )
        kmeans.train(vectors)
        _, idxs = kmeans.index.search(vectors, 1)
        
        self.codebook.weight.data = torch.tensor(kmeans.centroids).to(x.device)
        self.idxs.data = torch.tensor(idxs).to(x.device)
        
        self._initialized.data = torch.tensor(True).to(x.device)


    @torch.no_grad()
    def reassign(self, x):
        if self.with_additions:
            x = x + self.additions.reshape(x.shape)

        vectors = self.regroup(x)

        if self.faiss_index is None:
            self._set_faiss_index(self.codebook.weight.detach().cpu())

        self.faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.faiss_index)
        self.faiss_index.reset()
        self.faiss_index.add(self.codebook.weight.detach().cpu())
        self.faiss_index = faiss.index_gpu_to_cpu(self.faiss_index)

        _, idxs = self.faiss_index.search(vectors.cpu(), 1)
        self.idxs.copy_(torch.tensor(idxs, dtype=self.idxs.dtype, device=self.idxs.device))


    def quantize(self, x):
        x_shape = x.shape

        if self.scaler is not None:
            x = self.scaler.scale(x)

        if not self._initialized:
            self._initialize(x)

        if self.with_reassings:
            self.reassign(x)

        vectors = self.codebook(self.idxs.to(torch.int32))
        x_q = vectors.reshape(x_shape)

        if self.scaler is not None:
            x_q = self.scaler.unscale(x_q)

        return x_q - x.detach() + x # provides additions grad
    
    
    def forward(self, x):
        if not self._quantize:
            return x
        return self.quantize(x)


    @torch.no_grad()
    def __repr__(self):
        return f"{self.__class__.__name__}(vector_dim={self.group_size}, codebook_size={self.codebook_size}, scaler={self.scaler})"

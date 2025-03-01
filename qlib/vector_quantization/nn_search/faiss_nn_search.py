import torch
import faiss

faiss_settings = {
    'vector_dim': 8,
    'nlist' : 256,
    'nprobe' : 8
}


@torch.no_grad()
def reassign(vectors, codebook, reassine_params):
    quantizer = faiss.IndexFlatL2(faiss_settings['vector_dim'])
    faiss_index = faiss.IndexIVFFlat(
        quantizer, 
        faiss_settings['vector_dim'], 
        faiss_settings['nlist'], 
    )
    faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss_index)
    faiss_index.nprobe = faiss_settings['nprobe']
    faiss_index.train(codebook)
    faiss_index.add(codebook)
    indices = faiss_index.search(vectors, 1)[1][:, 0]
    return indices


# @torch.no_grad()
# def reassign(vectors, codebook, reassine_params=None):
#     faiss_index = faiss.IndexFlatL2(faiss_settings['vector_dim'])
#     faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss_index)
#     faiss_index.add(codebook)
#     indices = faiss_index.search(vectors, 1)[1][:, 0]
#     return indices
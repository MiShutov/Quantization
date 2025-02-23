import torch
import faiss

faiss_settings = {
    'vector_dim': 8,
    'nlist' : 256,
    'nprobe' : 8
}

@torch.no_grad()
def reassign(vectors, codebook):
    quantizer = faiss.IndexFlatL2(faiss_settings['vector_dim'])
    index_ivf = faiss.IndexIVFFlat(
        quantizer, 
        faiss_settings['vector_dim'], 
        faiss_settings['nlist'], 
        faiss.METRIC_L2
    )

    gpu_index_ivf = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index_ivf)
    gpu_index_ivf.nprobe = faiss_settings['nprobe']

    gpu_index_ivf.train(codebook)
    gpu_index_ivf.add(codebook)

    indices = gpu_index_ivf.search(vectors, 1)[1][:, 0]

    return indices
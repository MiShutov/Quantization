import torch
import faiss


@torch.no_grad()
def reassign_approx(vectors, codebook, reassine_params):
    vector_dim = vectors.shape[-1]
    quantizer = faiss.IndexFlatL2(vector_dim)
    faiss_index = faiss.IndexIVFFlat(
        quantizer, 
        vector_dim, 
        reassine_params['nlist'], 
    )
    faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss_index)
    faiss_index.nprobe = reassine_params['nprobe']
    faiss_index.train(codebook)
    faiss_index.add(codebook)
    indices = faiss_index.search(vectors, 1)[1][:, 0]
    return indices


@torch.no_grad()
def reassign_exact(vectors, codebook, reassine_params=None):
    vector_dim = vectors.shape[-1]
    faiss_index = faiss.IndexFlatL2(vector_dim)
    faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss_index)
    faiss_index.add(codebook)
    indices = faiss_index.search(vectors, 1)[1][:, 0]
    return indices


def reassign_torch(vectors, codebook, batch_size):
    n_batches = vectors.shape[0] // batch_size
    assert n_batches * batch_size == vectors.shape[0]
    all_cluster_assignments = []
    for batch_idx in range(n_batches):
        batch = vectors[batch_idx*batch_size:(batch_idx+1)*batch_size]
        distances = torch.cdist(batch, codebook)
       
        batch_assignments = torch.argmin(distances, dim=1)
        all_cluster_assignments.append(batch_assignments)
    cluster_assignments = torch.cat(all_cluster_assignments, axis=-1).to(torch.int32)
    return cluster_assignments


@torch.no_grad()
def reassign(vectors, codebook, reassine_params={}):
    if reassine_params['type']=='torch':
        return reassign_torch(vectors, codebook, reassine_params['batch_size'])
    else:
        raise

    # if reassine_params.get('nlist', False) and reassine_params.get('nprobe', False):
    #     return reassign_approx(vectors, codebook, reassine_params)
    # else:
    #     return reassign_exact(vectors, codebook, reassine_params)

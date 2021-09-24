from pykeops.torch import LazyTensor
import torch
  
def nearest_neighbour(query, source, radius=None):
    # input numpy array
    # query (n, 3)
    # source (m, 3)
    # return inds (n,)
    x_i = LazyTensor( query[:,None,:].float() )  # x_i.shape = (1e6, 1, 3)
    y_j = LazyTensor( source[None,:,:].float() )  # y_j.shape = ( 1, 2e6,3)
    D_ij = ((x_i - y_j) ** 2).sum(-1)  # (M**2, N) symbolic matrix of squared distances
    indKNN = D_ij.argKmin(1, dim=1)  # Grid <-> Samples, (M**2, K) integer tensor
    inds = indKNN[:,0]
    assert inds.shape[0] == query.shape[0]
    mask = inds > -1e9
    if radius != None:
        distance = torch.norm(source[inds].float()-query,2,1)
        mask = distance < radius
    return inds[mask], mask




import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def stick_filter(x, filter, h, w):
    # split x in into blocks with shape (h, w)
    x = view_as_blocks(x, (h, w))
    blockshape = x.shape

    # reshape so that each block in x is a row in a 2d array
    x = x.reshape(-1, x.shape[-2] * x.shape[-1])

    # convret x to sparse matrix so that the filter remains sparse
    x = sparse.csr_matrix(x)

    # get the count of non zero values in each filter
    non_zero = sparse.csr_matrix(np.diff(filter.indptr).astype(np.float64))

    x = (x.dot(filter.T).multiply(non_zero.power(-1))).power(2).dot(filter)
    x /= filter.shape[0]

    # convert back to dense matrix
    x = x.A

    # put blocks back together
    x  = x.reshape(blockshape)
    x = np.block([list(t) for t in x])

    # relu
    x[x < 0] = 0

    # per channel normalization
    x -= np.mean(x, 1, keepdims=True)
    x /= np.std(x, 1, keepdims=True)

    return x

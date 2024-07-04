import torch
def distance(X, Y, square=True):
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0)
    x = x * x
    x = torch.t(x.repeat(m, 1))
    y = torch.norm(Y, dim=0)
    y = y * y
    y = y.repeat(n, 1)
    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    return result

def get_Laplacian_from_weights(weights):
    W = torch.eye(weights.shape[0]) + weights
    degree = torch.sum(W, dim=1).pow(-0.5)
    return (W * degree).t()*degree

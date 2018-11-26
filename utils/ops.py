import torch


def truncated_normal(tensor, mean=0., std=1.):
    shape = tensor.shape

    sample = torch.randn(shape) * std + mean
    is_truncated = (sample > (2. * std + mean)) | (sample < (-2. * std + mean))

    while torch.any(is_truncated):
        repick = torch.randn(sample[is_truncated].shape) * std + mean
        sample[is_truncated] = repick
        is_truncated = (sample > (2. * std + mean)) | (sample < (-2. * std + mean))

    with torch.no_grad():
        tensor.copy_(sample)


def mean(tensor, dim=None, keepdim=False):
    if dim is None:
        return torch.mean(tensor, keepdim=keepdim)

    shape = tensor.shape
    dims = torch.Tensor([shape[d] for d in dim])
    return torch.sum(tensor, dim=dim, keepdim=keepdim) / torch.prod(dims)


def create_covariance_matrix(diags):
    shape = diags.shape

    covariance_matrix = torch.zeros(shape[0], shape[1], shape[1])
    for idx, diag in enumerate(diags):
        covariance_matrix[idx, range(shape[1]), range(shape[1])] = diag
    return covariance_matrix
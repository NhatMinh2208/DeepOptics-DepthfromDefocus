import torch
import numpy as np
def poly_coeff(t):
    t1 = t
    t2 = t ** 2
    t3 = t ** 3
    return (2 * t3 - 3 * t2 + 1), (t3 - 2 * t2 + t1), (-2 * t3 + 3 * t2), (t3 - t2)


def interp(x, y, xs, ind):
    """
    ind: left index of xs in x
    Assumes uniform sampling. It could be arbitrary sampling, but not implemented.
    """
    n, h, w = ind.shape
    dx = x[:, [1]] - x[:, [0]]
    diff = torch.stack([xs[i] - x[i, ind[i]] for i in torch.arange(n)], dim=0) # wl X (n_rho - 2) x (n_rho - 2)

    c0, c1, c2, c3 = poly_coeff(diff.unsqueeze(1) / dx.reshape(-1, 1, 1, 1)) 

    # Add depth dimension
    x = x.unsqueeze(1)
    m = (y[..., 1:] - y[..., :-1]) / (x[..., 1:] - x[..., :-1]) / dx.reshape(-1, 1, 1)
    m = torch.cat([m[..., [0]], (m[..., 1:] + m[..., :-1]) / 2, m[..., [-1]]], dim=-1)

    y0 = torch.stack([y[i, :, ind[i]] for i in torch.arange(n)], dim=0)
    y1 = torch.stack([y[i, :, ind[i] + 1] for i in torch.arange(n)], dim=0)
    m0 = torch.stack([m[i, :, ind[i]] for i in torch.arange(n)], dim=0)
    m1 = torch.stack([m[i, :, ind[i] + 1] for i in torch.arange(n)], dim=0)

    ys = c0 * y0 + c1 * m0 + c2 * y1 + c3 * m1

    return ys

# def linterp(x, y, sample, index):
#     '''
#     x: inputs\n
#     y: outputs\n
#     sample: samples \n
#     x must have size larger than 2 to run successfully\n
#     -->return: output of samples\n
#     '''
#     n, _, _ = index.shape # usually n_wl
#     index[:,0,0] = 0
#     diff = torch.stack([(sample[i] - x[i, index[i]]) / (x[i, index[i] + 1] - x[i, index[i]]) for i in torch.arange(n)], dim=0) 
#     temp = y[:,:, 0]
#     result = torch.stack([ y[i, :, index[i]] * (1 - diff[i]) + y[i, :, index[i] + 1] * diff[i]  for i in torch.arange(n)], dim=0) 
#     result[:,:,0,0] = temp
#     return result

def linterp(x, y, sample, index):
    '''
    x: inputs\n
    y: outputs\n
    sample: samples \n
    x must have size larger than 2 to run successfully\n
    -->return: output of samples\n
    '''
    n, _, _ = y.shape # usually n_wl
    index[0,0] = 0
    diff = (sample - x[index]) / (x[index + 1] - x[index]) # (10, 10)
    temp = y[:,:, 0]
    result = torch.stack([ y[i, :, index] * (1 - diff) + y[i, :, index + 1] * diff  for i in torch.arange(n)], dim=0) 
    result[:,:,0,0] = temp
    return result

def linterp2(x, y, sample, index, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    '''
    x: inputs\n
    y: outputs\n
    sample: samples \n
    x must have size larger than 2 to run successfully\n
    -->return: output of samples\n
    '''
    n, _, _ = y.shape # usually n_wl

    # print("this")
    # print(x.shape) #torch.Size([193])
    # print(y.shape) #torch.Size([3, 16, 192])

    #x_last_element = torch.tensor([x[-1]]).to(device)
    #y_last_element = y[:,:, -1].unsqueeze(-1).to(device)
    y_last_element = torch.zeros_like(y.squeeze(-1)).to(device)

    # print(y_last_element.shape) #torch.Size([3, 16, 192])

    #x = torch.cat([x, x_last_element], dim=-1)
    y = torch.cat([y, y_last_element], dim=-1)
    #x = torch.cat([x, x_last_element], dim=-1)
    y = torch.cat([y, y_last_element], dim=-1)

    # print("that")
    # print(x.shape) #torch.Size([193])
    # print(y.shape) #torch.Size([3, 16, 576])

    diff = (sample - x[index - 1]) / (x[index] - x[index - 1]) # (10, 10)
    #temp = y[:,:, 0]
    # print("diff:")
    # print(diff)
    result = torch.stack([ y[i, :, index - 1] * (1 - diff) + y[i, :, index] * diff  for i in torch.arange(n)], dim=0) 
    #result[:,:,0,0] = temp
    return result

def cubinterp(x, y,xs, ind, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    ind: left index of xs in x
    Assumes uniform sampling. It could be arbitrary sampling, but not implemented.
    x : masksize / 2 + 1
    y : n_wl x n_depth x masksize / 2
    xs, ind : masksize/2 x masksize/2
    """
    print(x.shape)
    print(y.shape)
    print(xs.shape)
    print(ind.shape)
    n, h, w = y.shape
    dx = x[1] - x[0]
    diff = xs - x[ind] # masksize/2 x masksize/2
    print(diff.shape)
    c0, c1, c2, c3 = poly_coeff(diff / dx)  # each coff is masksize/2 x masksize/2

    # Add depth dimension
    #print(dx)
    x = x.unsqueeze(0).unsqueeze(1)
    #print((x[..., 1:] - x[..., :-1]))
    y_last_element = y[:,:, -1].unsqueeze(-1).to(device)
    y = torch.cat([y, y_last_element], dim=-1)
    y = torch.cat([y, y_last_element], dim=-1)
    m = (y[..., 1:] - y[..., :-1]) / dx / dx
    m = torch.cat([m[..., [0]], (m[..., 1:] + m[..., :-1]) / 2, m[..., [-1]]], dim=-1)
    print(m.shape)
    y0 = torch.stack([y[i, :, ind] for i in torch.arange(n)], dim=0)
    y1 = torch.stack([y[i, :, ind + 1] for i in torch.arange(n)], dim=0)
    m0 = torch.stack([m[i, :, ind] for i in torch.arange(n)], dim=0)
    m1 = torch.stack([m[i, :, ind + 1] for i in torch.arange(n)], dim=0)

    ys = c0 * y0 + c1 * m0 + c2 * y1 + c3 * m1

    return ys






import torch

# Create a 1D real tensor
# x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

# # Apply rfft along the last dimension (default)
# result_default = torch.fft.rfft(x)

# # Apply rfft along the first dimension
# result_dim_0 = torch.fft.rfft(x, dim=0)

# print("Default rfft result (last dimension):")
# print(result_default)

# print("rfft result along the first dimension:")
# print(result_dim_0)
# print(result_dim_0.shape)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.rand([3, 16, 1024, 1024]).to(device)
y = torch.rand([3, 16, 63, 63]).to(device)
new_size = x.shape[-1] + y.shape[-1] - 1
x_padding = (new_size - x.shape[-1]) 
y_padding = (new_size - y.shape[-1]) 

# Pad the original tensor to the new size
padded_x = torch.nn.functional.pad(x, (0, x_padding, 0, x_padding), mode='constant', value=0)
padded_y = torch.nn.functional.pad(y, (0, y_padding, 0, y_padding), mode='constant', value=0)
# Check the new size

# print("Padded Tensor Size:", padded_x)
# print("Padded Tensor Size:", padded_y.size())
x_c = torch.complex(torch.cos(padded_x), torch.sin(padded_x))
y_c = torch.complex(torch.cos(padded_y), torch.sin(padded_y))
sz = padded_x.shape
x_ = torch.fft.fft2(x_c)
y_ = torch.fft.fft2(y_c)
mul_ = x_ * y_
conv = torch.fft.ifft2(mul_, sz[-2:])
print(conv)
print(conv.shape)
real = conv.real
img = conv.imag
print(real.shape)
# print(real)

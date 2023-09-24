import torch

# Create a 1D real tensor
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

# Apply rfft along the last dimension (default)
result_default = torch.fft.rfft(x)

# Apply rfft along the first dimension
result_dim_0 = torch.fft.rfft(x, dim=0)

print("Default rfft result (last dimension):")
print(result_default)

print("rfft result along the first dimension:")
print(result_dim_0)
print(result_dim_0.shape)
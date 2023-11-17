import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch.nn.functional as F

# Define the dimensions of the 2D tensor
rows, cols = 64, 64  # Adjust the size as needed

# Create a grid of coordinates centered at (0, 0)
x = torch.linspace(-1, 1, cols)
y = torch.linspace(-1, 1, rows)
x, y = torch.meshgrid(x, y)

# Calculate the radial gradient from the center
radius = torch.sqrt(x**2 + y**2)

# Normalize the radius to the range [0, 1]
radius = (radius - radius.min()) / (radius.max() - radius.min())

# Invert the gradient so that values are smaller at the center and larger at the edges
inverted_radius = 1 - radius

# Clip the values to ensure they are within the range [0, 1]
inverted_radius = inverted_radius.clamp(0, 1)
cmap = LinearSegmentedColormap.from_list('custom', [(0, 'black'), (1, 'white')])


tensor = torch.rand(3,16,64,64)
tensor = inverted_radius
print(tensor.shape)
print(tensor[0,0].shape)
plt.figure()
# normalized term
#tensor = tensor.cpu()
#plt.imshow(tensor[0,0], cmap=cmap)
plt.colorbar()
plt.show()
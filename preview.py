import torch
import matplotlib.pyplot as plt

dataset= torch.load("dataset.pt")

plt.imshow(dataset[-1].squeeze(), cmap='gray')

plt.show()
print(dataset[-1].squeeze().max())

#dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min())  # Normalize to [0,1]
#dataset = dataset * 2 - 1  # Scale to [-1,1] for better tanh-like behavior
# Normalize each image individually to [-1, 1]
dataset = (dataset - dataset.amin(dim=(2, 3), keepdim=True)) / \
          (dataset.amax(dim=(2, 3), keepdim=True) - dataset.amin(dim=(2, 3), keepdim=True) + 1e-8)
dataset = dataset * 2 - 1

print(dataset.shape)
plt.imshow(dataset[0].squeeze(), cmap='gray')
print(dataset[0].squeeze().max())

plt.savefig("preview.pdf", format="pdf")

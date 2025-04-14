from flow_matching.utils import ModelWrapper
from flow_matching.solver import Solver, ODESolver
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math

import torch
import torch.nn as nn

im_size = 64

class FourierEmbedding(nn.Module):
    def __init__(self, dim, scale=30.0):
        super().__init__()
        self.dim = dim
        self.scale = scale
        half_dim = dim // 2
        self.B = nn.Parameter(torch.randn(half_dim) *
                              scale, requires_grad=False)

    def forward(self, t):
        t = t.view(-1, 1)  # (batch_size, 1)
        proj = 2 * math.pi * t * self.B.unsqueeze(0)  # (batch_size, dim//2)
        embedding = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class TimeResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, time_embed_dim):
        super().__init__()
        # Time conditioning
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * out_dim)
        )

        # Main block for 64x64 images
        self.block = nn.Sequential(
            nn.GroupNorm(8, in_dim),
            nn.SiLU(),
            nn.Conv2d(in_dim, out_dim, 3, padding=1),
            nn.GroupNorm(8, out_dim),
            nn.SiLU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1)
        )
        self.skip = nn.Conv2d(
            in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, t_embed):
        scale, shift = self.time_mlp(t_embed).chunk(2, dim=-1)
        h = self.block(x)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return h + self.skip(x)


class VelocityField(nn.Module):
    def __init__(self, time_embed_dim=256, fourier_scale=30.0):
        super().__init__()
        # Time embedding
        self.time_embed = nn.Sequential(
            FourierEmbedding(time_embed_dim, fourier_scale),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # Encoder for 64x64 -> 32x32 -> 16x16
        self.down = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),    # 64x64x1 -> 64x64x64
            TimeResBlock(64, 64, time_embed_dim),
            nn.AvgPool2d(2),                   # 64x64x64 -> 32x32x64
            TimeResBlock(64, 128, time_embed_dim),
            nn.AvgPool2d(2)                    # 32x32x128 -> 16x16x128
        )
        # Bottleneck at 16x16
        self.mid = TimeResBlock(128, 128, time_embed_dim)

        # Decoder 16x16 -> 32x32 -> 64x64
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),       # 16x16x128 -> 32x32x128
            TimeResBlock(128, 64, time_embed_dim),
            nn.Upsample(scale_factor=2),       # 32x32x64 -> 64x64x64
            TimeResBlock(64, 64, time_embed_dim)
        )

        # Output layer (1 channel for grayscale)
        self.out = nn.Conv2d(64, 1, 3, padding=1)  # 64x64x64 -> 64x64x1

    def forward(self, x, t):
        # x: (B, 1, 64, 64), t: (B, 1)
        t_embed = self.time_embed(t)

        # Encoder
        h = self.down[0](x)
        for layer in self.down[1:]:
            if isinstance(layer, TimeResBlock):
                h = layer(h, t_embed)
            else:
                h = layer(h)

        # Bottleneck
        h = self.mid(h, t_embed)

        # Decoder
        for layer in self.up:
            if isinstance(layer, TimeResBlock):
                h = layer(h, t_embed)
            else:
                h = layer(h)

        return self.out(h)  # (B, 1, 64, 64)


if torch.cuda.is_available():
    device = 'cuda:0'
    print('Using gpu')
else:
    device = 'cpu'
    print('Using cpu.')


# step size for ode solver
step_size = 0.008

# Load the pickle file safely on CPU
vf = torch.load("velocity_field.pt",map_location=torch.device('cpu'))

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t)

wrapped_vf = WrappedModel(vf)


batch_size = 1  # batch size
eps_time = 1e-2
T = torch.linspace(0,1,50)  # sample times
T = T.to(device=device)
x_init = torch.randn(batch_size, 1, im_size, im_size, device=device)
solver = ODESolver(velocity_model=wrapped_vf)  # create an ODESolver class
sol = solver.sample(time_grid=T, x_init=x_init, method='rk4', step_size=step_size, return_intermediates=True)  # sample from the model
sol.shape
sol = sol.cpu()
T = T.cpu()

data = torch.load("dataset.pt")
data_min = data.min()
data_max = data.max()

def unnormalize(samples):
    """Reverse [-1,1] scaling back to [0,1]"""
    return (samples + 1) / 2 

def unscale(samples):
    return (samples) * (data_max - data_min) + data_min



# Calculate the indices of the time steps to plot
num_plots = 10
plot_indices = np.linspace(0, len(T) - 1, num_plots, dtype=int)

# Create the figure and subplots
fig, axs = plt.subplots(1, num_plots, figsize=(20, 20))

# Iterate over the selected time steps and plot
for i, plot_index in enumerate(plot_indices):
    axs[i].imshow(unscale(unnormalize(sol[plot_index].squeeze())), cmap='gray')
    axs[i].set_aspect('equal')
    axs[i].axis('off')
    axs[i].set_title('t= %.2f' % (T[plot_index]))  # Use the correct time value

plt.tight_layout()
plt.savefig('sampled_manual-2.pdf', format='pdf')


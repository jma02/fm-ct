# from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import warnings
import matplotlib.pyplot as plt
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
import numpy as np
from flow_matching.utils import ModelWrapper
from flow_matching.solver import ODESolver
import wandb
from datetime import timedelta

import argparse
import os
import time
import torch

from torch import nn
import torch.nn.functional as F
import math

from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning, module='torch')


# Argument parser for hyperparams
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--iterations', type=int, required=True)
parser.add_argument('--outdir', type=str, required=False)
args = parser.parse_args()


outdir = args.outdir
# Make output directory
if args.outdir:
    os.makedirs(outdir, exist_ok=True)
else:
    outdir = '.'


if torch.cuda.is_available():
    device = 'cuda:0'
    print('Using gpu')
else:
    device = 'cpu'
    print('Using cpu.')

torch.manual_seed(int(time.time_ns() % (2**32)))

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
    def __init__(self, time_embed_dim=256, fourier_scale=100.0):
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


# training arguments
lr = args.lr
batch_size = args.batch_size
iterations = args.iterations
# batch_size = 1
# iterations = 1
print_every = 10000


dataset = torch.load("dataset.pt")
dataset = (dataset - dataset.min()) / \
    (dataset.max() - dataset.min())  # Normalize to [0,1]
dataset = dataset * 2 - 1  # Scale to [-1,1] for better tanh-like behavior

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity=os.getenv("WANDB_TEAM_NAME"),
    # Set the wandb project where this run will be logged.
    project=os.getenv("WANDB_PROJECT"),
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": lr,
        "architecture": "Flow Matching, UNet+Skips",
        "dataset": "Circles",
        "epochs": (iterations * batch_size) / dataset.shape[0],
        "batch size": batch_size
    },
)

print("Dataset loaded from dataset.pt")
vf = VelocityField().to(device)

# instantiate an affine path object
path = AffineProbPath(scheduler=CondOTScheduler())


# init optimizer
optim = torch.optim.AdamW(vf.parameters(), lr=lr, weight_decay=1e-5)  # AdamW with decay

warmup_steps = 5000
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optim,
    max_lr=lr,
    total_steps=iterations,
    pct_start=warmup_steps / iterations
)

# train
print("Training velocity field")
for i in range(iterations):
    optim.zero_grad()

    # sample data (user's responsibility): in this case, (X_0,X_1) ~ pi(X_0,X_1) = N(X_0|0,I)q(X_1)
    # Shape: (batch_size, 1, im_size, im_size)
    x_1 = dataset[torch.randint(0, dataset.shape[0], (batch_size,))]
    x_1 = x_1.to(device)
    x_0 = torch.randn_like(x_1).to(device)

    # sample time (user's responsibility)
    t = torch.rand(x_1.shape[0]).to(device)

    # sample probability path
    path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

    # flow matching l2 loss
    loss = F.mse_loss(
        vf(path_sample.x_t, path_sample.t), path_sample.dx_t)

    # optimizer step
    loss.backward()  # backward

    # Apply gradient clipping
#    torch.nn.utils.clip_grad_norm_(
#        vf.parameters(), max_norm=0.5)  # Clip gradients

    optim.step()  # update
    scheduler.step(loss)
    if (i + 1) % print_every == 0:
        run.log({"loss": loss.item()})


class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t)


wrapped_vf = WrappedModel(vf)


def unscale(samples):
    """Reverse [-1,1] scaling back to [0,1]"""
    return (samples + 1) / 2


# step size for ode solver
step_size = 0.005

batch_size = 1  # batch size, works for batch_size > 1, but don't modify here
eps_time = 1e-2
T = torch.linspace(0, 1, 10)  # sample times
T = T.to(device=device)
x_init = torch.randn(batch_size, 1, im_size, im_size, device=device)
solver = ODESolver(velocity_model=wrapped_vf)  # create an ODESolver class
sol = solver.sample(time_grid=T, x_init=x_init, method='rk4',
                    step_size=step_size, return_intermediates=True)  # sample from the model
sol.shape
sol = sol.cpu()
T = T.cpu()

# Calculate the indices of the time steps to plot
num_plots = 10
plot_indices = np.linspace(0, len(T) - 1, num_plots, dtype=int)

# Create the figure and subplots
fig, axs = plt.subplots(1, num_plots, figsize=(20, 20))


# Iterate over the selected time steps and plot
for i, plot_index in enumerate(plot_indices):
    axs[i].imshow(unscale(sol[plot_index].squeeze()), cmap='gray')
    axs[i].set_aspect('equal')
    axs[i].axis('off')
    axs[i].set_title('t= %.2f' % (T[plot_index]))  # Use the correct time value

plt.tight_layout()
# plt.show()
plt.savefig(f'{outdir}/generated-sample.pdf', format='pdf')

# Save the model

torch.save(vf, f"{outdir}/velocity_field.pt")
print("Model saved as velocity_field.pt")

"""
# Evaluate using SSIM
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

real_images = dataset[:32].to(device)

# step size for ode solver
step_size = 0.05

batch_size = 32
eps_time = 1e-2
T = torch.linspace(0, 1, 10)  # sample times
T = T.to(device=device)
x_init = torch.randn(batch_size, 1, im_size, im_size, device=device)
solver = ODESolver(velocity_model=wrapped_vf)  # create an ODESolver class
sol = solver.sample(time_grid=T, x_init=x_init, method='midpoint',
                    step_size=step_size)  # sample from the model
sol.shape
sol = sol.cpu()
T = T.cpu()

generated_images = unscale(unnormalize(sol)).to(device)

# Ensure they are in the correct value range
if real_images.min() < 0:
    real_images = (real_images + 1) / 2
if generated_images.min() < 0:
    generated_images = (generated_images + 1) / 2

real_images = real_images.clamp(0, 1)
generated_images = generated_images.clamp(0, 1)
ssim_score = ssim(generated_images, real_images)
print(f"SSIM Score: {ssim_score.item():.4f}")
"""

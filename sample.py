from flow_matching.utils import ModelWrapper
from flow_matching.solver import Solver, ODESolver
import numpy as np
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn

im_size = 64
class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)



class UNetVelocityField(nn.Module):
    def __init__(self, in_channels, out_channels, features=[im_size, im_size * 2, im_size * 4, im_size * 8]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Adjust input channels to accommodate time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, im_size//2),
            nn.ReLU(),
            nn.Linear(im_size//2, im_size),
            nn.ReLU()
        )
        in_channels += 1  # Adding 1 for time channel

        # Encoder path
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder path
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feature * 2, feature))

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, t):
        # Process time embedding
        t = t.view(-1, 1)  # Ensure time is in the correct shape
        t_emb = self.time_mlp(t).view(x.shape[0], -1, 1, 1)  # Expand to match spatial dimensions
        t_emb = t_emb.expand(-1, -1, x.shape[2], x.shape[3])

        # Concatenate time embedding with input
        x = torch.cat((x, t_emb), dim=1)

        skip_connections = []

        # Encoder
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]
            x = torch.cat((x, skip_connection), dim=1)
            x = self.decoder[idx + 1](x)

        return self.final_conv(x)


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
T = torch.linspace(0,1,10)  # sample times
T = T.to(device=device)
x_init = torch.randn(batch_size, 1, im_size, im_size, device=device)
solver = ODESolver(velocity_model=wrapped_vf)  # create an ODESolver class
sol = solver.sample(time_grid=T, x_init=x_init, method='midpoint', step_size=step_size, return_intermediates=True)  # sample from the model
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
    axs[i].imshow(sol[plot_index].squeeze(), cmap='gray')
    axs[i].set_aspect('equal')
    axs[i].axis('off')
    axs[i].set_title('t= %.2f' % (T[plot_index]))  # Use the correct time value

plt.tight_layout()
plt.savefig('sampled_manual.pdf', format='pdf')


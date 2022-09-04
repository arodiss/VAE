import torch
from torch.nn import functional as F
from torch import nn
from typing import List


class VAE(nn.Module):
    def __init__(self,
        latent_dim: int,
        interim_dim: int = 32,
        in_channels: int = 1,
        hidden_dims: List = None,
        input_width: int = 64,
    ) -> None:
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        encoder_modules = []
        if hidden_dims is None:
            hidden_dims = [2, 4, 8, 16]

        for h_dim in hidden_dims:
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoded_width = int(input_width / (2 ** len(hidden_dims)))
        self.encoded_depth = hidden_dims[-1]
        encoded_dim = self.encoded_width * self.encoded_width * self.encoded_depth
        encoder_modules.append(nn.Sequential(
            nn.Flatten(),
            nn.Linear(encoded_dim, interim_dim),
            nn.LeakyReLU()
        ))
        self.encoder = nn.Sequential(*encoder_modules)

        self.fc_mu = nn.Linear(interim_dim, latent_dim)
        self.fc_var = nn.Linear(interim_dim, latent_dim)

        decoder_modules = []
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, interim_dim),
            nn.LeakyReLU(),
            nn.Linear(interim_dim, encoded_dim)
        )
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*decoder_modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=1, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def encode(self, object_to_encode):
        result = self.encoder(object_to_encode)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.encoded_depth, self.encoded_width, self.encoded_width)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, encoded_object):
        mu, log_var = self.encode(encoded_object)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), encoded_object, mu, log_var, z]

    def loss_function(self, recons, input, mu, log_var, z, kld_weight) -> dict:
        recons_loss = F.mse_loss(recons, input)
        kl_divergence = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kl_divergence
        return {
            'loss': loss,
            'Reconstruction_Loss': recons_loss.detach(),
            'KL_divergence': -kl_divergence.detach(),
            'z': z.detach(),
        }

    def generate(self, num_samples: int):
        z = torch.randn(num_samples, self.latent_dim)
        samples = self.decode(z)
        return samples

    def reconstruct(self, x):
        return self.forward(x)[0]

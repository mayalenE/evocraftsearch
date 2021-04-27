from evocraftsearch import OutputRepresentation
from evocraftsearch.utils.torch_utils import roll_n
import torch

class ImageRepresentation(OutputRepresentation):

    @staticmethod
    def default_config():
        default_config = OutputRepresentation.default_config()
        default_config.env_size = (16,16,16)
        default_config.channel_list = list(range(1, 10))
        default_config.distance_function = "L2"
        return default_config

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.n_channels = len(self.config.channel_list)
        self.n_latents = self.n_channels * self.config.env_size[0] * self.config.env_size[1] * self.config.env_size[2]


    def calc(self, observations):
        """
            Maps the observations of a system to an embedding vector
            Return a torch tensor
        """
        # filter low values
        last_potential = observations.potentials[-1]
        air_potential = last_potential[0,0,0,0].detach().item()
        filtered_im = torch.where(last_potential > air_potential, last_potential, torch.zeros_like(last_potential))
        filtered_im = filtered_im[:, :, :, self.config.channel_list]

        # recenter
        mu_0 = filtered_im.sum()
        if mu_0.item() > 0:
            # implementation of meshgrid in torch
            z = torch.arange(self.config.env_size[2])
            y = torch.arange(self.config.env_size[1])
            x = torch.arange(self.config.env_size[0])
            zz = z.view(1, 1, -1).repeat(self.config.env_size[0], self.config.env_size[1], 1)
            yy = y.view(1, -1, 1).repeat(self.config.env_size[0], 1, self.config.env_size[2])
            xx = x.view(-1, 1, 1).repeat(1, self.config.env_size[1], self.config.env_size[2])
            Z = (zz - int(self.config.env_size[2] / 2)).double()
            Y = (yy - int(self.config.env_size[1] / 2)).double()
            X = (xx - int(self.config.env_size[0] / 2)).double()

            centroid_z = ((Z.unsqueeze(-1).repeat(1, 1, 1, self.n_channels) * filtered_im).sum() / mu_0).round().int().item()
            centroid_y = ((Y.unsqueeze(-1).repeat(1, 1, 1, self.n_channels) * filtered_im).sum() / mu_0).round().int().item()
            centroid_x = ((X.unsqueeze(-1).repeat(1, 1, 1, self.n_channels) * filtered_im).sum() / mu_0).round().int().item()

            filtered_im = roll_n(filtered_im, 2, centroid_z)
            filtered_im = roll_n(filtered_im, 1, centroid_y)
            filtered_im = roll_n(filtered_im, 0, centroid_x)

        embedding = filtered_im.flatten()

        return embedding


    def calc_distance(self, embedding_a, embedding_b):
        """
            Compute the distance between 2 embeddings in the latent space
            /!\ batch mode embedding_a and embedding_b can be N*M or M
        """
        # l2 loss
        if self.config.distance_function == "L2":
            dist = (embedding_a - embedding_b).pow(2).sum(-1).sqrt()

        else:
            raise NotImplementedError

        return dist
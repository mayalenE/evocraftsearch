from evocraftsearch import OutputRepresentation
from evocraftsearch.utils.torch_utils import roll_n
from addict import Dict
import torch

class HistogramBlocksRepresentation(OutputRepresentation):

    @staticmethod
    def default_config():
        default_config = OutputRepresentation.default_config()
        default_config.env_size = (16,16,16)
        default_config.channel_list = list(range(1, 10))
        default_config.distance_function = "L2"
        return default_config

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        # model
        self.n_channels = len(self.config.channel_list)
        self.n_latents = self.n_channels


    def calc(self, observations):
        """
            Maps the observations of a system to an embedding vector
            Return a torch tensor
        """
        # filter low values
        discrete_last_potential = observations.potentials[-1].argmax(-1)
        embedding = discrete_last_potential.flatten().bincount() / discrete_last_potential.flatten().shape[0]

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


EPS = 0.0001
DISTANCE_WEIGHT = 2  # 1=linear, 2=quadratic, ...

def center_of_mass(input_array):

    normalizer = input_array.sum()
    img_size = input_array.shape
    grids = torch.meshgrid(*[torch.arange(0, i) for i in img_size])

    center = torch.tensor([(input_array* grids[dir].double().to(input_array.device)).sum() / normalizer for dir in range(input_array.ndim)])

    if torch.any(torch.isnan(center)):
        center = torch.tensor([int((input_array.shape[0] - 1) / 2), int((input_array.shape[1] - 1) / 2), int((input_array.shape[2] - 1) / 2)])

    return center


def calc_distance_matrix(img_size):

    dist_mat = torch.zeros(img_size)

    mid = torch.tensor([(img_size[dim] - 1)/ 2 for dim in range(len(img_size))])

    max_dist = int(torch.linalg.norm(mid))

    for z in range(img_size[0]):
        for y in range(img_size[1]):
            for x in range(img_size[2]):
                dist_mat[z][y][x] = (1 - int(torch.linalg.norm(mid - torch.tensor([z, y, x]))) / max_dist) ** DISTANCE_WEIGHT

    return dist_mat

def calc_image_moments(image):
    '''
    Calculates the image moments for an image.

    For more information see:
        - https://learnopencv.com/shape-matching-using-hu-moments-c-python/
    '''

    eps = 0.00001

    size_z = image.shape[0]
    size_y = image.shape[1]
    size_x = image.shape[2]

    x_grid, y_grid, z_grid = torch.meshgrid(torch.arange(0, size_x), torch.arange(0, size_y), torch.arange(0, size_z))
    x_grid = x_grid.to(image.device)
    y_grid = y_grid.to(image.device)
    z_grid = z_grid.to(image.device)

    image_moments = Dict()

    # image moments till order 4
    for i in range(4):
        for j in range(4):
            for k in range(4):
                m_ijk = torch.sum((z_grid ** i) * (y_grid ** j) * (x_grid ** k) * image)
                image_moments[f"m{i}{j}{k}"] = m_ijk

    if image_moments['m000'] < eps:
        image_moments['z_avg'] = (image.shape[0] - 1) / 2
        image_moments['y_avg'] = (image.shape[1] - 1) / 2
        image_moments['x_avg'] = (image.shape[2] - 1) / 2
    else:
        image_moments['z_avg'] = image_moments['m100'] / image_moments['m000']
        image_moments['y_avg'] = image_moments['m010'] / image_moments['m000']
        image_moments['x_avg'] = image_moments['m001'] / image_moments['m000']

    # Moment invariants: translation invariant
    for i in range(4):
        for j in range(4):
            for k in range(4):
                if i == j == k == 0:
                    continue
                mu_ijk = torch.sum(((z_grid-image_moments['z_avg']) ** i) * ((y_grid-image_moments['y_avg']) ** j) * ((x_grid-image_moments['x_avg']) ** k) * image)
                image_moments[f"mu{i}{j}{k}"] = mu_ijk

    # Moment invariants: scale invariant
    for i in range(4):
        for j in range(4):
            for k in range(4):
                if i == j == k == 0:
                    continue
                if image_moments['m000'] < eps:
                    eta_ijk = 0
                else:
                    eta_ijk = image_moments[f"mu{i}{j}{k}"] / image_moments['m000'] ** ((i+j+k) / 3 + 1)
                image_moments[f"eta{i}{j}{k}"] = eta_ijk

    return image_moments


class ImageStatisticsRepresentation(OutputRepresentation):

    @staticmethod
    def default_config():
        default_config = OutputRepresentation.default_config()
        default_config.env_size = (16,16,16)
        default_config.channel_list = list(range(1, 10))
        default_config.distance_function = "L2"
        default_config.device = "cuda"
        return default_config

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        # model
        self.statistic_names = ['activation_mass', 'activation_volume', 'activation_density', 'activation_mass_distribution']
        self.n_channels = len(self.config.channel_list)
        self.n_latents = len(self.statistic_names) * self.n_channels

    def calc_static_statistics(self, final_obs):
        '''Calculates the final statistics for evocraft last observation'''

        feature_vector = []

        size_z = self.config.env_size[0]
        size_y = self.config.env_size[1]
        size_x = self.config.env_size[2]
        num_of_cells = size_z * size_y * size_x

        # calc initial center of mass and use it as a reference point to "center" the world around it
        mid_z = (size_z - 1) / 2
        mid_y = (size_y - 1) / 2
        mid_x = (size_x - 1) / 2
        mid = torch.tensor([mid_z, mid_y, mid_x])

        for channel_idx in self.config.channel_list:
            activation_center_of_mass = torch.tensor(center_of_mass(final_obs[channel_idx]))
            activation_shift_to_center = mid - activation_center_of_mass

            activation = final_obs[channel_idx]
            centered_activation = roll_n(activation, 0, activation_shift_to_center[0].int())
            centered_activation = roll_n(centered_activation, 1, activation_shift_to_center[1].int())

            # calculate the image moments
            activation_moments = calc_image_moments(centered_activation)

            # activation mass
            activation_mass = activation_moments.m000
            activation_mass_data = activation_mass / num_of_cells  # activation is number of acitvated cells divided by the number of cells
            if 'activation_mass' in self.statistic_names:
                feature_vector.append(activation_mass_data)

            # activation volume
            activation_volume = torch.sum(torch.relu(activation-EPS))
            activation_volume_data = activation_volume / num_of_cells
            if 'activation_volume' in self.statistic_names:
                feature_vector.append(activation_volume_data)

            # activation density
            if activation_volume == 0:
                activation_density_data = torch.tensor(0.).to(self.config.device)
            else:
                activation_density_data = activation_mass / activation_volume
            if 'activation_density' in self.statistic_names:
                feature_vector.append(activation_density_data)

            # mass distribution around the center
            distance_weight_matrix = calc_distance_matrix(self.config.env_size).to(self.config.device)
            if activation_mass <= EPS:
                activation_mass_distribution = torch.tensor(1.0).to(self.config.device)
            else:
                activation_mass_distribution = torch.sum(distance_weight_matrix * centered_activation) / torch.sum(centered_activation)

            activation_mass_distribution_data = activation_mass_distribution
            if 'activation_mass_distribution' in self.statistic_names:
                feature_vector.append(activation_mass_distribution_data)

        feature_vector = torch.stack(feature_vector)

        return feature_vector


    def calc(self, observations):
        """
            Maps the observations of a system to an embedding vector
            Return a torch tensor
        """
        # filter low values
        last_potential = observations.potentials[-1]
        air_potential = last_potential[0,0,0,0].detach().item()
        filtered_im = torch.where(last_potential > air_potential, last_potential, torch.zeros_like(last_potential))

        embedding = self.calc_static_statistics(filtered_im)

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
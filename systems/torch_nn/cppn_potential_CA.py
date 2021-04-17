from addict import Dict
import torch
import numbers
from evocraftsearch import System
from evocraftsearch.utils.torch_utils import SphericPad
from evocraftsearch.spaces import Space, BoxSpace, DiscreteSpace, DictSpace, MultiBinarySpace
from evocraftsearch.evocraft.utils import get_minecraft_color_list
import pytorchneat
from pytorchneat.activations import str_to_activation
from copy import deepcopy
import warnings
import time
import MinkowskiEngine as ME
import open3d as o3d


def str_to_tuple_key(str_key):
    splitted_key = str_key.split("_")
    return (int(splitted_key[0]), int(splitted_key[1]))

def tuple_to_str_key(tuple_key):
    return f"{tuple_key[0]}_{tuple_key[1]}"

""" =============================================================================================
Initialization Space: 
    - CPPNInitSpace
============================================================================================= """

class CPPNSpace(Space):

    @staticmethod
    def default_config():
        default_config = Dict()
        return default_config

    def __init__(self, config={}, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        super().__init__(shape=None, dtype=None)

    def sample(self):
        genome = self.config.neat_config.genome_type(0)
        genome.configure_new(self.config.neat_config.genome_config)
        return genome

    def mutate(self, genome):
        # genome: policy_parameters.init_matnucleus_genome pytorchneat.SelfConnectionGenome
        new_genome = deepcopy(genome)
        new_genome.mutate(self.config.neat_config.genome_config)
        return new_genome

    def crossover(self, genome_1, genome_2):
        genome_1 = deepcopy(genome_1)
        genome_2 = deepcopy(genome_2)
        if genome_1.fitness is None:
            genome_1.fitness = 0.0
        if genome_2.fitness is None:
            genome_2.fitness = 0.0
        child_1 = self.config.neat_config.genome_type(0)
        child_1.configure_crossover(genome_1, genome_2, self.config.neat_config.genome_config)
        child_2 = self.config.neat_config.genome_type(0)
        child_2.configure_crossover(genome_1, genome_2, self.config.neat_config.genome_config)
        return child_1, child_2


    def contains(self, x):
        # TODO from neat config max connections/weights
        return True

    def clamp(self, x):
        # TODO from neat config max connections/weights
        return x

class CppnPotentialCAInitializationSpace(DictSpace):

    @staticmethod
    def default_config():
        default_config = Dict()
        default_config.neat_config = None
        default_config.cppn_n_passes = 3
        return default_config

    def __init__(self,  config={}, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        spaces = Dict(
            cppn_genome=CPPNSpace(self.config)
        )

        DictSpace.__init__(self, spaces=spaces)



""" =============================================================================================
Update Rule Space: 
============================================================================================= """

class BiasedMultiBinarySpace(MultiBinarySpace):

    def __init__(self, n, indpb_sample=1.0, indpb=1.0):

        if type(n) in [tuple, list, torch.tensor]:
            input_n = n
        else:
            input_n = (n,)
        if isinstance(indpb_sample, numbers.Number):
            indpb_sample = torch.full(input_n, indpb_sample, dtype=torch.float64)
        self.indpb_sample = torch.as_tensor(indpb_sample, dtype=torch.float64)

        MultiBinarySpace.__init__(self, n=n, indpb=indpb)

    def sample(self):
        return torch.bernoulli(self.indpb_sample).to(self.dtype)



class CppnPotentialCAUpdateRuleSpace(Space):

    @staticmethod
    def default_config():
        default_config = Dict()
        default_config.neat_config = None
        default_config.cppn_n_passes = 2
        return default_config

    def __init__(self, n_blocks, config={}, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        self.n_blocks = n_blocks

        self.T = BoxSpace(low=1.0, high=2.0, shape=(), mutation_mean=0.0, mutation_std=0.2, indpb=0.5, dtype=torch.float32)

        n_cross_channels = (self.n_blocks - 1) ** 2
        indpb_sample_cross_channels = []
        indpb_mutate_cross_channels = []
        for c0 in range(1, self.n_blocks):
            for c1 in range(1, self.n_blocks):
                if c1 == c0: # higher sampling rate and lower mutation rate for channelwise kernels
                    indpb_sample_cross_channels.append(1.0)
                    indpb_mutate_cross_channels.append(0.05)
                else:
                    indpb_sample_cross_channels.append(0.2)
                    indpb_mutate_cross_channels.append(0.1)

        self.C = BiasedMultiBinarySpace(n=n_cross_channels, indpb_sample=indpb_sample_cross_channels, indpb=indpb_mutate_cross_channels)

        self.K = DictSpace(
                R=DiscreteSpace(n=3, mutation_mean=0.0, mutation_std=0.5, indpb=0.5),
                cppn_genome=CPPNSpace(self.config),
                m=BoxSpace(low=0.0, high=1.0, shape=(), mutation_mean=0.0, mutation_std=0.1, indpb=0.5, dtype=torch.float32),
                s=BoxSpace(low=0.001, high=0.3, shape=(), mutation_mean=0.0, mutation_std=0.05, indpb=0.5, dtype=torch.float32),
            )

    def sample(self):
        x = Dict()
        x['T'] = self.T.sample()
        x['C'] = self.C.sample()
        x['K'] = Dict()
        unrolled_idx = 0
        for c0 in range(1, self.n_blocks):
            for c1 in range(1, self.n_blocks):
                is_cross_kernel = bool(x['C'][unrolled_idx])
                if is_cross_kernel:
                    x['K'][tuple_to_str_key((c0,c1))] = self.K.sample()
                unrolled_idx += 1
        return x

    def mutate(self, x):
        # genome: policy_parameters.init_matnucleus_genome pytorchneat.SelfConnectionGenome
        new_x = Dict()
        new_x['T'] = self.T.mutate(x['T'])
        new_x['C'] = self.C.mutate(x['C'])
        new_x['K'] = Dict()
        unrolled_idx = 0
        for c0 in range(1, self.n_blocks):
            for c1 in range(1, self.n_blocks):
                is_cross_kernel = new_x['C'][unrolled_idx]
                if is_cross_kernel:
                    was_cross_kernel = x['C'][unrolled_idx]
                    if was_cross_kernel:
                        new_x['K'][tuple_to_str_key((c0, c1))] = self.K.mutate(x['K'][tuple_to_str_key((c0, c1))])
                    else:
                        new_x['K'][tuple_to_str_key((c0, c1))] = self.K.sample()
                unrolled_idx += 1
        return new_x

    def crossover(self, x1, x2):
        # genome: policy_parameters.init_matnucleus_genome pytorchneat.SelfConnectionGenome
        child_1, child_2 = Dict(), Dict()
        child_1['T'], child_2['T'] = self.T.crossover(x1['T'], x2['T'])
        child_1['C'], child_2['C'] = self.C.crossover(x1['C'], x2['C'])
        child_1['K'], child_2['K'] = Dict(), Dict()

        unrolled_idx = 0
        for c0 in range(1, self.n_blocks):
            for c1 in range(1, self.n_blocks):
                was_x1_cross_kernel = x1['C'][unrolled_idx]
                was_x2_cross_kernel = x2['C'][unrolled_idx]
                is_child_1_cross_kernel = child_1['C'][unrolled_idx].bool().item()
                is_child_2_cross_kernel = child_2['C'][unrolled_idx].bool().item()

                if is_child_1_cross_kernel or is_child_2_cross_kernel:
                    if was_x1_cross_kernel:
                        cur_x1_K = x1['K'][tuple_to_str_key((c0, c1))]
                    else:
                        cur_x1_K = self.K.sample()

                    if was_x2_cross_kernel:
                        cur_x2_K = x2['K'][tuple_to_str_key((c0, c1))]
                    else:
                        cur_x2_K = self.K.sample()

                    cur_child_1, cur_child_2 = self.K.crossover(cur_x1_K, cur_x2_K)

                    if is_child_1_cross_kernel:
                        child_1['K'][tuple_to_str_key((c0, c1))] = cur_child_1

                    if is_child_2_cross_kernel:
                        child_2['K'][tuple_to_str_key((c0, c1))] = cur_child_2

                unrolled_idx += 1

        return child_1, child_2

    def contains(self, x):
        return self.T.contains(x['T']) and all(self.K.contains(K) for K in x['K'].values())

    def clamp(self, x):
        x['T'] = self.T.clamp(x['T'])
        for str_key, K in x['K'].items():
            x['K'][str_key] = self.K.clamp(K)
        return x



""" =============================================================================================
CppnPotentialCA Main
============================================================================================= """

# CppnPotentialCA Step version (faster)
class CppnPotentialCAStep(torch.nn.Module):
    """ Module pytorch that computes one CppnPotentialCA Step with the fft version"""

    def __init__(self, T, K, cppn_config, max_potential=10.0, is_soft_clip=False, SX=16, SY=16, SZ=16, device='cpu'):
        torch.nn.Module.__init__(self)

        self.cppn_config = cppn_config
        self.SX = SX
        self.SY = SY
        self.SZ = SZ
        self.is_soft_clip = is_soft_clip

        self.device = device

        self.register_parameter('T', torch.nn.Parameter(T))
        self.register_kernels(K)

        self.gfunc = lambda n, m, s: torch.exp(- (n - m) ** 2 / (2 * s ** 2)) * 2 - 1
        self.max_potential = max_potential

    def register_kernels(self, K):
        self.K = torch.nn.Module()
        for str_key, K in K.items():
            module_cur_K = torch.nn.Module()
            module_cur_K.register_buffer('R', K['R']+2)
            module_cur_K.add_module('pad', SphericPad(K['R']+2))
            module_cur_K.add_module('cppn_net', pytorchneat.rnn.RecurrentNetwork.create(K['cppn_genome'], self.cppn_config, device=self.device))
            module_cur_K.register_parameter('m', torch.nn.Parameter(K['m']))
            module_cur_K.register_parameter('s', torch.nn.Parameter(K['s']))
            self.K.add_module(str_key, module_cur_K)

    def reset_kernels(self):
        self.kernels = Dict()
        for str_key, K in self.K.named_children():
            kernel_SY = kernel_SX = kernel_SZ = 2 * K.R + 1
            cppn_input = pytorchneat.utils.create_image_cppn_input((kernel_SY, kernel_SX, kernel_SZ), is_distance_to_center=True, is_bias=True)
            cppn_output_kernel = (1.0 - K.cppn_net.activate(cppn_input, 2).abs()).squeeze(-1).unsqueeze(0).unsqueeze(0)
            kernel_sum = torch.sum(cppn_output_kernel)
            if kernel_sum > 0:
                self.kernels[str_key] = cppn_output_kernel / kernel_sum
            else:
                self.kernels[str_key] = cppn_output_kernel

    def forward(self, input):
        field = torch.zeros_like(input)
        for str_key, K in self.K.named_children():
            c0, c1 = str_to_tuple_key(str_key)
            with torch.no_grad():
                padded_input = K.pad(input[:, :, :, :, c0]).unsqueeze(0)
            potential = torch.nn.functional.conv3d(padded_input, weight=self.kernels[str_key])
            field[:, :, :, :, c1] += self.gfunc(potential, K.m, K.s).squeeze(1)

        # because c1=0 not in kernels, the potential of the air is fixed to cste, meaning that to survive potential of other block types must be > 1!
        output_potential = torch.clamp(input + (1.0 / self.T) * field, min=0.0, max=self.max_potential)

        return output_potential


class CppnPotentialCA(System, torch.nn.Module):

    @staticmethod
    def default_config():
        default_config = System.default_config()
        default_config.SX = 16
        default_config.SY = 16
        default_config.SZ = 16
        default_config.block_list = ['AIR', 'CLAY', 'SLIME', 'PISTON', 'STICKY_PISTON', 'REDSTONE_BLOCK'] # block 0 is always air
        default_config.final_step = 10

        default_config.air_potential = 1.0
        default_config.max_potential = 10.0
        default_config.initialization_cppn.presence_logit = 'gumbel' # 'threshold' or 'gumbel'
        default_config.initialization_cppn.presence_bias = 1.0  # bias for sampling presence (when negative encourages absence)
        default_config.initialization_cppn.occupation_ratio = 1.0 / 1.0 # the initial state is confined to occupation_ratio of the world grid

        return default_config


    def __init__(self, initialization_space=None, update_rule_space=None, intervention_space=None, config={}, device='cpu', **kwargs):
        System.__init__(self, config=config, device=device, **kwargs)
        torch.nn.Module.__init__(self)

        self.n_blocks = len(self.config.block_list)
        assert self.config.block_list[0] == 'AIR'
        self.blocks_colorlist = get_minecraft_color_list(self.config.block_list)

        self.device = device

        if initialization_space is not None:
            self.initialization_space = initialization_space
        else:
            self.initialization_space = CppnPotentialCAInitializationSpace()
        if update_rule_space is not None:
            self.update_rule_space = update_rule_space
        else:
            self.update_rule_space = CppnPotentialCAUpdateRuleSpace(self.n_blocks)
        self.intervention_space = intervention_space
        self.run_idx = 0

        self.to(self.device)

    def sample_policy_parameters(self):
        policy = Dict()
        policy['initialization'] = self.initialization_space.sample()
        policy['update_rule'] = self.update_rule_space.sample()
        return policy

    def crossover_policy_parameters(self, policy_1, policy_2):
        child_1_policy, child_2_policy = Dict(), Dict()
        child_1_policy['initialization'], child_2_policy['initialization'] = self.initialization_space.crossover(
            policy_1['initialization'], policy_2['initialization'])
        child_1_policy['update_rule'], child_2_policy['update_rule'] = self.update_rule_space.crossover(
            policy_1['update_rule'], policy_2['update_rule'])
        return child_1_policy, child_2_policy

    def mutate_policy_parameters(self, policy):
        new_policy = Dict()
        new_policy['initialization'] = self.initialization_space.mutate(policy['initialization'])
        new_policy['update_rule'] = self.update_rule_space.mutate(policy['update_rule'])
        return new_policy


    def reset(self, policy=None):
        # call the property setters
        self.initialization_parameters = policy['initialization']
        self.update_rule_parameters = policy['update_rule']

        # initialize CppnPotentialCA CA with update rule parameters
        cppn_potential_ca_step = CppnPotentialCAStep(self.update_rule_parameters['T'], self.update_rule_parameters['K'], self.update_rule_space.config.neat_config, self.config.max_potential, is_soft_clip=False, SX=self.config.SX, SY=self.config.SY, SZ=self.config.SZ, device=self.device)
        self.add_module('cppn_potential_ca_step', cppn_potential_ca_step)

        # initialize CppnPotentialCA initial potential with initialization_parameters
        # The explorer can sample:
        # (i) which channel to start with
        # (ii) for each selected channel:  a cppn with n_outputs = 1 and a bbox to position it (+resolution?)
        cppn_genome = self.initialization_parameters['cppn_genome']
        initialization_cppn = pytorchneat.rnn.RecurrentNetwork.create(cppn_genome, self.initialization_space.config.neat_config, device=self.device)
        self.add_module('initialization_cppn', initialization_cppn)

        # v1 differentiable: n_outputs CPPN = 2 + (n_blocks-1)
        # first two outputs are given a ReLU activation function and serve as logits (+- presence bias) for presence classification
        if self.config.initialization_cppn.presence_logit == 'gumbel':
            # the other ones are for logits block types so we assign them relu activations
            for i in range(self.n_blocks+1):
                self.initialization_cppn.output_activations[i] = str_to_activation['relu']
        # v2 hard threshold: n_outputs CPPN = 1 + (n_blocks-1)
        # first output is given an activation that maps to [-1,1] and then we threshold > presence bias for presence classification
        elif self.config.initialization_cppn.presence_logit == 'threshold':
            assert self.initialization_cppn.output_activations[0].__name__.split('_')[:-1] in ['delphineat_sigmoid', 'delphineat_gauss', 'tanh', 'sin']
            # the other ones are for logits block types so we assign them relu activations
            for i in range (1, self.n_blocks):
                self.initialization_cppn.output_activations[i] = str_to_activation['relu']
        else:
            raise NotImplementedError

        # push the nn.Module and the available devoce
        self.to(self.device)


    def generate_update_rule_kernels(self):
        self.cppn_potential_ca_step.reset_kernels()


    def generate_init_potential(self):
        # TODO: tau as attribute to decrease

        # the cppn generated output is confined to a limited space:
        cppn_output_width = int(self.config.SX * self.config.initialization_cppn.occupation_ratio)
        cppn_output_height = int(self.config.SY * self.config.initialization_cppn.occupation_ratio)
        cppn_output_depth = int(self.config.SZ * self.config.initialization_cppn.occupation_ratio)
        cppn_input = pytorchneat.utils.create_image_cppn_input((cppn_output_width, cppn_output_height, cppn_output_depth), is_distance_to_center=True, is_bias=True)
        cppn_output = self.initialization_cppn.activate(cppn_input, self.initialization_space.config.cppn_n_passes)

        # compute presence/absence mask
        if self.config.initialization_cppn.presence_logit == 'gumbel':
            cppn_output_presence_logits = cppn_output[:, :, :, :2]
            cppn_output_presence_logits[:, :, :, 1] += self.config.initialization_cppn.presence_bias
            cppn_output_presence_logits = torch.log(cppn_output_presence_logits)
            cppn_output_presence = torch.nn.functional.gumbel_softmax(cppn_output_presence_logits, tau=0.01, hard=True, eps=1e-10, dim=-1).argmax(-1)
        elif self.config.initialization_cppn.presence_logit == 'threshold':
            cppn_output_presence = cppn_output[:, :, :, :0] < self.config.initialization_cppn.presence_bias
        else:
            raise NotImplementedError
        cppn_output_presence = cppn_output_presence.unsqueeze(-1)

        # compute block potentials
        cppn_output_block_potentials = cppn_output_presence * torch.clamp(cppn_output[:, :, :, -(self.n_blocks-1):], min=0.0, max=self.config.max_potential)

        # generate the world sparse init potential
        init_potential = torch.zeros(1, self.config.SX, self.config.SY, self.config.SZ, self.n_blocks, dtype=torch.float64)
        init_potential[0, :, :, :, 0] = self.config.air_potential
        offset = int((self.config.SX - cppn_output_width) // 2.0)
        init_potential[0, offset:offset+cppn_output_width, offset:offset+cppn_output_height, offset:offset+cppn_output_depth, 1:] = cppn_output_block_potentials
        self.potential = init_potential.to(self.device)
        sparse_mask = torch.nn.functional.gumbel_softmax(torch.log(self.potential.detach()), tau=0.01, hard=True).argmax(-1) > 0 #todo: dont detach?
        self.sparse_potential = ME.to_sparse(sparse_mask.unsqueeze(-1).repeat(1, 1, 1, 1,self.n_blocks) * self.potential, format="BXXXC")
        self.step_idx = 0


    def update_initialization_parameters(self):
        new_initialization_parameters = Dict()
        new_initialization_parameters['cppn_genome'] = self.initialization_cppn.update_genome(self.initialization_parameters['cppn_genome'])
        if not self.initialization_space.contains(new_initialization_parameters):
            new_initialization_parameters = self.initialization_space.clamp(new_initialization_parameters)
            warnings.warn('provided parameters are not in the space range and are therefore clamped')
        self._initialization_parameters = new_initialization_parameters


    def update_update_rule_parameters(self):
        new_update_rule_parameters = deepcopy(self.update_rule_parameters)
        new_update_rule_parameters['T'] = self.cppn_potential_ca_step.T.data
        for str_key, module in self.cppn_potential_ca_step.K.named_children():
            new_update_rule_parameters['K'][str_key]['m'] = module.m.data
            new_update_rule_parameters['K'][str_key]['s'] = module.s.data
        if not self.update_rule_space.contains(new_update_rule_parameters):
            new_update_rule_parameters = self.update_rule_space.clamp(new_update_rule_parameters)
            warnings.warn('provided parameters are not in the space range and are therefore clamped')
        self._update_rule_parameters = new_update_rule_parameters


    def optimize(self, optim_steps, loss_func):
        """
        loss func: torch differentiable function that maps observations to a scalar
        """
        self.train()
        train_losses = []

        self.optimizer = torch.optim.Adam([{'params': self.initialization_cppn.parameters(), 'lr': 0.001},
                                             {'params': self.cppn_potential_ca_step.K.parameters(), 'lr': 0.01},
                                             {'params': self.cppn_potential_ca_step.T, 'lr': 0.1}])

        for optim_step_idx in range(optim_steps):

            # run system
            observations = self.system.run()

            # compute error between reached_goal and target_goal
            loss = loss_func(observations)

            # optimisation step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if optim_step_idx > 3 and abs(old_loss - loss.item()) < 1e-4:
                break
            old_loss = loss.item()

            train_losses.append(loss.item())

        # gather back the trained parameters
        self.system.update_initialization_parameters()
        self.system.update_update_rule_parameters()

        return train_losses

    def step(self, intervention_parameters=None):
        # clamp params if was changed outside of allowed bounds with gradient Descent
        with torch.no_grad():
            if not self.update_rule_space.T.contains(self.cppn_potential_ca_step.T.data.cpu()):
                self.cppn_potential_ca_step.T.data = self.update_rule_space.T.clamp(self.cppn_potential_ca_step.T.data)
            for str_key, module in self.cppn_potential_ca_step.K.named_children():
                if not self.update_rule_space.K['m'].contains(module.m.data.cpu()):
                    module.m.data = self.update_rule_space.K['m'].clamp(module.m.data)
                if not self.update_rule_space.K['s'].contains(module.s.data.cpu()):
                    module.s.data = self.update_rule_space.K['s'].clamp(module.s.data)
        self.potential = self.cppn_potential_ca_step(self.potential)
        sparse_mask = torch.nn.functional.gumbel_softmax(torch.log(self.potential.detach()), tau=0.01, hard=True).argmax(-1) > 0
        self.sparse_potential = ME.to_sparse(sparse_mask.unsqueeze(-1).repeat(1, 1, 1, 1, self.n_blocks) * self.potential, format="BXXXC")
        self.step_idx += 1

        return self.potential


    def forward(self):
        potential = self.step(None)
        return potential


    def run(self):
        self.generate_update_rule_kernels()
        self.generate_init_potential()
        observations = Dict()
        observations.timepoints = list(range(self.config.final_step))
        observations.potentials = torch.empty((self.config.final_step, self.config.SX, self.config.SY, self.config.SZ, self.n_blocks))
        observations.potentials[0] = self.potential[0]
        sparse_potentials_coords = [self.sparse_potential.C]
        sparse_potentials_feats = [self.sparse_potential.F]
        for step_idx in range(1, self.config.final_step):
            cur_observation = self.step(None)
            observations.potentials[step_idx] = cur_observation[0]
            sparse_potentials_coords.append(self.sparse_potential.C)
            sparse_potentials_feats.append(self.sparse_potential.F)

        return observations


    def render(self, mode="human"):
        vis = o3d.visualization.Visualizer()
        vis.create_window('Discovery',800,800)
        pcd = o3d.geometry.PointCloud()
        cur_potential = self.potential[0].cpu().detach()
        coords = []
        feats = []
        for i in range(self.config.SX):
            for j in range(self.config.SY):
                for k in range(self.config.SZ):
                    block_id = cur_potential[i, j, k].cpu().detach().argmax()
                    if block_id > 0:
                        coords.append(torch.tensor([i, j, k], dtype=torch.float64))
                        feats.append(block_id.unsqueeze(-1))
        if len(coords) > 0:
            coords = torch.stack(coords)
            feats = torch.stack(feats)
            pcd.points = o3d.utility.Vector3dVector(coords)
            pcd.colors = o3d.utility.Vector3dVector(torch.stack([torch.tensor(self.blocks_colorlist[feats[i]][:3]) for i in range(len(feats))]))
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1.0)
            vis.add_geometry(voxel_grid)
        if mode == "human":
           vis.run()
           vis.close()
           return
        elif mode == "rgb_array":
            out_image = vis.capture_screen_float_buffer(True)
            return out_image
        else:
            raise NotImplementedError


    def render_video(self, states, mode="human"):
        vis = o3d.visualization.Visualizer()
        vis.create_window('states', 800, 800)
        pcd = o3d.geometry.PointCloud()
        step_idx = 0
        cur_frame_C = []
        cur_frame_F = []
        for coords, feats in zip(states.C, states.F):
            cur_step_idx = coords[0]
            if cur_step_idx != step_idx:
                vis.clear_geometries()
                if len(cur_frame_C) > 0:
                    pcd.points = o3d.utility.Vector3dVector(cur_frame_C)
                    pcd.colors = o3d.utility.Vector3dVector(torch.stack([torch.tensor(self.blocks_colorlist[cur_frame_F[i].argmax()][:3]) for i in range(len(cur_frame_F))]))
                    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1.0)
                    vis.add_geometry(voxel_grid)
                    vis.poll_events()
                    vis.update_renderer()
                    time.sleep(2)
                cur_frame_C = []
                cur_frame_F = []
                step_idx = cur_step_idx
            else:
                cur_frame_C.append(coords[1:])
                cur_frame_F.append(feats)
        if mode == "human":
            vis.destroy_window()
            vis.close()
            return
        elif mode == "rgb_array":
            out_image = vis.capture_screen_float_buffer(True)
            return out_image
        else:
            raise NotImplementedError


    def close(self):
        pass
from addict import Dict
import torch
import numbers
from evocraftsearch import System
from evocraftsearch.utils.torch_utils import SphericPad, roll_n, complex_mult_torch, soft_clip, soft_max
from evocraftsearch.spaces import Space, BoxSpace, DiscreteSpace, DictSpace, MultiBinarySpace
import pytorchneat
from pytorchneat.activations import str_to_activation
from copy import deepcopy
import warnings
import time
import MinkowskiEngine as ME
import open3d as o3d
import matplotlib.pyplot as plt
cmap = plt.cm.jet
colorlist = [cmap(i) for i in range(0,cmap.N,20)]

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
        default_config.cppn_n_passes = 2
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
        return torch.bernoulli(self.indpb_sample)



class CppnPotentialCAUpdateRuleSpace(Space):

    @staticmethod
    def default_config():
        default_config = Dict()
        default_config.n_blocks = 10
        default_config.neat_config = None
        default_config.cppn_n_passes = 2
        return default_config

    def __init__(self, config={}, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        self.T = BoxSpace(low=1.0, high=20.0, shape=(), mutation_mean=0.0, mutation_std=0.5, indpb=1.0, dtype=torch.float32)

        n_cross_channels = (self.config.n_blocks - 1) ** 2
        indpb_sample_cross_channels = []
        indpb_mutate_cross_channels = []
        for c0 in range(1, self.config.n_blocks):
            for c1 in range(1, self.config.n_blocks):
                if c1 == c0: # higher sampling rate and lower mutation rate for channelwise kernels
                    indpb_sample_cross_channels.append(1.0)
                    indpb_mutate_cross_channels.append(0.05)
                else:
                    indpb_sample_cross_channels.append(0.4)
                    indpb_mutate_cross_channels.append(0.1)

        self.C = BiasedMultiBinarySpace(n=n_cross_channels, indpb_sample=indpb_sample_cross_channels, indpb=indpb_mutate_cross_channels)

        self.K = DictSpace(
                R=DiscreteSpace(n=3, mutation_mean=0.0, mutation_std=0.5, indpb=1.0),
                cppn_genome=CPPNSpace(self.config),
                m=BoxSpace(low=0.0, high=1.0, shape=(), mutation_mean=0.0, mutation_std=0.1, indpb=1.0, dtype=torch.float32),
                s=BoxSpace(low=0.001, high=0.3, shape=(), mutation_mean=0.0, mutation_std=0.05, indpb=1.0, dtype=torch.float32),
            )

    def sample(self):
        x = Dict()
        x['T'] = self.T.sample()
        x['C'] = self.C.sample()
        x['K'] = Dict()
        unrolled_idx = 0
        for c0 in range(1, self.config.n_blocks):
            for c1 in range(1, self.config.n_blocks):
                is_cross_kernel = x['C'][unrolled_idx]
                if is_cross_kernel:
                    x['K'][tuple_to_str_key((c0,c1))] = self.K.sample()
        return x

    def mutate(self, x):
        # genome: policy_parameters.init_matnucleus_genome pytorchneat.SelfConnectionGenome
        new_x = Dict()
        new_x['T'] = self.T.mutate(x['T'])
        new_x['C'] = self.C.mutate(x['C'])
        new_x['K'] = Dict()
        unrolled_idx = 0
        for c0 in range(1, self.config.n_blocks):
            for c1 in range(1, self.config.n_blocks):
                is_cross_kernel = new_x['C'][unrolled_idx]
                if is_cross_kernel:
                    was_cross_kernel = x['C'][unrolled_idx]
                    if was_cross_kernel:
                        new_x['K'][tuple_to_str_key((c0, c1))] = self.K.mutate(x['K'][tuple_to_str_key((c0, c1))])
                    else:
                        new_x['K'][tuple_to_str_key((c0, c1))] = self.K.sample()
                unrolled_idx += 1
        return new_x

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
            module_cur_K.add_module('cppn_net', pytorchneat.rnn.RecurrentNetwork.create(K['cppn_genome'], self.cppn_config))
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
        default_config.n_blocks = 10 # block 0 is always air
        default_config.final_step = 10

        default_config.air_potential = 0.5
        default_config.max_potential = 10.0
        default_config.initialization_cppn.presence_logit = 'gumbel' # 'threshold' or 'gumbel'
        default_config.initialization_cppn.presence_bias = -0.3 # bias for sampling presence (when negative encourages absence)
        default_config.initialization_cppn.occupation_ratio = 1.0 / 2.0 # the initial state is confined to occupation_ratio of the world grid

        return default_config


    def __init__(self, initialization_space=None, update_rule_space=None, intervention_space=None, config={}, device=torch.device('cpu'), **kwargs):
        System.__init__(self, config=config, device=device, **kwargs)
        torch.nn.Module.__init__(self)

        self.device = device

        if initialization_space is not None:
            self.initialization_space = initialization_space
        else:
            self.initialization_space = CppnPotentialCAInitializationSpace()
        if update_rule_space is not None:
            self.update_rule_space = update_rule_space
        else:
            self.update_rule_space = CppnPotentialCAUpdateRuleSpace()
        self.intervention_space = intervention_space
        self.run_idx = 0

        self.to(self.device)


    def reset(self, initialization_parameters=None, update_rule_parameters=None):
        # call the property setters
        self.initialization_parameters = initialization_parameters
        self.update_rule_parameters = update_rule_parameters

        # initialize CppnPotentialCA CA with update rule parameters
        cppn_potential_ca_step = CppnPotentialCAStep(self.update_rule_parameters['T'], self.update_rule_parameters['K'], self.update_rule_space.config.neat_config, self.config.max_potential, is_soft_clip=False, SX=self.config.SX, SY=self.config.SY, SZ=self.config.SZ, device=self.device)
        self.add_module('cppn_potential_ca_step', cppn_potential_ca_step)

        # initialize CppnPotentialCA initial potential with initialization_parameters
        cppn_genome = self.initialization_parameters['cppn_genome']
        initialization_cppn = pytorchneat.rnn.RecurrentNetwork.create(cppn_genome, self.initialization_space.config.neat_config, device=self.device)
        self.add_module('initialization_cppn', initialization_cppn)

        # v1 differentiable: n_outputs CPPN = 2 + (n_blocks-1)
        # first two outputs are given a ReLU activation function and serve as logits (+- presence bias) for presence classification
        if self.config.initialization_cppn.presence_logit == 'gumbel':
            # the other ones are for logits block types so we assign them relu activations
            for i in range(self.config.n_blocks+1):
                self.initialization_cppn.output_activations[i] = str_to_activation['relu']
        # v2 hard threshold: n_outputs CPPN = 1 + (n_blocks-1)
        # first output is given an activation that maps to [-1,1] and then we threshold > presence bias for presence classification
        elif self.config.initialization_cppn.presence_logit == 'threshold':
            assert self.initialization_cppn.output_activations[0].__name__.split('_')[:-1] in ['delphineat_sigmoid', 'delphineat_gauss', 'tanh', 'sin']
            # the other ones are for logits block types so we assign them relu activations
            for i in range (1, self.config.n_blocks):
                self.initialization_cppn.output_activations[i] = str_to_activation['relu']
        else:
            raise NotImplementedError


        # push the nn.Module and the available devoce
        self.to(self.device)

        self.generate_init_potential()

    def generate_update_rule_kernels(self):
        self.cppn_potential_ca_step.reset_kernels()

    def generate_init_potential(self):
        # TODO: taus as attribute to decrease

        # the cppn generated output is confined to a limited space:
        cppn_output_height = int(self.config.SY * self.config.initialization_cppn.occupation_ratio)
        cppn_output_width = int(self.config.SX * self.config.initialization_cppn.occupation_ratio)
        cppn_output_depth = int(self.config.SY * self.config.initialization_cppn.occupation_ratio)
        cppn_input = pytorchneat.utils.create_image_cppn_input((cppn_output_height, cppn_output_width, cppn_output_depth), is_distance_to_center=True, is_bias=True)
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
        cppn_output_block_potentials = torch.clamp(cppn_output[:, :, :, -(self.config.n_blocks-1):], min=0.0, max=self.config.max_potential)

        # generate the world sparse init potential
        init_potential = torch.zeros(1, self.config.SY, self.config.SX, self.config.SZ, self.config.n_blocks, dtype=torch.float64)
        init_potential[0, :, :, :, 0] = self.config.air_potential
        offset = int((self.config.SY - cppn_output_height) // 2.0)
        init_potential[0, offset:offset+cppn_output_height, offset:offset+cppn_output_width, offset:offset+cppn_output_depth, 1:] = cppn_output_block_potentials
        self.potential = init_potential.to(self.device)
        self.state = torch.nn.functional.gumbel_softmax(torch.log(self.potential), tau=0.01, hard=True)
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
        new_update_rule_parameters['m'] = self.cppn_potential_ca_step.m.data
        new_update_rule_parameters['s'] = self.cppn_potential_ca_step.s.data
        if not self.update_rule_space.contains(new_update_rule_parameters):
            new_update_rule_parameters = self.update_rule_space.clamp(new_update_rule_parameters)
            warnings.warn('provided parameters are not in the space range and are therefore clamped')
        self._update_rule_parameters = new_update_rule_parameters

    def step(self, intervention_parameters=None):
        # clamp params if was changed outside of allowed bounds with gradient Descent
        if not self.update_rule_space.T.contains(self.cppn_potential_ca_step.T.data):
            self.cppn_potential_ca_step.T.data = self.update_rule_space.T.clamp(self.cppn_potential_ca_step.T.data)
        for str_key, module in self.cppn_potential_ca_step.K.named_children():
            if not self.update_rule_space.K['m'].contains(module.m.data):
                module.m.data = self.update_rule_space.K['m'].clamp(module.m.data)
            if not self.update_rule_space.K['s'].contains(module.s.data):
                module.s.data = self.update_rule_space.K['s'].clamp(module.s.data)
        #self.render()
        self.potential = self.cppn_potential_ca_step(self.potential)
        self.state = torch.nn.functional.gumbel_softmax(torch.log(self.potential), tau=0.01, hard=True)
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
        observations.potentials = torch.empty((self.config.final_step, self.config.SX, self.config.SY, self.config.SZ, self.config.n_blocks))
        observations.potentials[0] = self.potential[0]
        state_coords = []
        state_feats = []
        for i in range(self.config.SY):
            for j in range(self.config.SX):
                for k in range(self.config.SZ):
                    block_one_hot = self.state[0, i, j, k]
                    if block_one_hot.detach().argmax() > 0:
                        state_coords.append(torch.tensor([0, i, j, k], dtype=torch.float64))
                        state_feats.append(block_one_hot)
        for step_idx in range(1, self.config.final_step):
            cur_observation = self.step(None)
            observations.potentials[step_idx] = cur_observation[0]
            for i in range(self.config.SY):
                for j in range(self.config.SX):
                    for k in range(self.config.SZ):
                        block_one_hot = self.state[0, i, j, k]
                        if block_one_hot.detach().argmax() > 0:
                            state_coords.append(torch.tensor([step_idx, i, j, k], dtype=torch.float64))
                            state_feats.append(block_one_hot)
        if len(state_coords) > 0:
            observations.states = ME.SparseTensor(features=torch.stack(state_feats), coordinates=torch.stack(state_coords))

        return observations


    def render(self, mode="human"):
        vis = o3d.visualization.Visualizer()
        vis.create_window('Discovery',800,800)
        cur_state = self.state[0].cpu().detach()
        coords = []
        feats = []
        for i in range(self.config.SY):
            for j in range(self.config.SX):
                for k in range(self.config.SZ):
                    block_id = cur_state[i,j,k].argmax()
                    if block_id > 0:
                        coords.append(torch.tensor([i, j, k], dtype=torch.float64))
                        feats.append(block_id.unsqueeze(-1))
        if len(coords) > 0:
            sparse_state = ME.SparseTensor(torch.stack(feats), torch.stack(coords))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(sparse_state.C)
            pcd.colors = o3d.utility.Vector3dVector(torch.stack([torch.tensor(colorlist[sparse_state.F[i].cpu().detach()][:3]) for i in range(len(sparse_state.F))]))
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
                    pcd.colors = o3d.utility.Vector3dVector(torch.stack([torch.tensor(colorlist[cur_frame_F[i].argmax()][:3]) for i in range(len(cur_frame_F))]))
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
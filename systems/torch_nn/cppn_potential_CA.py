from addict import Dict
import torch
import numbers
from evocraftsearch import System
from evocraftsearch.utils.torch_utils import SphericPad
from evocraftsearch.spaces import Space, BoxSpace, DiscreteSpace, DictSpace, MultiBinarySpace, CPPNSpace
from evocraftsearch.evocraft.utils import get_minecraft_color_list
import pytorchneat
from math import floor, cos, sin, pi
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
============================================================================================= """

class CppnPotentialCAInitializationSpace(DictSpace):

    def __init__(self, n_blocks, neat_config):

        self.n_blocks = n_blocks
        self.neat_config = neat_config

        self.C = MultiBinarySpace(n=n_blocks-1, indpb=0.01)
        self.I = DictSpace(
                cppn_genome=CPPNSpace(neat_config),
                occupation_ratio=BoxSpace(low=0.2, high=0.4, shape=(), mutation_mean=0.0, mutation_std=0.1, indpb=0.5, dtype=torch.float32),
            )

    def sample(self):
        x = Dict()
        x['C'] = self.C.sample()
        x['I'] = Dict()
        for c in range(1, self.n_blocks):
            is_channel = bool(x['C'][c-1])
            if is_channel:
                x['I'][str(c)] = self.I.sample()
        return x

    def mutate(self, x):
        # genome: policy_parameters.init_matnucleus_genome pytorchneat.SelfConnectionGenome
        new_x = Dict()
        new_x['C'] = self.C.mutate(x['C'])
        new_x['I'] = Dict()
        for c in range(1, self.n_blocks):
            is_channel = bool(new_x['C'][c - 1])
            if is_channel:
                was_channel = bool(x['C'][c - 1])
                if was_channel:
                    new_x['I'][str(c)] = self.I.mutate(x['I'][str(c)])
                else:
                    new_x['I'][str(c)] = self.I.sample()
        return new_x

    def crossover(self, x1, x2):
        # genome: policy_parameters.init_matnucleus_genome pytorchneat.SelfConnectionGenome
        child_1, child_2 = Dict(), Dict()
        child_1['C'], child_2['C'] = self.C.crossover(x1['C'], x2['C'])
        child_1['I'], child_2['I'] = Dict(), Dict()

        for c in range(1, self.n_blocks):
            was_x1_channel = bool(x1['C'][c - 1])
            was_x2_channel = bool(x2['C'][c - 1])
            is_child_1_channel = bool(child_1['C'][c - 1])
            is_child_2_channel = bool(child_2['C'][c - 1])

            if is_child_1_channel or is_child_2_channel:
                if was_x1_channel:
                    cur_x1_K = x1['I'][str(c)]
                else:
                    cur_x1_K = self.I.sample()

                if was_x2_channel:
                    cur_x2_K = x2['I'][str(c)]
                else:
                    cur_x2_K = self.I.sample()

                cur_child_1, cur_child_2 = self.I.crossover(cur_x1_K, cur_x2_K)

                if is_child_1_channel:
                    child_1['I'][str(c)] = cur_child_1

                if is_child_2_channel:
                    child_2['I'][str(c)] = cur_child_2

        return child_1, child_2

    def contains(self, x):
        return all(self.I.contains(I) for I in x['I'].values())

    def clamp(self, x):
        for str_key, K in x['I'].items():
            x['I'][str_key] = self.I.clamp(K)
        return x


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

    def __init__(self, n_blocks, neat_config):

        self.n_blocks = n_blocks
        self.neat_config = neat_config

        self.T = BoxSpace(low=1.0, high=20.0, shape=(), mutation_mean=0.0, mutation_std=0.2, indpb=0.5, dtype=torch.float32)

        n_cross_channels = (self.n_blocks - 1) ** 2
        indpb_sample_cross_channels = []
        indpb_mutate_cross_channels = []
        for c0 in range(1, self.n_blocks):
            for c1 in range(1, self.n_blocks):
                if c1 == c0: # higher sampling rate and lower mutation rate for channelwise kernels
                    indpb_sample_cross_channels.append(1.0)
                    indpb_mutate_cross_channels.append(0.05)
                else:
                    indpb_sample_cross_channels.append(0.1)
                    indpb_mutate_cross_channels.append(0.1)

        self.C = BiasedMultiBinarySpace(n=n_cross_channels, indpb_sample=indpb_sample_cross_channels, indpb=indpb_mutate_cross_channels)

        self.K = DictSpace(
                R=DiscreteSpace(n=6, mutation_mean=0.0, mutation_std=0.5, indpb=0.5),
                cppn_genome=CPPNSpace(neat_config),
                m=BoxSpace(low=0.1, high=0.5, shape=(), mutation_mean=0.0, mutation_std=0.1, indpb=0.5, dtype=torch.float32),
                s=BoxSpace(low=0.1, high=0.3, shape=(), mutation_mean=0.0, mutation_std=0.05, indpb=0.5, dtype=torch.float32),
                h=BoxSpace(low=0.1, high=1.0, shape=(), mutation_mean=0.0, mutation_std=0.05, indpb=0.5, dtype=torch.float32),
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
                is_cross_kernel = bool(new_x['C'][unrolled_idx])
                if is_cross_kernel:
                    was_cross_kernel = bool(x['C'][unrolled_idx])
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
                was_x1_cross_kernel = bool(x1['C'][unrolled_idx])
                was_x2_cross_kernel = bool(x2['C'][unrolled_idx])
                is_child_1_cross_kernel = bool(child_1['C'][unrolled_idx])
                is_child_2_cross_kernel = bool(child_2['C'][unrolled_idx])

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
# CppnPotentialCA Initialization
class CppnPotentialCAInitialization(torch.nn.Module):
    """ Module pytorch that computes one CppnPotentialCA Initialization """

    def __init__(self, I, cppn_config, max_potential=1.0, device='cpu'):
        torch.nn.Module.__init__(self)

        self.cppn_config = cppn_config
        self.max_potential = max_potential
        self.device = device
        self.register_cppns(I)

        self.to(self.device)


    def register_cppns(self, I):
        self.I = torch.nn.Module()
        for str_key, I in I.items():
            module_cur_I = torch.nn.Module()
            module_cur_I.register_buffer('occupation_ratio', I['occupation_ratio'])
            module_cur_I.add_module('cppn_net', pytorchneat.rnn.RecurrentNetwork.create(I['cppn_genome'], self.cppn_config, device=self.device))
            self.I.add_module(str_key, module_cur_I)


    def forward(self, output_size=(16,16,16,10)):
        output_img = torch.zeros(output_size, device=self.device)
        for str_key, I in self.I.named_children():
            c = int(str_key)
            # the cppn generated output is confined to a limited space:
            c_output_size = tuple([int(s * I.occupation_ratio) for s in output_size[:-1]])
            c_cppn_input = pytorchneat.utils.create_image_cppn_input(c_output_size, is_distance_to_center=True, is_bias=True)
            c_cppn_output = (self.max_potential - self.max_potential * I.cppn_net.activate(c_cppn_input, 3).abs()).squeeze() # TODO: config cppn_n_passes
            offset_X = floor((output_size[0] - c_cppn_output.shape[0]) / 2)
            offset_Y = floor((output_size[1] - c_cppn_output.shape[1]) / 2)
            offset_Z = floor((output_size[2] - c_cppn_output.shape[2]) / 2)
            output_img[offset_X:offset_X+c_cppn_output.shape[0], offset_Y:offset_Y+c_cppn_output.shape[1], offset_Z:offset_Z+c_cppn_output.shape[2], c] = c_cppn_output
        return output_img


# CppnPotentialCA Step
class CppnPotentialCAStep(torch.nn.Module):
    """ Module pytorch that computes one CppnPotentialCA Step """

    def __init__(self, T, K, cppn_config, max_potential=1.0, is_soft_clip=False, device='cpu'):
        torch.nn.Module.__init__(self)

        self.cppn_config = cppn_config
        self.is_soft_clip = is_soft_clip

        self.device = device

        self.register_parameter('T', torch.nn.Parameter(T))
        self.register_kernels(K)

        self.gfunc = lambda n, m, s: torch.exp(- (n - m) ** 2 / (2 * s ** 2)) * 2 - 1
        self.max_potential = max_potential

    def register_kernels(self, K):
        self.K = torch.nn.Module()
        for str_key, cur_K in K.items():
            module_cur_K = torch.nn.Module()
            module_cur_K.register_buffer('R', cur_K['R']+2)
            module_cur_K.add_module('pad', SphericPad(cur_K['R']+2))
            module_cur_K.add_module('cppn_net', pytorchneat.rnn.RecurrentNetwork.create(cur_K['cppn_genome'], self.cppn_config, device=self.device))
            module_cur_K.register_parameter('m', torch.nn.Parameter(cur_K['m']))
            module_cur_K.register_parameter('s', torch.nn.Parameter(cur_K['s']))
            module_cur_K.register_parameter('h', torch.nn.Parameter(cur_K['h']))
            self.K.add_module(str_key, module_cur_K)

    def reset_kernels(self):
        self.kernels = Dict()
        for str_key, K in self.K.named_children():
            kernel_SY = kernel_SX = kernel_SZ = 2 * K.R + 1
            cppn_input = pytorchneat.utils.create_image_cppn_input((kernel_SY, kernel_SX, kernel_SZ), is_distance_to_center=True, is_bias=True)
            cppn_output_kernel = (self.max_potential - self.max_potential * K.cppn_net.activate(cppn_input, 3).abs()).squeeze(-1).unsqueeze(0).unsqueeze(0)
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
            delta_field_c1 = self.gfunc(potential, K.m, K.s).squeeze(1)
            field[:, :, :, :, c1] = field[:, :, :, :, c1] + K.h * delta_field_c1

        # because c1=0 not in kernels, the potential of the air is fixed to cste, meaning that to survive potential of other block types must be > air_potential
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
        return default_config


    def __init__(self, initialization_space=None, update_rule_space=None, intervention_space=None, config={}, device='cpu', **kwargs):
        System.__init__(self, config=config, device=device, **kwargs)
        torch.nn.Module.__init__(self)

        self.n_blocks = len(self.config.block_list)
        assert self.config.block_list[0] == 'AIR'
        self.blocks_colorlist = get_minecraft_color_list(self.config.block_list)

        self.device = device

        self.initialization_space = initialization_space
        self.update_rule_space = update_rule_space
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

        # initialize CppnPotentialCA initial potential with initialization_parameters
        ca_init = CppnPotentialCAInitialization(self.initialization_parameters['I'], self.initialization_space.neat_config, max_potential=self.config.max_potential, device=self.device)
        self.add_module('ca_init', ca_init)

        # initialize CppnPotentialCA CA step with update rule parameters
        ca_step = CppnPotentialCAStep(self.update_rule_parameters['T'], self.update_rule_parameters['K'], self.update_rule_space.neat_config, self.config.max_potential, is_soft_clip=False, device=self.device)
        self.add_module('ca_step', ca_step)

        # push the nn.Module and the available devoce
        self.to(self.device)

        self.generate_update_rule_kernels()
        self.generate_init_potential()


    def generate_update_rule_kernels(self):
        self.ca_step.reset_kernels()


    def generate_init_potential(self):
        init_potential = self.ca_init(output_size=(self.config.SX, self.config.SY, self.config.SZ, self.n_blocks)).unsqueeze(0)
        init_potential[:, :, :, :, 0] = self.config.air_potential
        self.potential = init_potential
        #sparse_mask = torch.nn.functional.gumbel_softmax(torch.log(self.potential.detach()), tau=0.01, hard=True).argmax(-1) > 0 #todo: dont detach?
        #self.sparse_potential = ME.to_sparse(sparse_mask.unsqueeze(-1).repeat(1, 1, 1, 1,self.n_blocks) * self.potential, format="BXXXC")
        self.step_idx = 0


    def update_initialization_parameters(self):
        new_initialization_parameters = deepcopy(self.initialization_parameters)
        for str_key, module in self.ca_init.I.named_children():
            new_initialization_parameters['I'][str_key]['cppn_genome'] = module.cppn_net.update_genome(self.initialization_parameters['I'][str_key]['cppn_genome'])
        if not self.initialization_space.contains(new_initialization_parameters):
            new_initialization_parameters = self.initialization_space.clamp(new_initialization_parameters)
            warnings.warn('provided parameters are not in the space range and are therefore clamped')
        self._initialization_parameters = new_initialization_parameters


    def update_update_rule_parameters(self):
        new_update_rule_parameters = deepcopy(self.update_rule_parameters)
        new_update_rule_parameters['T'] = self.ca_step.T.data
        for str_key, module in self.ca_step.K.named_children():
            new_update_rule_parameters['K'][str_key]['cppn_genome'] = module.cppn_net.update_genome(self.update_rule_parameters['K'][str_key]['cppn_genome'])
            new_update_rule_parameters['K'][str_key]['m'] = module.m.data
            new_update_rule_parameters['K'][str_key]['s'] = module.s.data
            new_update_rule_parameters['K'][str_key]['h'] = module.h.data
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

        # param_list = [] #[{'params': self.ca_step.T, 'lr': 0.01}]
        # for str_key, K in self.ca_step.K.named_children():
        #     param_list.append({'params': K.m, 'lr': 0.01})
        #     param_list.append({'params': K.s, 'lr': 0.01})
        #     param_list.append({'params': K.h, 'lr': 0.01})

        # self.optimizer = torch.optim.Adam([{'params': self.ca_init.I.parameters(), 'lr': 0.01},
        #                                     {'params': self.ca_step.K.parameters(), 'lr': 0.01},
        #                                     {'params': self.ca_step.T, 'lr': 0.01}])

        #self.optimizer = torch.optim.Adam(param_list)

        self.optimizer = torch.optim.Adam([{'params': self.ca_init.I.parameters(), 'lr': 0.1},
                          {'params': self.ca_step.K.parameters(), 'lr': 0.1},
                          {'params': self.ca_step.T, 'lr': 0.1}
                          ])

        for optim_step_idx in range(optim_steps):

            # run system
            for str_key, K in self.update_rule_parameters.K.items():
                c0, c1 = str_to_tuple_key(str_key)
            observations = self.run()
            if self.is_dead:
                break

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
        self.update_initialization_parameters()
        self.update_update_rule_parameters()

        return train_losses

    def step(self, intervention_parameters=None):
        # clamp params if was changed outside of allowed bounds with gradient Descent
        with torch.no_grad():
            if not self.update_rule_space.T.contains(self.ca_step.T.data):
                self.ca_step.T.data = self.update_rule_space.T.clamp(self.ca_step.T.data)
            for str_key, module in self.ca_step.K.named_children():
                if not self.update_rule_space.K['m'].contains(module.m.data):
                    module.m.data = self.update_rule_space.K['m'].clamp(module.m.data)
                if not self.update_rule_space.K['s'].contains(module.s.data):
                    module.s.data = self.update_rule_space.K['s'].clamp(module.s.data)
                if not self.update_rule_space.K['h'].contains(module.h.data):
                    module.h.data = self.update_rule_space.K['h'].clamp(module.h.data)
        self.potential = self.ca_step(self.potential)
        #sparse_mask = torch.nn.functional.gumbel_softmax(torch.log(self.potential.detach()), tau=0.01, hard=True).argmax(-1) > 0
        #self.sparse_potential = ME.to_sparse(sparse_mask.unsqueeze(-1).repeat(1, 1, 1, 1, self.n_blocks) * self.potential, format="BXXXC")
        self.step_idx += 1

        self.is_dead = self.potential.detach().argmax(-1).sum() == 0

        return self.potential


    def forward(self):
        potential = self.step(None)
        return potential


    def run(self, render=False, render_mode="human"):
        self.generate_update_rule_kernels()
        self.generate_init_potential()
        observations = Dict()
        observations.timepoints = list(range(self.config.final_step))
        observations.potentials = torch.empty((self.config.final_step, self.config.SX, self.config.SY, self.config.SZ, self.n_blocks))
        observations.potentials[0] = self.potential[0]
        # sparse_potentials_coords = [self.sparse_potential.C]
        # sparse_potentials_feats = [self.sparse_potential.F]
        for step_idx in range(1, self.config.final_step):
            if render:
                self.render(mode=render_mode)
            _ = self.step(None)
            observations.potentials[step_idx] = self.potential[0]
            if self.is_dead:
                break
            # sparse_potentials_coords.append(self.sparse_potential.C)
            # sparse_potentials_feats.append(self.sparse_potential.F)

        return observations


    def render(self, mode="human"):
        vis = o3d.visualization.Visualizer()
        vis.create_window('Discovery',800,800)
        pcd = o3d.geometry.PointCloud()
        cur_potential = self.potential[0].cpu().detach()
        coords = []
        feats = []
        offset = torch.tensor([self.config.SX/2.,self.config.SY/2.,self.config.SZ/2.])
        for i in range(self.config.SX):
            for j in range(self.config.SY):
                for k in range(self.config.SZ):
                    block_id = cur_potential[i, j, k].cpu().detach().argmax()
                    if block_id > 0:
                        coords.append(torch.tensor([i, j, k], dtype=torch.float64) - offset)
                        feats.append(block_id.unsqueeze(-1))
        if len(coords) > 0:
            coords = torch.stack(coords)
            feats = torch.stack(feats)
            pcd.points = o3d.utility.Vector3dVector(coords)
            pcd.colors = o3d.utility.Vector3dVector(torch.stack([torch.tensor(self.blocks_colorlist[feats[i]][:3]) for i in range(len(feats))]))
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1.0)
            vis.add_geometry(voxel_grid)
        if mode == "human":
            p_cam = o3d.camera.PinholeCameraParameters()
            p_cam.intrinsic.set_intrinsics(width=800, height=800, fx=0, fy=0, cx=399.5, cy=399.5)
            R = get_rotation_matrix(pi/4.0, pi/4., 0)
            d = 1.2*max(self.config.SX, self.config.SY, self.config.SZ)
            t = torch.tensor([self.config.SX/8.0, 0, d])
            p_cam.extrinsic = torch.tensor([[R[0,0], R[0,1], R[0,2], t[0]],
                                            [R[1,0], R[1,1], R[1,2], t[1]],
                                            [R[2,0], R[2,1], R[2,2], t[2]],
                                            [0., 0., 0., 1.]])
            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(p_cam)
            #ctr.rotate(200,200)
            #ctr.set_zoom(1.4)
            vis.run()
            vis.close()
            return
        elif mode == "rgb_array":
            out_image = vis.capture_screen_float_buffer(True)
            return out_image
        else:
            raise NotImplementedError

    """
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
    """

    def close(self):
        pass


def get_rotation_matrix(theta_x, theta_y, theta_z):
    R_x = torch.tensor([[1., 0., 0.],
                        [0., cos(theta_x), -sin(theta_x)],
                        [0., sin(theta_x), cos(theta_x)]])
    R_y = torch.tensor([[cos(theta_y), 0., sin(theta_y)],
                        [0., 1.0, 0.0],
                        [-sin(theta_y), 0.0, cos(theta_y)]])
    R_z = torch.tensor([[cos(theta_z), -sin(theta_z), 0.0],
                        [sin(theta_z), cos(theta_z), 0.0],
                        [0., 0., 1.]])
    R = R_z.matmul(R_y.matmul(R_x))
    return R
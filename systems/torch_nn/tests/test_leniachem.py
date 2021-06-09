from unittest import TestCase
import torch
from exputils.seeding import set_seed
import neat
import pytorchneat
from evocraftsearch.systems import LeniaChem
from evocraftsearch.systems.torch_nn.leniachem import LeniaChemInitializationSpace, LeniaChemInitialization
from evocraftsearch.systems.torch_nn.leniachem import LeniaChemUpdateRuleSpace, LeniaChemStep
from math import floor
import matplotlib.pyplot as plt


class TestLeniaChem(TestCase):

    def test_initialisation(self):
        set_seed(0)

        torch.backends.cudnn.enabled = False
        ## Load System
        env_size = (32, 32, 1)
        n_channels = 10
        max_potential = 1.0
        device = "cuda"

        neat_config = neat.Config(pytorchneat.selfconnectiongenome.SelfConnectionGenome,
                                  neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet,
                                  neat.DefaultStagnation,
                                  'template_neat_cppn.cfg'
                                  )

        initialization_space = LeniaChemInitializationSpace(n_channels, neat_config)
        initialization_parameters = initialization_space.sample()
        ca_init = LeniaChemInitialization(initialization_parameters['I'], neat_config, max_potential=max_potential, device=device)

        output_size = env_size + (n_channels, )
        output_img = torch.zeros(output_size, device=device)
        for str_key, I in ca_init.I.named_children():
            c = int(str_key)
            # the cppn generated output is confined to a limited space:
            c_output_size = tuple([max(int(s * I.occupation_ratio), 1) for s in output_size[:-1]])
            c_cppn_input = pytorchneat.utils.create_image_cppn_input(c_output_size, is_distance_to_center=True, is_bias=True)
            c_cppn_output = (ca_init.max_potential - ca_init.max_potential * I.cppn_net.activate(c_cppn_input, 3).abs()).squeeze(-1)
            # plt.imshow(c_cppn_output[c_output_size[0]//2,:,:].cpu().detach(), cmap='gray', vmin=0.0, vmax=1.0)
            # plt.show()
            # plt.imshow(c_cppn_output[:, c_output_size[1] // 2, :].cpu().detach(), cmap='gray', vmin=0.0, vmax=1.0)
            # plt.show()
            plt.imshow(c_cppn_output[:, :, c_output_size[2] // 2].cpu().detach(), cmap='gray', vmin=0.0, vmax=1.0)
            plt.show()
            offset_X = floor((output_size[0] - c_cppn_output.shape[0]) / 2)
            offset_Y = floor((output_size[1] - c_cppn_output.shape[1]) / 2)
            offset_Z = floor((output_size[2] - c_cppn_output.shape[2]) / 2)
            output_img[offset_X:offset_X + c_cppn_output.shape[0], offset_Y:offset_Y + c_cppn_output.shape[1],
            offset_Z:offset_Z + c_cppn_output.shape[2], c] = c_cppn_output


        return

    def test_kernels(self):
        set_seed(0)
        torch.backends.cudnn.enabled = False
        ## Load System
        leniachem_config = LeniaChem.default_config()
        leniachem_config.SX = 16
        leniachem_config.SY = 16
        leniachem_config.SZ = 16
        leniachem_config.final_step = 4
        # leniachem_config.blocks_list = block_list

        neat_config = neat.Config(pytorchneat.selfconnectiongenome.SelfConnectionGenome,
                                  neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet,
                                  neat.DefaultStagnation,
                                  'template_neat_cppn.cfg'
                                  )
        update_rule_space = LeniaChemUpdateRuleSpace(len(leniachem_config.blocks_list), neat_config)

        for creature_idx in range(10):
            update_rule_parameters = update_rule_space.sample()
            ca_step = LeniaChemStep(update_rule_parameters['T'], update_rule_parameters['K'], neat_config, is_soft_clip=False, device='cuda')
            ca_step.reset_kernels()
            for str_key, K in ca_step.K.named_children():
                print(str_key)
                kernel = ca_step.kernels[str_key]
                plt.imshow(kernel.squeeze()[:, :, kernel.shape[-1] // 2].cpu().detach(), cmap='gray')
                plt.show()

        return



    def test_leniachem(self):
        # import grpc
        # from evocraftsearch.evocraft import minecraft_pb2_grpc
        # channel = grpc.insecure_channel('localhost:5001') #WORLD ORIGIN: (0,4,0)
        # client = minecraft_pb2_grpc.MinecraftServiceStub(channel)

        set_seed(0)
        torch.backends.cudnn.enabled = False
        ## Load System
        leniachem_config = LeniaChem.default_config()
        leniachem_config.SX = 32
        leniachem_config.SY = 32
        leniachem_config.SZ = 1
        leniachem_config.final_step = 100
        #leniachem_config.blocks_list = block_list

        neat_config = neat.Config(pytorchneat.selfconnectiongenome.SelfConnectionGenome,
                                                              neat.DefaultReproduction,
                                                              neat.DefaultSpeciesSet,
                                                              neat.DefaultStagnation,
                                                              'template_neat_cppn.cfg'
                                                              )
        initialization_space = LeniaChemInitializationSpace(len(leniachem_config.blocks_list), neat_config, occupation_ratio_range=[0.1,0.2])
        update_rule_space = LeniaChemUpdateRuleSpace(len(leniachem_config.blocks_list), 1, neat_config, RX_max=5, RY_max=5, RZ_max=1)
        system = LeniaChem(initialization_space=initialization_space, update_rule_space=update_rule_space, config=leniachem_config, device='cuda')
        for creature_idx in range(100):
            if creature_idx % 2 == 0:
                base_policy_parameters = system.sample_policy_parameters()
            policy_parameters = system.mutate_policy_parameters(base_policy_parameters)
            system.reset(policy=policy_parameters)
            observations = system.run(render=False)
            #system.render_traj_in_minecraft(observations.potentials, client, arena_bbox=((creature_idx % 10)*max(system.config.SX,16),4,(creature_idx // 10)*max(system.config.SZ,16),40,40,40), blocks_list=system.config.blocks_list)
            # system.render_slices_gif(observations.potentials[0], f"creature_{creature_idx}_start.gif", slice_along="z")
            # system.render_slices_gif(observations.potentials[-1], f"creature_{creature_idx}_end.gif", slice_along="z")
            system.render_rollout(observations, f"creature_{creature_idx}_traj")

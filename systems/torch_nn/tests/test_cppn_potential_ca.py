from unittest import TestCase
import torch
from exputils.seeding import set_seed
import neat
import pytorchneat
from evocraftsearch.systems import CppnPotentialCA
from evocraftsearch.systems.torch_nn.cppn_potential_CA import CppnPotentialCAInitializationSpace, CppnPotentialCAInitialization
from evocraftsearch.systems.torch_nn.cppn_potential_CA import CppnPotentialCAUpdateRuleSpace, CppnPotentialCAStep




class TestCppnPotentialCA(TestCase):

    def test_initialisation(self):
        from math import floor
        import matplotlib.pyplot as plt

        set_seed(0)

        torch.backends.cudnn.enabled = False
        ## Load System
        env_size = (16,16,16)
        n_channels = 10
        max_potential = 1.0
        device = "cuda"

        neat_config = neat.Config(pytorchneat.selfconnectiongenome.SelfConnectionGenome,
                                  neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet,
                                  neat.DefaultStagnation,
                                  'template_neat_cppn.cfg'
                                  )

        initialization_space = CppnPotentialCAInitializationSpace(n_channels, neat_config)
        initialization_parameters = initialization_space.sample()
        ca_init = CppnPotentialCAInitialization(initialization_parameters['I'], neat_config, max_potential=max_potential, device=device)

        output_size = env_size + (n_channels, )
        output_img = torch.zeros(output_size, device=device)
        for str_key, I in ca_init.I.named_children():
            c = int(str_key)
            # the cppn generated output is confined to a limited space:
            c_output_size = tuple([int(s * I.occupation_ratio) for s in output_size[:-1]])
            print(c_output_size)
            c_cppn_input = pytorchneat.utils.create_image_cppn_input(c_output_size, is_distance_to_center=True, is_bias=True)
            c_cppn_output = (ca_init.max_potential - ca_init.max_potential * I.cppn_net.activate(c_cppn_input, 3).abs()).squeeze(-1)
            for i in range(3):
                plt.imshow(c_cppn_output[c_output_size[i]//2,:,:])
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
        cppn_potential_ca_config = CppnPotentialCA.default_config()
        cppn_potential_ca_config.SX = 16
        cppn_potential_ca_config.SY = 16
        cppn_potential_ca_config.SZ = 16
        cppn_potential_ca_config.final_step = 4
        # cppn_potential_ca_config.blocks_list = block_list

        neat_config = neat.Config(pytorchneat.selfconnectiongenome.SelfConnectionGenome,
                                  neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet,
                                  neat.DefaultStagnation,
                                  'template_neat_cppn.cfg'
                                  )
        update_rule_space = CppnPotentialCAUpdateRuleSpace(len(cppn_potential_ca_config.blocks_list), neat_config)
        update_rule_parameters = update_rule_space.sample()
        ca_step = CppnPotentialCAStep(update_rule_parameters['T'], update_rule_parameters['K'], neat_config, is_soft_clip=False, device='cuda')

        return



    def test_cppn_potential_ca(self):
        # import grpc
        # from evocraftsearch.evocraft import minecraft_pb2_grpc
        # channel = grpc.insecure_channel('localhost:5001') #WORLD ORIGIN: (0,4,0)
        # client = minecraft_pb2_grpc.MinecraftServiceStub(channel)

        set_seed(0)
        torch.backends.cudnn.enabled = False
        ## Load System
        cppn_potential_ca_config = CppnPotentialCA.default_config()
        cppn_potential_ca_config.SX = 32
        cppn_potential_ca_config.SY = 32
        cppn_potential_ca_config.SZ = 4
        cppn_potential_ca_config.final_step = 40
        #cppn_potential_ca_config.blocks_list = block_list

        neat_config = neat.Config(pytorchneat.selfconnectiongenome.SelfConnectionGenome,
                                                              neat.DefaultReproduction,
                                                              neat.DefaultSpeciesSet,
                                                              neat.DefaultStagnation,
                                                              'template_neat_cppn.cfg'
                                                              )
        initialization_space = CppnPotentialCAInitializationSpace(len(cppn_potential_ca_config.blocks_list), neat_config)
        update_rule_space = CppnPotentialCAUpdateRuleSpace(len(cppn_potential_ca_config.blocks_list), neat_config)
        system = CppnPotentialCA(initialization_space=initialization_space, update_rule_space=update_rule_space, config=cppn_potential_ca_config, device='cuda')
        for creature_idx in range(40):
            if creature_idx % 5 == 0:
                base_policy_parameters = system.sample_policy_parameters()
            policy_parameters = system.mutate_policy_parameters(base_policy_parameters)
            system.reset(policy=policy_parameters)
            observations = system.run(render=False)
            #system.render_traj_in_minecraft(observations.potentials, client, arena_bbox=((creature_idx % 10)*max(system.config.SX,16),4,(creature_idx // 10)*max(system.config.SZ,16),40,40,40), blocks_list=system.config.blocks_list)
            system.render_slices_gif(observations.potentials[0], f"creature_{creature_idx}_start.gif", blocks_colorlist=system.blocks_colorlist, slice_along="z")
            system.render_slices_gif(observations.potentials[-1], f"creature_{creature_idx}_end.gif", blocks_colorlist=system.blocks_colorlist, slice_along="z")
            system.render_rollout(observations, f"creature_{creature_idx}_traj")

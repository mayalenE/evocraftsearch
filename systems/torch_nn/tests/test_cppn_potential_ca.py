from unittest import TestCase
import torch.optim
from exputils.seeding import set_seed
import neat
import pytorchneat
from evocraftsearch.systems import CppnPotentialCA
from evocraftsearch.systems.torch_nn.cppn_potential_CA import CppnPotentialCAInitializationSpace, CppnPotentialCAUpdateRuleSpace

class TestCppnPotentialCA(TestCase):

    def test_cppn_potential_ca(self):
        set_seed(4)
        torch.backends.cudnn.enabled = False
        ## Load System
        cppn_potential_ca_config = CppnPotentialCA.default_config()
        cppn_potential_ca_config.SX = 16
        cppn_potential_ca_config.SY = 16
        cppn_potential_ca_config.SZ = 16
        cppn_potential_ca_config.final_step = 40
        #cppn_potential_ca_config.block_list = block_list

        neat_config = neat.Config(pytorchneat.selfconnectiongenome.SelfConnectionGenome,
                                                              neat.DefaultReproduction,
                                                              neat.DefaultSpeciesSet,
                                                              neat.DefaultStagnation,
                                                              'template_neat_cppn.cfg'
                                                              )
        initialization_space = CppnPotentialCAInitializationSpace(len(cppn_potential_ca_config.block_list), neat_config)
        update_rule_space = CppnPotentialCAUpdateRuleSpace(len(cppn_potential_ca_config.block_list), neat_config)
        system = CppnPotentialCA(initialization_space=initialization_space, update_rule_space=update_rule_space, config=cppn_potential_ca_config, device='cuda')
        desired_block_id = 1
        for creature_idx in range(50):
            print('----------------------------------------')
            policy_parameters = system.sample_policy_parameters()
            system.reset(policy = policy_parameters)
            optimizer = torch.optim.Adam([{'params': system.ca_init.I.parameters(), 'lr': 0.1},
                                          {'params': system.ca_step.K.parameters(), 'lr': 0.1},
                                          {'params': system.ca_step.T, 'lr': 0.1}
                                         ])

            for optim_idx in range(1):
                observations = system.run(render=False)
                if system.is_dead:
                    break
                else:
                    system.render_gif(observations.potentials, gif_filepath=f'creature_{creature_idx}.gif')
                # if ((observations.potentials[-1].argmax(-1)==desired_block_id).sum() == 0):
                #     break
                loss = torch.nn.functional.cross_entropy(observations.potentials[-1].view(-1, system.n_blocks), desired_block_id*torch.ones(cppn_potential_ca_config.SX*cppn_potential_ca_config.SY*cppn_potential_ca_config.SZ, dtype=torch.long))
                optimizer.zero_grad()
                loss.backward()
                # print(loss, (system.potential[0,:,:,:,:].argmax(-1)==desired_block_id).sum())
                # if optim_idx == 0:
                #     for n,p in system.named_parameters():
                #         if p.grad is not None:
                #             print(n,p.grad.sum())
                optimizer.step()

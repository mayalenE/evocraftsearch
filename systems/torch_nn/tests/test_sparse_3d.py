from unittest import TestCase

import torch.optim
from addict import Dict
import neat
import pytorchneat
from evocraftsearch.systems import SparseCA
from evocraftsearch.systems.minkowski_nn.sparse3D import SparseCAInitializationSpace, SparseCAUpdateRuleSpace

class TestLenia(TestCase):

    def test_sparse_ca(self):
        # Load System: here sparse_ca
        sparse_ca_config = SparseCA.default_config()
        sparse_ca_config.SX = 16
        sparse_ca_config.SY = 16
        sparse_ca_config.SZ = 16
        sparse_ca_config.n_blocks = 10
        sparse_ca_config.final_step = 10

        initialization_space_config = Dict()
        initialization_space_config.neat_config = neat.Config(pytorchneat.selfconnectiongenome.SelfConnectionGenome,
                                                          neat.DefaultReproduction,
                                                          neat.DefaultSpeciesSet,
                                                          neat.DefaultStagnation,
                                                          '/home/mayalen/code/my_packages/evocraftsearch/systems/minkowski_nn/tests/test_neat_sparse_ca_input.cfg'
                                                          )
        initialization_space = SparseCAInitializationSpace(config=initialization_space_config)

        update_rule_space_config = Dict()
        update_rule_space_config.neat_config = neat.Config(pytorchneat.selfconnectiongenome.SelfConnectionGenome,
                                                          neat.DefaultReproduction,
                                                          neat.DefaultSpeciesSet,
                                                          neat.DefaultStagnation,
                                                          '/home/mayalen/code/my_packages/evocraftsearch/systems/minkowski_nn/tests/test_neat_sparse_ca_kernels.cfg'
                                                          )
        update_rule_space = SparseCAUpdateRuleSpace(config=update_rule_space_config)

        system = SparseCA(initialization_space=initialization_space, update_rule_space=update_rule_space, config=sparse_ca_config)
        desired_block_id = 1
        for creature_idx in range(50):
            print('----------------------------------------')
            initialization_parameters = system.initialization_space.sample()
            update_rule_parameters = system.update_rule_space.sample()
            system.reset(initialization_parameters=initialization_parameters,
                        update_rule_parameters=update_rule_parameters)
            optimizer = torch.optim.Adam([{'params': system.initialization_cppn.parameters(), 'lr': 0.001},
                                          {'params': system.sparse_ca_step.K.parameters(), 'lr': 0.01},
                                          {'params': system.sparse_ca_step.T, 'lr': 0.1}
                                         ])
            for optim_idx in range(20):
                observations = system.run()
                #system.render_video(observations.states)
                if ((system.state[0,:,:,:,desired_block_id]==1).sum() == 0):
                    break;
                loss = torch.nn.functional.cross_entropy(system.potential[0].view(-1, 10), desired_block_id*torch.ones(16*16*16, dtype=torch.long))
                optimizer.zero_grad()
                loss.backward()
                print(loss, (system.state[0,:,:,:,desired_block_id]==1).sum())
                # for n,p in system.sparse_ca_step.kernel_cppn_nets.named_parameters():
                #     if p.grad is not None:
                #         print(n,p.grad.sum())
                #system.render()
                optimizer.step()

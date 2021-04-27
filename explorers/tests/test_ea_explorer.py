from unittest import TestCase
import neat
import pytorchneat
from evocraftsearch.systems import CppnPotentialCA
from evocraftsearch.systems.torch_nn.cppn_potential_CA import CppnPotentialCAInitializationSpace, CppnPotentialCAUpdateRuleSpace
from evocraftsearch.output_fitness.torch_nn.reconstruction import ReconstructionFitness
from evocraftsearch import ExplorationDB
from evocraftsearch.evocraft.utils import load_target
from evocraftsearch.explorers import EAExplorer
from exputils.seeding import set_seed
import torch


class TestEAExplorer(TestCase):
    def test_ea_explorer(self):
        set_seed(0)
        torch.backends.cudnn.enabled = False  # Somehow cudnn decrease performances in our case :O

        # Load Target
        target, block_list = load_target("/home/mayalen/code/08-EvoCraft/structures/desert_temple.nbt")
        pad = 1
        max_size = max(target.shape)
        pad_x = pad + (max_size - target.shape[0] + 1) // 2
        pad_y = pad + (max_size - target.shape[1] + 1) // 2
        pad_z = pad + (max_size - target.shape[2] + 1) // 2
        SX = target.shape[0] + 2 * pad_x
        SY = target.shape[1] + 2 * pad_y
        SZ = target.shape[2] + 2 * pad_z
        print(SX,SY,SZ)
        n_blocks = target.shape[-1]
        air_one_hot = torch.nn.functional.one_hot(torch.tensor(0), n_blocks).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        target = torch.cat([target, air_one_hot.repeat(pad_x, target.shape[1], target.shape[2], 1)], dim=0)
        target = torch.cat([target, air_one_hot.repeat(target.shape[0], pad_y, target.shape[2], 1)], dim=1)
        target = torch.cat([target, air_one_hot.repeat(target.shape[0], target.shape[1], pad_z, 1)], dim=2)
        target = torch.cat([air_one_hot.repeat(pad_x, target.shape[1], target.shape[2], 1), target], dim=0)
        target = torch.cat([air_one_hot.repeat(target.shape[0], pad_y, target.shape[2], 1), target], dim=1)
        target = torch.cat([air_one_hot.repeat(target.shape[0], target.shape[1], pad_z, 1), target], dim=2)

        # Load System
        cppn_potential_ca_config = CppnPotentialCA.default_config()
        cppn_potential_ca_config.SX = SX
        cppn_potential_ca_config.SY = SY
        cppn_potential_ca_config.SZ = SZ
        cppn_potential_ca_config.final_step = 20
        cppn_potential_ca_config.blocks_list = block_list

        neat_config = neat.Config(pytorchneat.selfconnectiongenome.SelfConnectionGenome,
                                  neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet,
                                  neat.DefaultStagnation,
                                  'template_neat_cppn.cfg'
                                  )
        initialization_space = CppnPotentialCAInitializationSpace(len(cppn_potential_ca_config.blocks_list), neat_config)
        update_rule_space = CppnPotentialCAUpdateRuleSpace(len(cppn_potential_ca_config.blocks_list), neat_config)
        system = CppnPotentialCA(initialization_space=initialization_space, update_rule_space=update_rule_space,
                                 config=cppn_potential_ca_config, device='cuda')
        system.potential = target.unsqueeze(0)
        system.render()

        # Load ExplorationDB
        db_config = ExplorationDB.default_config()
        db_config.db_directory = '.'
        db_config.save_observations = True
        db_config.load_observations = True
        exploration_db = ExplorationDB(config=db_config)

        # Load EA Explorer
        output_fitness = ReconstructionFitness(target=target)

        explorer_config = EAExplorer.default_config()
        explorer_config.population_size = 20
        explorer_config.tournament_size = 5
        explorer_config.fitness_optim_steps = 20
        explorer = EAExplorer(system, exploration_db, output_fitness, config=explorer_config)

        # Run EA Explorer
        explorer.run(10)

        # # save
        # explorer.save('explorer.pickle')
        #
        # # restart from checkpoint
        # explorer = EAExplorer.load('explorer.pickle', load_data=False, map_location='cpu')
        # explorer.db = ExplorationDB(config=db_config)
        # explorer.db.load(map_location='cpu')
        # explorer.run(20, continue_existing_run=True)


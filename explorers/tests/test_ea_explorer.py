from unittest import TestCase
from addict import Dict
import neat
import pytorchneat
from evocraftsearch.systems import CppnPotentialCA
from evocraftsearch.systems.torch_nn.cppn_potential_CA import CppnPotentialCAInitializationSpace, CppnPotentialCAUpdateRuleSpace
from evocraftsearch.output_fitness.torch_nn.reconstruction import ReconstructionFitness
from evocraftsearch import ExplorationDB
from evocraftsearch.explorers import EAExplorer
import os
from nbt import nbt
import torch

def load_target(nbt_filepath):
    assert os.path.isfile(nbt_filepath) and '.nbt' in nbt_filepath
    nbtfile = nbt.NBTFile(nbt_filepath, 'rb')

    min_x, max_x = (nbtfile['blocks'][0]['pos'][0].value, nbtfile['blocks'][0]['pos'][0].value)
    min_y, max_y = (nbtfile['blocks'][0]['pos'][1].value, nbtfile['blocks'][0]['pos'][1].value)
    min_z, max_z = (nbtfile['blocks'][0]['pos'][2].value, nbtfile['blocks'][0]['pos'][2].value)

    block_types = [block_type['Name'].value.split(":")[-1].upper() for block_type in nbtfile['palette']]
    air_idx = block_types.index("AIR")  #force air to be the first channel

    target_shape = (nbtfile['size'][0].value, nbtfile['size'][1].value, nbtfile['size'][2].value, len(block_types))
    target = torch.empty(target_shape)

    for block in nbtfile['blocks']:
        cur_x, cur_y, cur_z = block['pos'][0].value, block['pos'][1].value, block['pos'][2].value
        min_x = min(min_x, cur_x)
        max_x = max(max_x, cur_x)
        min_y = min(min_y, cur_y)
        max_y = max(max_y, cur_y)
        min_z = min(min_z, cur_z)
        max_z = max(max_z, cur_z)

        cur_block_type = block['state'].value
        if cur_block_type == 0:
            cur_block_type = air_idx
        elif cur_block_type == air_idx:
            cur_block_type = 0
        target[cur_x, cur_y, cur_z] = torch.nn.functional.one_hot(torch.tensor(cur_block_type), len(block_types))

    return target


class TestEAExplorer(TestCase):
    def test_ea_explorer(self):

        # Load Target
        target = load_target("/home/mayalen/code/08-EvoCraft/structures/Extra_dark_oak.nbt")

        # Load System
        cppn_potential_ca_config = CppnPotentialCA.default_config()
        cppn_potential_ca_config.SX = target.shape[0]
        cppn_potential_ca_config.SY = target.shape[1]
        cppn_potential_ca_config.SZ = target.shape[2]
        cppn_potential_ca_config.n_blocks = target.shape[-1]
        cppn_potential_ca_config.final_step = 10

        initialization_space_config = Dict()
        initialization_space_config.neat_config = neat.Config(pytorchneat.selfconnectiongenome.SelfConnectionGenome,
                                                              neat.DefaultReproduction,
                                                              neat.DefaultSpeciesSet,
                                                              neat.DefaultStagnation,
                                                              '/home/mayalen/code/my_packages/evocraftsearch/systems/torch_nn/tests/test_neat_cppn_potential_ca_input.cfg'
                                                              )
        initialization_space = CppnPotentialCAInitializationSpace(config=initialization_space_config)

        update_rule_space_config = Dict()
        update_rule_space_config.neat_config = neat.Config(pytorchneat.selfconnectiongenome.SelfConnectionGenome,
                                                           neat.DefaultReproduction,
                                                           neat.DefaultSpeciesSet,
                                                           neat.DefaultStagnation,
                                                           '/home/mayalen/code/my_packages/evocraftsearch/systems/torch_nn/tests/test_neat_cppn_potential_ca_kernels.cfg'
                                                           )
        update_rule_space = CppnPotentialCAUpdateRuleSpace(n_blocks=cppn_potential_ca_config.n_blocks, config=update_rule_space_config)

        system = CppnPotentialCA(initialization_space=initialization_space, update_rule_space=update_rule_space,
                                 config=cppn_potential_ca_config, device='cuda')

        # Load ExplorationDB
        db_config = ExplorationDB.default_config()
        db_config.db_directory = '.'
        db_config.save_observations = True
        db_config.load_observations = True
        exploration_db = ExplorationDB(config=db_config)

        # Load EA Explorer
        output_fitness = ReconstructionFitness(target=target)

        explorer_config = EAExplorer.default_config()
        explorer_config.population_size = 10
        explorer_config.selection_size = 5
        explorer_config.fitness_optimizer = Dict()
        explorer_config.fitness_optimizer.optim_steps = 20
        explorer_config.fitness_optimizer.name = "Adam"
        explorer_config.fitness_optimizer.initialization_cppn.parameters.lr = 1e-2
        explorer_config.fitness_optimizer.lenia_step.parameters.lr = 1e-3
        explorer = EAExplorer(system, exploration_db, output_fitness, config=explorer_config)

        # Run Imgep Explorer
        explorer.run(10)

        # # save
        # explorer.save('explorer.pickle')
        #
        # # restart from checkpoint
        # explorer = EAExplorer.load('explorer.pickle', load_data=False, map_location='cpu')
        # explorer.db = ExplorationDB(config=db_config)
        # explorer.db.load(map_location='cpu')
        # explorer.run(20, continue_existing_run=True)


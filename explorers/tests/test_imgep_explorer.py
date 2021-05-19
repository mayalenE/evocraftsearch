from unittest import TestCase
import torch
import neat
import pytorchneat
from evocraftsearch.systems import CppnPotentialCA
from evocraftsearch.systems.torch_nn.cppn_potential_CA import CppnPotentialCAInitializationSpace, CppnPotentialCAUpdateRuleSpace
from evocraftsearch.output_representation import ImageStatisticsRepresentation
from evocraftsearch import ExplorationDB
from evocraftsearch.explorers import IMGEPExplorer, BoxGoalSpace
from exputils.seeding import set_seed


class TestIMGEPExplorer(TestCase):
    def test_imgep_explorer(self):
        set_seed(0)
        torch.backends.cudnn.enabled = False
        ## Load System
        cppn_potential_ca_config = CppnPotentialCA.default_config()
        cppn_potential_ca_config.SX = 32
        cppn_potential_ca_config.SY = 32
        cppn_potential_ca_config.SZ = 1
        cppn_potential_ca_config.final_step = 100

        neat_config = neat.Config(pytorchneat.selfconnectiongenome.SelfConnectionGenome,
                                  neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet,
                                  neat.DefaultStagnation,
                                  'template_neat_cppn.cfg'
                                  )
        initialization_space = CppnPotentialCAInitializationSpace(len(cppn_potential_ca_config.blocks_list),
                                                                  neat_config, occupation_ratio_range=[0.1, 0.2])
        update_rule_space = CppnPotentialCAUpdateRuleSpace(len(cppn_potential_ca_config.blocks_list), 1, neat_config,
                                                           RX_max=5, RY_max=5, RZ_max=1)
        system = CppnPotentialCA(initialization_space=initialization_space, update_rule_space=update_rule_space,
                                 config=cppn_potential_ca_config, device='cuda')


        # Load ExplorationDB
        db_config = ExplorationDB.default_config()
        db_config.db_directory = '.'
        db_config.save_observations = True
        db_config.save_rollout_render = True
        db_config.load_observations = True
        exploration_db = ExplorationDB(config=db_config)

        # Load Imgep Explorer
        ## Load Goal Space Representation
        output_representation_config = ImageStatisticsRepresentation.default_config()
        output_representation_config.env_size = (system.config.SX, system.config.SY, system.config.SZ)
        output_representation_config.channel_list = list(range(1, system.n_blocks))
        output_representation_config.device = "cuda"
        output_representation = ImageStatisticsRepresentation(config=output_representation_config)
        goal_space = BoxGoalSpace(output_representation)

        ## Load imgep explorer
        explorer_config = IMGEPExplorer.default_config()
        explorer_config.num_of_random_initialization = 20
        explorer_config.reach_goal_optim_steps = 20
        explorer = IMGEPExplorer(system, exploration_db, goal_space, config=explorer_config)

        # Run Imgep Explorer
        # explorer.run(50)

        # save
        explorer.save('explorer.pickle')

        # restart from checkpoint
        explorer = IMGEPExplorer.load('explorer.pickle', load_data=False, map_location='cpu')
        explorer.db = ExplorationDB(config=db_config)
        explorer.db.load(map_location='cpu')
        explorer.run(50, continue_existing_run=True)


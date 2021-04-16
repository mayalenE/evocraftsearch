from unittest import TestCase
from addict import Dict
import neat
import pytorchneat
from evocraftsearch.systems import Lenia
from evocraftsearch.systems.lenia import LeniaInitializationSpace, LeniaUpdateRuleSpace
from evocraftsearch.output_representation import LeniaHandDefinedRepresentation, LeniaImageRepresentation
from evocraftsearch import ExplorationDB
from evocraftsearch.explorers import IMGEPExplorer, BoxGoalSpace

class TestIMGEPExplorer(TestCase):
    def test_imgep_explorer(self):

        # Load System: here lenia
        lenia_config = Lenia.default_config()
        lenia_config.SX = 256
        lenia_config.SY = 256
        lenia_config.final_step = 40
        lenia_config.version = 'pytorch_fft'

        initialization_space_config = Dict()
        initialization_space_config.cppn_n_passes = 4
        initialization_space_config.neat_config = neat.Config(pytorchneat.selfconnectiongenome.SelfConnectionGenome,
                                                              neat.DefaultReproduction,
                                                              neat.DefaultSpeciesSet,
                                                              neat.DefaultStagnation,
                                                              '/home/mayalen/code/my_packages/evocraftsearch/systems/tests/test_neat_lenia_input.cfg'
                                                              )
        initialization_space = LeniaInitializationSpace(config=initialization_space_config)

        system = Lenia(initialization_space=initialization_space, config=lenia_config, device='cuda')

        # Load ExplorationDB
        db_config = ExplorationDB.default_config()
        db_config.db_directory = '.'
        db_config.save_observations = True
        db_config.load_observations = True
        exploration_db = ExplorationDB(config=db_config)

        # Load Imgep Explorer
        ## Load Goal Space Representation
        output_representation_config = LeniaImageRepresentation.default_config()
        output_representation_config.env_size = (system.config.SX, system.config.SY)
        output_representation = LeniaImageRepresentation(config=output_representation_config)
        goal_space = BoxGoalSpace(output_representation)

        ## Load imgep explorer
        explorer_config = IMGEPExplorer.default_config()
        explorer_config.num_of_random_initialization = 4
        explorer_config.reach_goal_optimizer = Dict()
        explorer_config.reach_goal_optimizer.optim_steps = 100
        explorer_config.reach_goal_optimizer.name = "Adam"
        explorer_config.reach_goal_optimizer.initialization_cppn.parameters.lr = 1e-2
        explorer_config.reach_goal_optimizer.lenia_step.parameters.lr = 1e-3
        explorer = IMGEPExplorer(system, exploration_db, goal_space, config=explorer_config)

        # Run Imgep Explorer
        explorer.run(10)

        # # save
        # explorer.save('explorer.pickle')
        #
        # # restart from checkpoint
        # explorer = IMGEPExplorer.load('explorer.pickle', load_data=False, map_location='cpu')
        # explorer.db = ExplorationDB(config=db_config)
        # explorer.db.load(map_location='cpu')
        # explorer.run(20, continue_existing_run=True)


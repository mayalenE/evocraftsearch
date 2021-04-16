from unittest import TestCase
from addict import Dict
import neat
import pytorchneat
from evocraftsearch.systems import Lenia
from evocraftsearch.systems.lenia import LeniaInitializationSpace, LeniaUpdateRuleSpace
from evocraftsearch import ExplorationDB
from image_representation import VAE
from evocraftsearch.explorers import IMGEP_OGL_Explorer, TorchNNBoxGoalSpace


class TestIMGEP_OGL_Explorer(TestCase):
    def test_imgep_ogl_explorer(self):

        # Load System: here lenia
        lenia_config = Lenia.default_config()
        lenia_config.SX = 256
        lenia_config.SY = 256
        lenia_config.final_step = 100
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
        vae_config = VAE.default_config()
        vae_config.network.parameters.input_size = (system.config.SX, system.config.SY)
        vae_config.network.parameters.n_latents = 16
        vae_config.network.parameters.n_conv_layers = 4
        vae_config.network.parameters.feature_layer = 2
        vae_config.network.parameters.encoder_conditional_type = "gaussian"
        vae_config.network.weights_init.name = "pytorch"
        vae_config.loss.name = "VAE"
        vae_config.loss.parameters.reconstruction_dist = "bernoulli"
        vae_config.optimizer.name = "Adam"
        vae_config.optimizer.parameters.lr = 1e-3
        vae_config.optimizer.parameters.weight_decay = 1e-5
        vae_config.checkpoint.folder = "./training/checkpoints"
        vae_config.logging.folder = "./training/logs"
        vae_config.logging.record_loss_every = 1
        vae_config.logging.record_valid_images_every = 10
        vae_config.logging.record_embeddings_every = 10
        output_representation = VAE(config=vae_config)
        goal_space = TorchNNBoxGoalSpace(output_representation, autoexpand=True)

        ## Load imgep explorer
        explorer_config = IMGEP_OGL_Explorer.default_config()
        explorer_config.num_of_random_initialization = 30
        explorer_config.reach_goal_optimizer = Dict()
        explorer_config.reach_goal_optimizer.optim_steps = 50
        explorer_config.reach_goal_optimizer.name = "Adam"
        explorer_config.reach_goal_optimizer.initialization_cppn.parameters.lr = 1e-2
        explorer_config.reach_goal_optimizer.lenia_step.parameters.lr = 1e-3
        explorer_config.goalspace_training.dataset_augment = False
        explorer_config.goalspace_training.train_batch_size = 64
        explorer_config.goalspace_training.valid_batch_size = 32
        explorer_config.goalspace_training.frequency = 40
        explorer_config.goalspace_training.n_epochs = 40
        explorer_config.goalspace_training.importance_sampling_last = 0.3

        explorer = IMGEP_OGL_Explorer(system, exploration_db, goal_space, config=explorer_config)

        # Run Imgep Explorer
        explorer.run(200)

        # # save
        # explorer.save('explorer.pickle')
        #
        # # restart from checkpoint
        # explorer = IMGEP_OGL_Explorer.load('explorer.pickle', load_data=False, map_location='cpu')
        # explorer.db = ExplorationDB(config=db_config)
        # explorer.db.load(map_location='cpu')
        # explorer.run(45, continue_existing_run=True)


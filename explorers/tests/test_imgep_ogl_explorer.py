from unittest import TestCase
import torch
import neat
import pytorchneat
from evocraftsearch.systems import LeniaChem
from evocraftsearch.systems.torch_nn.leniachem import LeniaChemInitializationSpace, LeniaChemUpdateRuleSpace
from evocraftsearch import ExplorationDB
from exputils.seeding import set_seed
from image_representation import VAE
from evocraftsearch.explorers import IMGEP_OGL_Explorer, TorchNNBoxGoalSpace


class TestIMGEP_OGL_Explorer(TestCase):
    def test_imgep_ogl_explorer(self):
        set_seed(1)
        torch.backends.cudnn.enabled = False  # Somehow cudnn decrease performances in our case :O

        # Load System
        cppn_potential_ca_config = LeniaChem.default_config()
        cppn_potential_ca_config.SX = 16
        cppn_potential_ca_config.SY = 16
        cppn_potential_ca_config.SZ = 16
        cppn_potential_ca_config.final_step = 40

        neat_config = neat.Config(pytorchneat.selfconnectiongenome.SelfConnectionGenome,
                                  neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet,
                                  neat.DefaultStagnation,
                                  'template_neat_cppn.cfg'
                                  )
        initialization_space = LeniaChemInitializationSpace(len(cppn_potential_ca_config.blocks_list), neat_config)
        update_rule_space = LeniaChemUpdateRuleSpace(len(cppn_potential_ca_config.blocks_list), neat_config)
        system = LeniaChem(initialization_space=initialization_space, update_rule_space=update_rule_space,
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
        vae_config = VAE.default_config()
        vae_config.network.parameters.input_size = (system.config.SZ, system.config.SY, system.config.SX)
        vae_config.network.parameters.n_latents = 16
        vae_config.network.parameters.n_channels = system.n_blocks
        vae_config.network.parameters.n_conv_layers = 2
        vae_config.network.parameters.feature_layer = 1
        vae_config.network.parameters.encoder_conditional_type = "gaussian"
        vae_config.network.weights_init.name = "pytorch"
        vae_config.device = 'cuda'
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
        explorer_config.frequency_of_random_initialization = 10
        explorer_config.reach_goal_optim_steps = 20
        explorer_config.goalspace_training.dataset_augment = True
        explorer_config.goalspace_training.train_batch_size = 64
        explorer_config.goalspace_training.valid_batch_size = 32
        explorer_config.goalspace_training.frequency = 40
        explorer_config.goalspace_training.n_epochs = 40
        explorer_config.goalspace_training.importance_sampling_last = 0.3

        explorer = IMGEP_OGL_Explorer(system, exploration_db, goal_space, config=explorer_config)

        # Run Imgep Explorer
        explorer.run(10)

        # # save
        # explorer.save('explorer.pickle')
        #
        # # restart from checkpoint
        # explorer = IMGEP_OGL_Explorer.load('explorer.pickle', load_data=False, map_location='cpu')
        # explorer.db = ExplorationDB(config=db_config)
        # explorer.db.load(map_location='cpu')
        # explorer.run(45, continue_existing_run=True)


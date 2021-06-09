from unittest import TestCase
from addict import Dict
import torch
import neat
import pytorchneat
from evocraftsearch.systems import LeniaChem
from evocraftsearch.systems.torch_nn.leniachem import LeniaChemInitializationSpace, LeniaChemUpdateRuleSpace
from evocraftsearch import ExplorationDB
from exputils.seeding import set_seed
from image_representation import HOLMES_VAE
from evocraftsearch.explorers import IMGEP_HOLMES_Explorer, HolmesGoalSpace
import os
import shutil

class TestIMGEP_HOLMES_Explorer(TestCase):
    def test_imgep_holmes_explorer(self):
        set_seed(1)
        torch.backends.cudnn.enabled = False  # Somehow cudnn decrease performances in our case :O

        # Load System
        leniachem_config = LeniaChem.default_config()
        leniachem_config.SX = 16
        leniachem_config.SY = 16
        leniachem_config.SZ = 16
        leniachem_config.final_step = 40

        neat_config = neat.Config(pytorchneat.selfconnectiongenome.SelfConnectionGenome,
                                  neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet,
                                  neat.DefaultStagnation,
                                  'template_neat_cppn.cfg'
                                  )
        initialization_space = LeniaChemInitializationSpace(len(leniachem_config.blocks_list), neat_config)
        update_rule_space = LeniaChemUpdateRuleSpace(len(leniachem_config.blocks_list), neat_config)
        system = LeniaChem(initialization_space=initialization_space, update_rule_space=update_rule_space,
                                 config=leniachem_config, device='cuda')

        # Load ExplorationDB
        db_config = ExplorationDB.default_config()
        db_config.db_directory = '.'
        db_config.save_observations = True
        db_config.save_rollout_render = True
        db_config.load_observations = True
        exploration_db = ExplorationDB(config=db_config)

        # Load Imgep Explorer
        ## Load Goal Space Representation
        holmes_vae_config = HOLMES_VAE.default_config()
        holmes_vae_config.node.network.parameters.input_size = (16, 16, 16)
        holmes_vae_config.node.network.parameters.n_channels = 1
        holmes_vae_config.node.network.parameters.n_latents = 16
        holmes_vae_config.node.network.parameters.n_conv_layers = 2
        holmes_vae_config.node.network.parameters.feature_layer = 1
        holmes_vae_config.node.network.parameters.encoder_conditional_type = "gaussian"
        holmes_vae_config.node.network.weights_init.name = "pytorch"

        holmes_vae_config.device = "cuda"
        holmes_vae_config.dtype = torch.float32

        holmes_vae_config.node.create_connections = {"lf": True, "gf": False, "gfi": True, "lfi": True, "recon": True}

        holmes_vae_config.loss.name = "VAE"
        holmes_vae_config.loss.parameters.reconstruction_dist = "bernoulli"
        holmes_vae_config.optimizer.name = "Adam"
        holmes_vae_config.optimizer.parameters.lr = 1e-3
        holmes_vae_config.optimizer.parameters.weight_decay = 1e-5

        holmes_vae_config.checkpoint.folder = "./training/checkpoints/holmes_vae3d"
        holmes_vae_config.logging.folder = "./training/logs/holmes_vae3d"
        if os.path.exists(holmes_vae_config.logging.folder):
            shutil.rmtree(holmes_vae_config.logging.folder)
        holmes_vae_config.logging.record_loss_every = 1
        holmes_vae_config.logging.record_valid_images_every = 10
        holmes_vae_config.logging.record_embeddings_every = 100

        output_representation = HOLMES_VAE(config=holmes_vae_config)
        goal_space = HolmesGoalSpace(output_representation, autoexpand=True)

        training_config = Dict()
        training_config.n_epochs = 5000
        training_config.split_trigger.active = True
        training_config.split_trigger.fitness_key = 'recon'
        training_config.split_trigger.type = 'plateau'
        training_config.split_trigger.parameters = Dict(epsilon=20, n_steps_average=50)
        training_config.split_trigger.conditions = Dict(min_init_n_epochs=200, n_min_points=500, n_max_splits=10,
                                                        n_epochs_min_between_splits=100)
        training_config.split_trigger.save_model_before_after = True
        training_config.split_trigger.boundary_config.z_fitness = "recon_loss"
        training_config.split_trigger.boundary_config.algo = "cluster.KMeans"
        training_config.alternated_backward.active = True
        training_config.alternated_backward.ratio_epochs = {"connections": 2, "core": 8}

        ## Load imgep explorer
        explorer_config = IMGEP_HOLMES_Explorer.default_config()
        explorer_config.num_of_random_initialization = 100
        explorer_config.frequency_of_random_initialization = 10
        explorer_config.reach_goal_optim_steps = 20

        explorer_config.goalspace_preprocess = lambda x: x.argmax(1).unsqueeze(1).float() / system.n_blocks

        explorer_config.goalspace_training.dataset_config = Dict()
        explorer_config.goalspace_training.dataset_config.n_channels = 1
        explorer_config.goalspace_training.dataset_config.img_size = (system.config.SZ, system.config.SY, system.config.SX)
        explorer_config.goalspace_training.dataset_config.data_augmentation = True
        explorer_config.goalspace_training.dataset_config.preprocess = explorer_config.goalspace_preprocess

        explorer_config.goalspace_training.dataset_append_trajectory = False
        explorer_config.goalspace_training.train_batch_size = 64
        explorer_config.goalspace_training.valid_batch_size = 32
        explorer_config.goalspace_training.frequency = 100
        explorer_config.goalspace_training.n_epochs = 100
        explorer_config.goalspace_training.importance_sampling_last = 0.3


        explorer_config.goalspace_training.split_trigger.active = True
        explorer_config.goalspace_training.split_trigger.fitness_key = 'recon'
        explorer_config.goalspace_training.split_trigger.type = 'plateau'
        explorer_config.goalspace_training.split_trigger.parameters = Dict(epsilon=20, n_steps_average=50)
        explorer_config.goalspace_training.split_trigger.conditions = Dict(min_init_n_epochs=100, n_min_points=200, n_max_splits=10, n_epochs_min_between_splits=20)
        explorer_config.goalspace_training.split_trigger.save_model_before_after = True
        explorer_config.goalspace_training.split_trigger.boundary_config.z_fitness = "recon_loss"
        explorer_config.goalspace_training.split_trigger.boundary_config.algo = "cluster.KMeans"
        explorer_config.goalspace_training.alternated_backward.active = True
        explorer_config.goalspace_training.alternated_backward.ratio_epochs = {"connections": 2, "core": 8}

        explorer = IMGEP_HOLMES_Explorer(system, exploration_db, goal_space, config=explorer_config)

        # Run Imgep Explorer
        explorer.run(500)

        # # save
        # explorer.save('explorer.pickle')
        #
        # # restart from checkpoint
        # explorer = IMGEP_HOLMES_Explorer.load('explorer.pickle', load_data=False, map_location='cpu')
        # explorer.db = ExplorationDB(config=db_config)
        # explorer.db.load(map_location='cpu')
        # explorer.run(45, continue_existing_run=True)


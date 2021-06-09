import pytorchneat
import neat
import os
import torch
from evocraftsearch import ExplorationDB
from evocraftsearch.systems import LeniaChem
from evocraftsearch.systems.torch_nn.leniachem import LeniaChemInitializationSpace, LeniaChemUpdateRuleSpace
from image_representation import VAE, MEVAE, HOLMES_VAE
from evocraftsearch.output_representation import ImageStatisticsRepresentation, HistogramBlocksRepresentation
from evocraftsearch.explorers import IMGEPExplorer, BoxGoalSpace, IMGEP_OGL_Explorer, TorchNNBoxGoalSpace, IMGEP_HOLMES_Explorer, HolmesGoalSpace
from image_representation.datasets.preprocess import TensorRandomCentroidCrop, TensorRandomRoll, TensorRandomFlip, TensorRandomSphericalRotation
from torchvision.transforms import Compose
from copy import deepcopy

def get_seed():
    return int(0)

def get_system_config():
    system_config = LeniaChem.default_config()
    system_config.SX = int(16)
    system_config.SY = int(16)
    system_config.SZ = int(16)
    system_config.final_step = int(100)
    system_config.blocks_list = ["AIR", "RED_SANDSTONE", "LAPIS_BLOCK", "ORANGE_SHULKER_BOX", "WOOL", "PRISMARINE", "SNOW", "SLIME", "DIAMOND_BLOCK", "PURPUR_BLOCK", "STONE"]

    system_config.device = "cuda"

    return system_config

def get_initialization_space_config():

    neat_config = neat.Config(pytorchneat.selfconnectiongenome.SelfConnectionGenome,
                              neat.DefaultReproduction,
                              neat.DefaultSpeciesSet,
                              neat.DefaultStagnation,
                              'neat_config.cfg'
                              )

    initialization_space_config = LeniaChemInitializationSpace.default_config()

    # intra channel probabilities
    initialization_space_config.sample_channel = 0.8
    initialization_space_config.mutate_channel = 0.1

    # occupation ratio range
    initialization_space_config.occupation_ratio_low = float(1)
    initialization_space_config.occupation_ratio_high = float(1)
    initialization_space_config.occupation_ratio_mutation_std = 0.1
    initialization_space_config.occupation_ratio_mutation_indpb = 0.0

    return neat_config, initialization_space_config

def get_update_rule_space_config():

    n_kernels = int(1)

    update_rule_space_config = LeniaChemUpdateRuleSpace.default_config()

    # T
    update_rule_space_config.T_low = 1.0
    update_rule_space_config.T_high = 20.0
    update_rule_space_config.T_mutation_std = 1.0
    update_rule_space_config.T_mutation_indpb = 0.2
    # intra - kernel probabilities
    update_rule_space_config.sample_intra_channel = 1.0
    update_rule_space_config.mutate_intra_channel = 0.00
    # cross-kernel probabilities
    update_rule_space_config.sample_cross_channel = float(0.4)
    update_rule_space_config.mutate_cross_channel = 0.0
    # R
    update_rule_space_config.RX_max = int(5)
    update_rule_space_config.RY_max = int(5)
    update_rule_space_config.RZ_max = int(5)
    update_rule_space_config.R_mutation_std = 1
    update_rule_space_config.R_mutation_indpb = 0.2
    # m
    update_rule_space_config.m_low = 0.1
    update_rule_space_config.m_high = 0.6
    update_rule_space_config.m_mutation_std = 0.1
    update_rule_space_config.m_mutation_indpb = 0.2
    # s
    update_rule_space_config.s_low = 0.001
    update_rule_space_config.s_high = 0.2
    update_rule_space_config.s_mutation_std = 0.05
    update_rule_space_config.s_mutation_indpb = 0.2
    # h
    update_rule_space_config.h_low = 0.1
    update_rule_space_config.h_high = 0.9
    update_rule_space_config.h_mutation_std = 0.1
    update_rule_space_config.h_mutation_indpb = 0.2


    return n_kernels, update_rule_space_config

def get_exploration_db_config():
    db_config = ExplorationDB.default_config()
    db_config.db_directory = "data/exploration_db"
    db_config.save_observations = True
    db_config.load_observations = True
    db_config.save_rollout_render = True
    db_config.memory_size_run_data = 10
    return db_config

def get_representation_cls():
    representation_cls = eval("HOLMES_VAE")
    return representation_cls

def get_representation_config():
    system_config = get_system_config()
    representation_cls = get_representation_cls()

    if representation_cls.__name__ in ["ImageStatisticsRepresentation", "HistogramBlocksRepresentation"]:
        representation_config = representation_cls.default_config()
        representation_config.env_size = (system_config.SZ, system_config.SY, system_config.SX)
        representation_config.channel_list = list(range(0, len(system_config.blocks_list)))
        representation_config.device = "cuda"

    elif "VAE" in representation_cls.__name__ :
        representation_config = representation_cls.default_config()
        representation_config.network.name = "Dumoulin"
        representation_config.network.parameters = {"input_size": (16,16,16), "n_channels": 3, "n_latents": 16, "n_conv_layers": 2, "feature_layer": 1, "hidden_channels": 8, "hidden_dim": None,  "encoder_conditional_type": "gaussian"}
        representation_config.network.weights_init.name = "kaiming_normal"
        representation_config.device = "cuda"
        representation_config.loss.name = "VAE"
        representation_config.loss.parameters = {"reconstruction_dist": "bernoulli"}
        representation_config.optimizer.name = "Adam"
        representation_config.optimizer.parameters = {"lr": 1e-3, "weight_decay": 1e-5 }
        representation_config.checkpoint.folder = "data/training/checkpoints"
        representation_config.logging.folder = "data/training/logs"
        representation_config.logging.record_loss_every = 1
        representation_config.logging.record_valid_images_every = 100
        representation_config.logging.record_embeddings_every = 100

        if "HOLMES" in representation_cls.__name__:
            representation_config.node = deepcopy(representation_config)
            representation_config.node.create_connections = {"lf": True, "gf": False, "gfi":True, "lfi": True, "recon": True }

    else:
        raise NotImplementedError

    return representation_config

def get_goal_space_cls():
    representation_cls = "HOLMES_VAE"
    if "HOLMES" in representation_cls:
        goal_space_cls = HolmesGoalSpace

    elif "VAE" in representation_cls:
        goal_space_cls = TorchNNBoxGoalSpace
    else:
        goal_space_cls = BoxGoalSpace
    return goal_space_cls


def get_goal_space_config():
    goal_space_cls = get_goal_space_cls()
    goal_space_config = None

    if "Holmes" in goal_space_cls.__name__:
        goal_space_config = goal_space_cls.default_config()

    return goal_space_config

def get_explorer_cls():
    representation_cls = "HOLMES_VAE"

    if "HOLMES" in representation_cls:
        explorer_cls = IMGEP_HOLMES_Explorer
    elif "VAE" in representation_cls:
        explorer_cls = IMGEP_OGL_Explorer
    else:
        explorer_cls = IMGEPExplorer

    return explorer_cls

def get_explorer_config():
    system_config = get_system_config()

    explorer_cls = get_explorer_cls()
    explorer_config = explorer_cls.default_config()
    explorer_config.seed = int(0)
    explorer_config.num_of_random_initialization = 200
    explorer_config.frequency_of_random_initialization = 10
    explorer_config.reach_goal_optim_steps = int(0)

    if explorer_cls.__name__ in ["IMGEP_OGL_Explorer", "IMGEP_HOLMES_Explorer"]:

        explorer_config.goalspace_training.dataset_n_timepoints = int(1)
        explorer_config.goalspace_training.dataset_type = "rgb"

        explorer_config.goalspace_training.train_dataset.filepath = "data/training/train_dataset.h5"
        explorer_config.goalspace_training.train_dataset.config.load_data = False
        explorer_config.goalspace_training.train_dataset.config.data_cache_size = int(1) * int(400)

        if "rgb" == "rgb":
            n_channels = 3
        elif "rgb" == "onehot" or "rgb" == "potential":
            n_channels = len(system_config.blocks_list)

        obs_size = [n_channels, system_config.SZ, system_config.SY, system_config.SX]
        if 1 in obs_size:
            obs_size.remove(1)
        explorer_config.goalspace_training.train_dataset.config.obs_size = tuple(obs_size)
        explorer_config.goalspace_training.train_dataset.config.obs_dtype = "float32"

        # dataset augmentation
        if bool(1):
            spatial_dims = len(obs_size[1:])
            if spatial_dims == 3:
                random_center_crop = TensorRandomCentroidCrop(p=0.6, size=tuple(obs_size[1:]), scale=(0.5, 1.0), ratio_x=(1., 1.), ratio_y=(1., 1.), interpolation='trilinear')
                random_roll = TensorRandomRoll(p=(0.6, 0.6, 0.6), max_delta=(0.5, 0.5, 0.5), spatial_dims=3)
                random_spherical_rotation = TensorRandomSphericalRotation(p=0.6, max_degrees=(20,20,20), n_channels=n_channels, img_size=tuple(obs_size[1:]))
                random_z_flip = TensorRandomFlip(p=0.2, dim_flip=-3)
                random_y_flip = TensorRandomFlip(p=0.2, dim_flip=-2)
                random_x_flip = TensorRandomFlip(p=0.2, dim_flip=-1)
                transform = Compose([random_center_crop, random_roll, random_spherical_rotation, random_z_flip, random_y_flip, random_x_flip])
            elif spatial_dims == 2:
                random_center_crop = TensorRandomCentroidCrop(p=0.6, size=tuple(obs_size[1:]), scale=(0.5, 1.0), ratio_x=(1., 1.), interpolation='bilinear')
                random_roll = TensorRandomRoll(p=(0.6, 0.6), max_delta=(0.5, 0.5), spatial_dims=2)
                random_spherical_rotation = TensorRandomSphericalRotation(p=0.6, max_degrees=20, n_channels=n_channels, img_size=tuple(obs_size[1:]))
                random_y_flip = TensorRandomFlip(p=0.2, dim_flip=-2)
                random_x_flip = TensorRandomFlip(p=0.2, dim_flip=-1)
                transform = Compose([random_center_crop, random_roll, random_spherical_rotation, random_y_flip, random_x_flip])
            explorer_config.goalspace_training.train_dataset.transform = transform

        explorer_config.goalspace_training.valid_dataset.filepath = "data/training/valid_dataset.h5"
        explorer_config.goalspace_training.valid_dataset.config = deepcopy(explorer_config.goalspace_training.train_dataset.config)

        explorer_config.goalspace_training.train_dataloader.batch_size = int(128)
        explorer_config.goalspace_training.train_dataloader.num_workers = 0
        explorer_config.goalspace_training.train_dataloader.drop_last = False
        explorer_config.goalspace_training.train_dataloader.collate_fn = eval("None")

        explorer_config.goalspace_training.valid_dataloader.batch_size = int(128)
        explorer_config.goalspace_training.valid_dataloader.num_workers = 0
        explorer_config.goalspace_training.valid_dataloader.drop_last = False
        explorer_config.goalspace_training.valid_dataloader.collate_fn = eval("None")

        explorer_config.goalspace_training.frequency = int(400)
        explorer_config.goalspace_training.n_epochs = int(400)
        explorer_config.goalspace_training.importance_sampling_last = float(0.3)

        if explorer_cls.__name__ == "IMGEP_HOLMES_Explorer":
            explorer_config.goalspace_training.split_trigger.active = bool(1)
            explorer_config.goalspace_training.split_trigger.fitness_key = "recon"
            explorer_config.goalspace_training.split_trigger.type = "plateau"
            explorer_config.goalspace_training.split_trigger.parameters = {"epsilon": 20, "n_steps_average": 100}
            explorer_config.goalspace_training.split_trigger.conditions = {"min_init_n_epochs": 2000, "n_min_points": 750, "n_max_splits": 10, "n_epochs_min_between_splits": 400}
            explorer_config.goalspace_training.split_trigger.save_model_before_after = True
            explorer_config.goalspace_training.split_trigger.boundary_config = {"z_fitness": "recon_loss", "algo": "cluster.KMeans"}
            explorer_config.goalspace_training.alternated_backward = {"active": True, "ratio_epochs": {"connections": 2, "core": 8}}

    return explorer_config

def get_number_of_explorations():
    return 5000


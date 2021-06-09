from evocraftsearch import ExplorationDB
from evocraftsearch.systems import LeniaChem
from evocraftsearch.systems.torch_nn.leniachem import LeniaChemInitializationSpace, LeniaChemUpdateRuleSpace
from image_representation import VAE, MEVAE
from evocraftsearch.output_representation import ImageStatisticsRepresentation
from evocraftsearch.explorers import IMGEPExplorer, BoxGoalSpace, IMGEP_OGL_Explorer, TorchNNBoxGoalSpace
from evocraftsearch.output_fitness.displacement import DisplacementFitness
import os
import torch
import warnings
from exputils.seeding import set_seed
from evocraftsearch.utils import map_nested_dicts_modify
import importlib

def run_exploration(experiment_config, output_data_folder):

    print('Loading Config ... ')
    seed = experiment_config.get_seed()
    neat_config, initialization_space_config = experiment_config.get_initialization_space_config()
    n_kernels, update_rule_space_config = experiment_config.get_update_rule_space_config()
    system_config = experiment_config.get_system_config()
    exploration_db_config = experiment_config.get_exploration_db_config()
    representation_cls = experiment_config.get_representation_cls()
    representation_config = experiment_config.get_representation_config()
    goal_space_cls = experiment_config.get_goal_space_cls()
    goal_space_config = experiment_config.get_goal_space_config()
    explorer_cls = experiment_config.get_explorer_cls()
    explorer_config = experiment_config.get_explorer_config()
    n_exploration_runs = experiment_config.get_number_of_explorations()

    score_function = None

    print('Set seed ... ')
    set_seed(seed)

    print('Disable Torch CudNN backend')
    torch.backends.cudnn.enabled = False

    # experiment from scratch
    if not os.path.exists(os.path.join(output_data_folder, 'explorer.pickle')):

        print('Create System ... ')

        initialization_space = LeniaChemInitializationSpace(len(system_config.blocks_list), neat_config, config=initialization_space_config)

        update_rule_space = LeniaChemUpdateRuleSpace(len(system_config.blocks_list), n_kernels, neat_config, config=update_rule_space_config)

        system = LeniaChem(initialization_space=initialization_space, update_rule_space=update_rule_space, config=system_config, device=system_config.device)

        print('Create DB ... ')

        exploration_db = ExplorationDB(config=exploration_db_config)

        print('Create Representation ... ')

        representation = representation_cls(config=representation_config)

        if goal_space_config is not None:
            goal_space = goal_space_cls(representation, autoexpand=True, config=goal_space_config)
        else:
            goal_space = goal_space_cls(representation, autoexpand=True)

        print('Create Explorer ... ')

        explorer = explorer_cls(system=system, explorationdb=exploration_db, goal_space=goal_space, score_function=score_function, config=explorer_config)

        continue_existing_run = False

    # seek existing explorer to restart crashed experiment
    else:

        print('Load existing Explorer ... ')

        explorer = explorer_cls.load(os.path.join(output_data_folder, 'explorer.pickle'), load_data=False, map_location='cpu')
        del explorer.policy_library[-1]
        if "HOLMES" in explorer_cls.__name__:
            map_nested_dicts_modify(explorer.goal_library, lambda node_goal_library: node_goal_library[:-1])
        else:
            explorer.goal_library = explorer.goal_library[:-1]

        run_ids = list(range(len(explorer.policy_library)))
        explorer.system.run_idx = len(explorer.policy_library)
        explorer.db = ExplorationDB(config=exploration_db_config)
        explorer.db.load(run_ids=run_ids, map_location='cpu')
        explorer.goal_space = goal_space_cls.load(os.path.join(output_data_folder, 'goal_space.pickle'), map_location='cpu')

        policy_len = len(explorer.policy_library)
        if "HOLMES" in explorer_cls.__name__:
            goal_library_len = len(list(explorer.goal_library.values())[0])
        else:
            goal_library_len = len(explorer.goal_library)
        n_runs = len(explorer.db.run_ids)
        assert policy_len == goal_library_len == n_runs
        continue_existing_run = True
        warnings.warn(f'/!\ Explorer already existing in {os.path.join(output_data_folder, "explorer.pickle")} => Reloading it and starting experiment from that state, at run {len(explorer.policy_library)}')


    print('Run exploration ...')
    explorer.run(n_exploration_runs, continue_existing_run=continue_existing_run, save_frequency=20, save_filepath=output_data_folder)

    print('Save explorer ...')
    explorer.save(output_data_folder)

    print('Finished.')
    return explorer

if __name__ == '__main__':
    # Put the name of experiment you want to reproduce
    experiment_name = "16_3D_degrowth"  # "64_2D_growth", "64_2D_degrowth", "32_2D_growth", "32_2D_growth", "16_3D_growth", "16_3D_degrowth"

    print("Load Config")
    config_filename = f"experiment_config_{experiment_name}"
    spec = importlib.util.spec_from_file_location(f"experiment_config_{experiment_name}", f"experiment_config_{experiment_name}.py")
    experiment_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment_config)

    print("Create Output Folder")
    output_data_folder = "./data/"
    if not os.path.exists(output_data_folder):
        os.makedirs(output_data_folder)
    if not os.path.exists(os.path.join(output_data_folder, 'exploration_db')):
        os.makedirs(os.path.join(output_data_folder, 'exploration_db'))
    if not os.path.exists(os.path.join(output_data_folder, 'training')):
        os.makedirs(os.path.join(output_data_folder, 'training'))

    run_exploration(experiment_config, output_data_folder)

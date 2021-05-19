import torch
import random
from addict import Dict
from evocraftsearch import Explorer
from evocraftsearch.explorers import IMGEP_OGL_Explorer
from evocraftsearch.spaces import BoxSpace, DictSpace
from evocraftsearch.utils import sample_value, map_nested_dicts, map_nested_dicts_modify, map_nested_dicts_modify_with_other
from image_representation.datasets import torch_dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import numbers
from tqdm import tqdm
import os
from exputils.seeding import set_seed


class HolmesGoalSpace(DictSpace):

    @staticmethod
    def default_config():
        default_config = Dict()

        # niche selection
        default_config.niche_selection.type = 'random'

        # goal selection
        default_config.goal_selection.type = 'random'

        return default_config

    def __init__(self, representation, autoexpand=True, low=0., high=0., shape=None, dtype=torch.float32, config={}, **kwargs):
        """
        representation: TorchNN Holmes Image Representation  a flat dictionary with node's keys, we wrap here its main function
        """
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        self.representation = representation
        self.autoexpand = autoexpand

        if shape is not None:
            if isinstance(shape, list) or isinstance(shape, tuple):
                assert len(shape) == 1 and shape[0] == self.representation.n_latents
            elif isinstance(shape, numbers.Number):
                assert shape == self.representation.n_latents

        spaces_keys = representation.network.get_leaf_pathes()
        low_dict = Dict.fromkeys(spaces_keys, 0.0)
        if isinstance(low, numbers.Number):
            low_dict = Dict.fromkeys(spaces_keys, low)
        elif isinstance(low, Dict):
            low_dict.update(low)
        high_dict = Dict.fromkeys(spaces_keys, 0.0)
        if isinstance(high, numbers.Number):
            high_dict = Dict.fromkeys(spaces_keys, high)
        elif isinstance(high, Dict):
            high_dict.update(high)

        spaces = Dict()
        for k in spaces_keys:
            spaces[k] = BoxSpace(low=low_dict[k], high=high_dict[k], shape=(representation.n_latents,), dtype=dtype)
        DictSpace.__init__(self, spaces=spaces)

    def update(self):
        if self.representation.network.get_leaf_pathes() != list(self.spaces.keys()):
            self.spaces = Dict()
            for leaf_path in self.representation.network.get_leaf_pathes():
                self.spaces[leaf_path] = BoxSpace(low=0., high=0.0, shape=(self.representation.network.get_child_node(leaf_path).n_latents, ), dtype=torch.float32)

    def map(self, observations, preprocess=None, dataset_type="potential", **kwargs):
        if dataset_type == "potential":
            data = observations.potentials[-1].unsqueeze(0)
        elif dataset_type == "onehot":
            data = observations.onehot_states[-1].unsqueeze(0)
        elif dataset_type == "rgb":
            data = observations.rgb_states[-1].unsqueeze(0)

        squeeze_dims = torch.where(torch.tensor(data.shape[2:]) == 1)[0]
        for dim in squeeze_dims:
            data = data.squeeze(dim.item() + 2)

        embedding = self.representation.calc_embedding(data, mode="exhaustif")
        map_nested_dicts_modify(embedding, lambda z: z.squeeze())

        if self.autoexpand:
            new_embedding = map_nested_dicts(embedding, lambda z: z.cpu().detach())
            for node_path, space in self.spaces.items():
                node_embedding = new_embedding[node_path]
                is_nan_mask = torch.isnan(node_embedding)
                node_embedding[is_nan_mask] = space.low[is_nan_mask]
                space.low = torch.min(space.low, node_embedding)
                node_embedding[is_nan_mask] = space.high[is_nan_mask]
                space.high = torch.max(space.high, node_embedding)

        return embedding


    def calc_distance(self, embedding_a, embedding_b, target_space, **kwargs):
        #/!\ for now embedding_b is a dict wich embedding vector(s) per space and embedding_a is a single vector
        embedding_b = embedding_b[target_space]
        #L2 by default
        scale = (self.spaces[target_space].high - self.spaces[target_space].low).to(self.representation.config.device)
        scale[torch.where(scale == 0.0)] = 1.0
        low = self.spaces[target_space].low.to(self.representation.config.device)
        embedding_a = (embedding_a - low) / scale
        embedding_b = (embedding_b - low) / scale
        dist = (embedding_a - embedding_b).pow(2).sum(-1).sqrt()
        return dist


    def sample(self):
        if self.config.niche_selection.type == 'random':
            node_path = random.choice(list(self.spaces.keys()))
            space = self.spaces[node_path]
        else:
            raise NotImplementedError

        if self.config.goal_selection.type == 'random':
            goal = space.sample().to(self.representation.config.device)

        else:
            raise NotImplementedError

        return node_path, goal

    def save(self, filepath):
        representation_checkpoint = self.representation.get_checkpoint()
        low_dict = Dict.fromkeys(self.spaces.keys(), None)
        high_dict = Dict.fromkeys(self.spaces.keys(), None)
        for k, space in self.spaces.items():
            low_dict[k] = space.low
            high_dict[k] = space.high
        goal_space_checkpoint = {'representation_cls': self.representation.__class__,
                                 'representation': representation_checkpoint,
                                 'autoexpand': self.autoexpand,
                                 'low': low_dict,
                                 'high': high_dict,
                                 }
        torch.save(goal_space_checkpoint, filepath)


    @staticmethod
    def load(filepath, map_location='cpu'):

        goal_space_checkpoint = torch.load(filepath, map_location=map_location)
        representation_cls = goal_space_checkpoint["representation_cls"]
        representation_config = goal_space_checkpoint["representation"]["config"]
        representation = representation_cls(config=representation_config)
        representation.n_epochs = goal_space_checkpoint["representation"]["epoch"]
        split_history = goal_space_checkpoint["representation"]["split_history"]
        for split_node_path, split_node_attr in split_history.items():
            representation.split_node(split_node_path)
            node = representation.network.get_child_node(split_node_path)
            node.boundary = split_node_attr["boundary"]
            node.feature_range = split_node_attr["feature_range"]
        representation.split_history = split_history
        representation.network.load_state_dict(goal_space_checkpoint["representation"]["network_state_dict"])
        representation.optimizer.load_state_dict(goal_space_checkpoint["representation"]["optimizer_state_dict"])
        for state in representation.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        goal_space = HolmesGoalSpace(representation=representation, low=goal_space_checkpoint["low"], high=goal_space_checkpoint["high"], autoexpand=goal_space_checkpoint["autoexpand"])

        return goal_space



class IMGEP_HOLMES_Explorer(IMGEP_OGL_Explorer):
    """
    Basic explorer that samples goals in a goalspace and uses a policy library to generate parameters to reach the goal.
    """

    # Set these in ALL subclasses
    goal_space = None  # defines the obs->goal representation and the goal sampling strategy (self.goal_space.sample())
    reach_goal_optimizer = None

    @staticmethod
    def default_config():
        default_config = IMGEP_OGL_Explorer.default_config()

        default_config.goalspace_training.split_trigger.active = True
        default_config.goalspace_training.split_trigger.fitness_key = 'recon'
        default_config.goalspace_training.split_trigger.type = 'plateau'
        default_config.goalspace_training.split_trigger.parameters = Dict(epsilon=20, n_steps_average=50)
        default_config.goalspace_training.split_trigger.conditions = Dict(min_init_n_epochs=2000, n_min_points=500, n_max_splits=10, n_epochs_min_between_splits=200)
        default_config.goalspace_training.split_trigger.save_model_before_after = True
        default_config.goalspace_training.split_trigger.boundary_config.z_fitness = "recon_loss"
        default_config.goalspace_training.split_trigger.boundary_config.algo = "cluster.KMeans"
        default_config.goalspace_training.alternated_backward.active = True
        default_config.goalspace_training.alternated_backward.ratio_epochs = {"connections": 2, "core": 8}


        return default_config

    def __init__(self, system, explorationdb, goal_space, config={}, **kwargs):
        Explorer.__init__(self, system=system, explorationdb=explorationdb, config=config, **kwargs)

        self.goal_space = goal_space

        # initialize policy library
        self.policy_library = []

        # initialize goal library as Dict
        self.goal_library = Dict()
        self.goal_library["0"] = torch.empty((0,) + self.goal_space.spaces["0"].shape)


    def get_source_policy_idx(self, target_space, target_goal):

        if self.config.source_policy_selection.type == 'optimal':
            # get distance to other goals
            goal_distances = self.goal_space.calc_distance(target_goal, self.goal_library, target_space)

            # select goal with minimal distance
            isnan_distances = torch.isnan(goal_distances)
            canditates = torch.where(goal_distances == goal_distances[~isnan_distances].min())[0]
            source_policy_idx = random.choice(canditates)

        elif self.config.source_policy_selection.type == 'random':
            source_policy_idx = sample_value(('discrete', 0, len(self.goal_library[target_space]) - 1))

        else:
            raise ValueError('Unknown source policy selection type {!r} in the configuration!'.format(self.config.source_policy_selection.type))

        return source_policy_idx


    def run(self, n_exploration_runs, continue_existing_run=False, save_frequency=None, save_filepath=None):

        print('Exploration: ')
        progress_bar = tqdm(total=n_exploration_runs)

        # prepare train and valid datasets
        dataset_cls = torch_dataset.HDF5Dataset
        train_dataset = dataset_cls(self.config.goalspace_training.train_dataset.filepath, config=self.config.goalspace_training.train_dataset.config)
        weights_train_dataset = [1.]
        weighted_train_sampler = WeightedRandomSampler(weights_train_dataset, 1)
        train_loader = DataLoader(train_dataset,
                                  sampler=weighted_train_sampler,
                                  batch_size=self.config.goalspace_training.train_dataloader.batch_size,
                                  num_workers=self.config.goalspace_training.train_dataloader.num_workers,
                                  drop_last=self.config.goalspace_training.train_dataloader.drop_last,
                                  collate_fn=self.config.goalspace_training.train_dataloader.collate_fn)

        self.config.goalspace_training.dataset_config.data_augmentation = False
        valid_dataset = dataset_cls(self.config.goalspace_training.valid_dataset.filepath, config=self.config.goalspace_training.valid_dataset.config)
        weights_valid_dataset = [1.]
        weighted_valid_sampler = WeightedRandomSampler(weights_valid_dataset, 1)
        valid_loader = DataLoader(valid_dataset,
                                  sampler=weighted_valid_sampler,
                                  batch_size=self.config.goalspace_training.valid_dataloader.batch_size,
                                  num_workers=self.config.goalspace_training.valid_dataloader.num_workers,
                                  drop_last=self.config.goalspace_training.valid_dataloader.drop_last,
                                  collate_fn=self.config.goalspace_training.valid_dataloader.collate_fn)

        if continue_existing_run:
            run_idx = len(self.policy_library)
            progress_bar.update(run_idx)

            # Update goal library with latest representation (and goal space extent when autoexpand) in case was not done before saving
            self.goal_space.update()
            self.goal_space.representation.eval()
            if list(self.goal_space.spaces.keys()) != list(self.goal_library.keys()):
                self.goal_library = Dict()
                for node_path, space in self.goal_space.spaces.items():
                    self.goal_library[node_path] = torch.empty((0,) + space.shape)

            map_nested_dicts_modify(self.goal_library, lambda node_goal_library: node_goal_library.to(self.goal_space.representation.config.device))

            # recreate train/valid datasets from saved exploration_db
            for run_data_idx in self.db.run_ids:
                run_data = self.db.get_run_data(run_data_idx)

                with torch.no_grad():
                    reached_goal = self.goal_space.map(run_data.observations,
                                                       preprocess=self.config.goalspace_preprocess,
                                                       dataset_type=self.config.goalspace_training.dataset_type)
                    map_nested_dicts_modify_with_other(self.goal_library, reached_goal, lambda ob1, ob2: torch.cat(
                        [ob1[:run_data_idx], ob2.reshape(1, -1).detach(), ob1[run_data_idx + 1:]]))

            assert len(self.policy_library) == len(list(self.goal_library.values())[0]) == len(self.db.run_ids)

        else:
            self.policy_library = []
            self.goal_library = Dict()
            self.goal_library["0"] = torch.empty((0,) + self.goal_space.spaces["0"].shape).to(self.goal_space.representation.config.device)
            run_idx = 0


        while run_idx < n_exploration_runs:

            set_seed(100000 * self.config.seed + run_idx)

            # Initial Random Sampling of Parameters
            if run_idx < self.config.num_of_random_initialization:

                target_goal = None
                source_policy_idx = None

                policy_parameters = self.system.sample_policy_parameters()
                self.system.reset(policy=policy_parameters)

                self.goal_space.representation.eval()
                with torch.no_grad():
                    observations = self.system.run()
                    reached_goal = self.goal_space.map(observations, preprocess=self.config.goalspace_preprocess, dataset_type=self.config.goalspace_training.dataset_type)

                optim_step_idx = 0
                dist_to_target = None

            # Goal-directed Sampling of Parameters
            else:

                # sample a goal space from the goal space
                target_space, target_goal = self.goal_space.sample()  # provide the explorer to sampling function if needed (ef: for sampling in less dense region we need access to self.goal_library, etc)

                # get source policy which should be mutated
                source_policy_idx = self.get_source_policy_idx(target_space, target_goal)
                source_policy = self.policy_library[source_policy_idx]

                # apply a mutation
                policy_parameters = self.system.mutate_policy_parameters(source_policy)
                self.system.reset(policy=policy_parameters)

                # Optimization toward target goal
                if isinstance(self.system, torch.nn.Module) and self.config.reach_goal_optim_steps > 0:
                    train_losses = self.system.optimize(self.config.reach_goal_optim_steps,
                                                        lambda obs: self.goal_space.calc_distance(target_goal, 
                                                                                                  self.goal_space.map(
                                                                                                      obs, 
                                                                                                      preprocess=self.config.goalspace_preprocess,
                                                                                                      dataset_type=self.config.goalspace_training.dataset_type,
                                                                                                  ),
                                                                                                  target_space))
                    print(train_losses)
                    policy_parameters['initialization'] = self.system.initialization_parameters
                    policy_parameters['update_rule'] = self.system.update_rule_parameters

                with torch.no_grad():
                    observations = self.system.run()
                    reached_goal = self.goal_space.map(observations, preprocess=self.config.goalspace_preprocess, dataset_type=self.config.goalspace_training.dataset_type)
                    loss = self.goal_space.calc_distance(target_goal, reached_goal, target_space)
                    dist_to_target = loss.item()


            # save results
            self.db.add_run_data(id=run_idx,
                                 policy_parameters=policy_parameters,
                                 observations=observations,
                                 source_policy_idx=source_policy_idx,
                                 target_goal=target_goal,
                                 reached_goal=reached_goal,
                                 dist_to_target=dist_to_target)

            if self.db.config.save_rollout_render:
                self.system.render_rollout(observations, filepath=os.path.join(self.db.config.db_directory,
                                                                                   f'run_{run_idx}_rollout'))


            # add policy and reached goal into the libraries
            # do it after the run data is saved to not save them if there is an error during the saving
            self.policy_library.append(policy_parameters)
            map_nested_dicts_modify_with_other(self.goal_library, reached_goal, lambda ob1, ob2: torch.cat([ob1, ob2.reshape(1, -1).detach()]))

            # append new discovery to train/valid dataset
            t_slide = observations.potentials.shape[0] // (self.config.goalspace_training.dataset_n_timepoints -1)
            for t_idx in range(self.config.goalspace_training.dataset_n_timepoints):
                timepoint = max(-1 - t_idx * t_slide, -observations.potentials.shape[0])

                discrete_potential = observations.potentials[timepoint].detach().argmax(0)
                is_dead = torch.all(discrete_potential.eq(discrete_potential[0, 0, 0]))  # if all values converge to same block, we consider the output dead
                if is_dead: # if dead we dont add to the training database
                    continue

                if self.config.goalspace_training.dataset_type == "potential":
                    data = observations.potentials[timepoint]
                elif self.config.goalspace_training.dataset_type == "onehot":
                    data = observations.onehot_states[timepoint]
                elif self.config.goalspace_training.dataset_type == "rgb":
                    data = observations.rgb_states[timepoint]

                # convert to 2D if one of the SX,SY,SZ dims is 1
                squeeze_dims = torch.where(torch.tensor(data.shape[1:]) == 1)[0]
                for dim in squeeze_dims:
                    data = data.squeeze(dim.item() + 1)
                if (len(train_loader.dataset) + len(valid_loader.dataset)) % 10 == 0:
                    valid_loader.dataset.add_data(data, torch.tensor([0]))
                else:
                    train_loader.dataset.add_data(data, torch.tensor([0]))


            # save explorer
            if (save_frequency is not None) and (run_idx % save_frequency == 0):
                if (save_filepath is not None) and (os.path.exists(save_filepath)):
                    self.save(save_filepath)

            # Training stage
            if run_idx % self.config.goalspace_training.frequency == 0 and len(train_loader.dataset) > 0:
                stage_idx = run_idx // self.config.goalspace_training.frequency
                representation_filepath = os.path.join(self.goal_space.representation.config.checkpoint.folder,
                                                       'stage_{:06d}_weight_model.pth'.format(stage_idx))

                # current trick to avoid redoing training stage when already done
                if not os.path.exists(representation_filepath):

                    # Importance Sampling
                    if stage_idx == 1:
                        weights = [1.0 / len(train_loader.dataset)] * len(train_loader.dataset)
                        train_loader.sampler.num_samples = len(weights)
                        train_loader.sampler.weights = torch.tensor(weights, dtype=torch.double)
                    elif stage_idx > 1:
                        weights = [(
                                               1.0 - self.config.goalspace_training.importance_sampling_last) / self.train_dset_laststage_counter] * self.train_dset_laststage_counter
                        weights += ([self.config.goalspace_training.importance_sampling_last / (
                                    len(train_loader.dataset) - self.train_dset_laststage_counter)] * (
                                                len(train_loader.dataset) - self.train_dset_laststage_counter))
                        train_loader.sampler.num_samples = len(weights)
                        train_loader.sampler.weights = torch.tensor(weights, dtype=torch.double)

                    valid_weights = [1.0 / len(valid_loader.dataset)] * len(valid_loader.dataset)
                    valid_loader.sampler.num_samples = len(valid_weights)
                    valid_loader.sampler.weights = torch.tensor(valid_weights, dtype=torch.double)

                    # Training
                    torch.backends.cudnn.enabled = True
                    self.goal_space.representation.train()
                    self.goal_space.representation.run_training(train_loader=train_loader,
                                                                training_config=self.config.goalspace_training,
                                                                valid_loader=valid_loader)
                    torch.backends.cudnn.enabled = False

                    # update counters
                    self.train_dset_laststage_counter = len(train_loader.dataset)

                    # Save the trained representation
                    self.goal_space.representation.save(representation_filepath)

                    # Update goal space and goal library
                    self.goal_space.update()  # if holmes has splitted will update goal space keys accordingly
                    self.goal_space.representation.eval()
                    ## update library and goal space extent with discoveries projected with the updated representation
                    if list(self.goal_space.spaces.keys()) != list(self.goal_library.keys()):
                        self.goal_library = Dict()
                        for node_path, space in self.goal_space.spaces.items():
                            self.goal_library[node_path] = torch.empty((0,) + space.shape).to(
                                self.goal_space.representation.config.device)
                    with torch.no_grad():
                        for old_run_idx in self.db.run_ids:
                            reached_goal = self.goal_space.map(self.db.get_run_data(old_run_idx).observations,
                                                               preprocess=self.config.goalspace_preprocess,
                                                               dataset_type=self.config.goalspace_training.dataset_type)
                            map_nested_dicts_modify_with_other(self.goal_library, reached_goal,
                                                               lambda ob1, ob2: torch.cat([ob1[:old_run_idx],
                                                                                           ob2.reshape(1,
                                                                                                       -1).detach(),
                                                                                           ob1[
                                                                                           old_run_idx + 1:]]))

                    # save after training
                    if (save_filepath is not None) and (os.path.exists(save_filepath)):
                        self.save(save_filepath)

            # increment run_idx
            run_idx += 1
            progress_bar.update(1)

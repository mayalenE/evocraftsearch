import torch
from addict import Dict
from evocraftsearch.explorers import IMGEPExplorer
from evocraftsearch.spaces import BoxSpace
from image_representation.datasets import torch_dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import numbers
from tqdm import tqdm
import os
from exputils.seeding import set_seed

class TorchNNBoxGoalSpace(BoxSpace):
    def __init__(self, representation, autoexpand=True, low=0., high=0., shape=None, dtype=torch.float32):
        """
        representation: TorchNN Image Representation, we wrap here its main function
        """
        self.representation = representation
        self.autoexpand = autoexpand
        if shape is not None:
            if isinstance(shape, list) or isinstance(shape, tuple):
                assert len(shape) == 1 and shape[0] == self.representation.n_latents
            elif isinstance(shape, numbers.Number):
                assert shape == self.representation.n_latents
        BoxSpace.__init__(self, low=low, high=high, shape=(self.representation.n_latents,), dtype=dtype)

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

        embedding = self.representation.calc_embedding(data).squeeze()

        if self.autoexpand:
            embedding_c = embedding.cpu().detach()
            is_nan_mask = torch.isnan(embedding_c)
            if is_nan_mask.sum() > 0:
                embedding_c[is_nan_mask] = self.low[is_nan_mask]
                self.low = torch.min(self.low, embedding_c)
                embedding_c[is_nan_mask] = self.high[is_nan_mask]
                self.high = torch.max(self.high, embedding_c)
            else:
                self.low = torch.min(self.low, embedding_c)
                self.high = torch.max(self.high, embedding_c)

        return embedding

    def calc_distance(self, embedding_a, embedding_b, **kwargs):
        scale = (self.high - self.low).to(self.representation.config.device)
        low = self.low.to(self.representation.config.device)
        # L2 by default
        embedding_a = (embedding_a - low) / scale
        embedding_b = (embedding_b - low) / scale
        dist = (embedding_a - embedding_b).pow(2).sum(-1).sqrt()
        return dist

    def sample(self):
        return BoxSpace.sample(self).to(self.representation.config.device)

    def save(self, filepath):

        representation_checkpoint = self.representation.get_checkpoint()
        goal_space_checkpoint = {'representation_cls': self.representation.__class__,
                                 'representation': representation_checkpoint,
                                 'autoexpand': self.autoexpand,
                                 'low': self.low,
                                 'high': self.high,
                                 }
        torch.save(goal_space_checkpoint, filepath)


    @staticmethod
    def load(filepath, map_location='cpu'):

        goal_space_checkpoint = torch.load(filepath, map_location=map_location)
        representation_cls = goal_space_checkpoint["representation_cls"]
        representation_config = goal_space_checkpoint["representation"]["config"]
        representation = representation_cls(config=representation_config)
        representation.n_epochs = goal_space_checkpoint["representation"]["epoch"]
        representation.network.load_state_dict(goal_space_checkpoint["representation"]["network_state_dict"])
        representation.optimizer.load_state_dict(goal_space_checkpoint["representation"]["optimizer_state_dict"])
        for state in representation.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        goal_space = TorchNNBoxGoalSpace(representation=representation, low=goal_space_checkpoint["low"], high=goal_space_checkpoint["high"], autoexpand=goal_space_checkpoint["autoexpand"])
        return goal_space



class IMGEP_OGL_Explorer(IMGEPExplorer):
    """
    Basic explorer that samples goals in a goalspace and uses a policy library to generate parameters to reach the goal.
    """

    # Set these in ALL subclasses
    goal_space = None  # defines the obs->goal representation and the goal sampling strategy (self.goal_space.sample())
    reach_goal_optimizer = None

    @staticmethod
    def default_config():
        default_config = Dict()
        # base config
        default_config.seed = None
        default_config.num_of_random_initialization = 10  # number of random runs at the beginning of exploration to populate the IMGEP memory

        # Pi: source policy parameters config
        default_config.source_policy_selection = Dict()
        default_config.source_policy_selection.type = 'optimal'  # either: 'optimal', 'random'

        # Opt: Optimizer to reach goal
        default_config.reach_goal_optim_steps = 10

        # R: Goal space Training
        default_config.goalspace_preprocess = None

        default_config.goalspace_training = Dict()
        default_config.goalspace_training.dataset_n_timepoints = 1 #1: loads only last state
        default_config.goalspace_training.dataset_type = "potential"

        default_config.goalspace_training.train_dataset.filepath = None
        default_config.goalspace_training.train_dataset.config = Dict()

        default_config.goalspace_training.valid_dataset.filepath = None
        default_config.goalspace_training.valid_dataset.config = Dict()

        default_config.goalspace_training.train_dataloader.batch_size = 64
        default_config.goalspace_training.train_dataloader.num_workers = 0
        default_config.goalspace_training.train_dataloader.drop_last = False
        default_config.goalspace_training.train_dataloader.collate_fn = None

        default_config.goalspace_training.valid_dataloader.batch_size = 32
        default_config.goalspace_training.valid_dataloader.num_workers = 0
        default_config.goalspace_training.valid_dataloader.drop_last = False
        default_config.goalspace_training.valid_dataloader.collate_fn = None

        default_config.goalspace_training.frequency = 100
        default_config.goalspace_training.n_epochs = 100 #train n_epochs epochs every frequency runs
        default_config.goalspace_training.importance_sampling_last = 0.3 # importance of the last runs when training representation


        return default_config


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

            # recreate train/valid datasets from saved exploration_db
            for run_data_idx in self.db.run_ids:
                run_data = self.db.get_run_data(run_data_idx)

                # Update goal library with latest representation (and goal space extent when autoexpand) in case was not done before saving
                self.goal_space.representation.eval()
                with torch.no_grad():
                    self.goal_library[run_data_idx] = self.goal_space.map(run_data.observations, preprocess=self.config.goalspace_preprocess,
                                                                         dataset_type=self.config.goalspace_training.dataset_type).detach()

            assert len(self.policy_library) == len(self.goal_library) == len(self.db.run_ids)

        else:
            self.policy_library = []
            self.goal_library = torch.empty((0,) + self.goal_space.shape)
            self.train_dset_laststage_counter = 0
            run_idx = 0

        self.goal_library = self.goal_library.to(self.goal_space.representation.config.device)


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
                target_goal = self.goal_space.sample()  # provide the explorer to sampling function if needed (ef: for sampling in less dense region we need access to self.goal_library, etc)

                # get source policy which should be mutated
                source_policy_idx = self.get_source_policy_idx(target_goal)
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
                                                                                                      )))
                    print(train_losses)
                    policy_parameters['initialization'] = self.system.initialization_parameters
                    policy_parameters['update_rule'] = self.system.update_rule_parameters

                with torch.no_grad():
                    observations = self.system.run()
                    reached_goal = self.goal_space.map(observations, preprocess=self.config.goalspace_preprocess, dataset_type=self.config.goalspace_training.dataset_type)
                    loss = self.goal_space.calc_distance(target_goal, reached_goal)
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
            self.goal_library = torch.cat([self.goal_library, reached_goal.reshape(1, -1).detach()])

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

                    # Update goal library with latest representation (and goal space extent when autoexpand)
                    self.goal_space.representation.eval()
                    with torch.no_grad():
                        for old_run_idx in self.db.run_ids:
                            self.goal_library[old_run_idx] = self.goal_space.map(
                                self.db.get_run_data(old_run_idx).observations,
                                preprocess=self.config.goalspace_preprocess,
                                dataset_type=self.config.goalspace_training.dataset_type).detach()

                    # save after training
                    if (save_filepath is not None) and (os.path.exists(save_filepath)):
                        self.save(save_filepath)

            # increment run_idx
            run_idx += 1
            progress_bar.update(1)

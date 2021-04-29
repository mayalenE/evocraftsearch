import torch
import random
from addict import Dict
from evocraftsearch import Explorer
from evocraftsearch.spaces import BoxSpace
from evocraftsearch.utils import sample_value
from image_representation.datasets.torch_dataset import EvocraftDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import numbers
from tqdm import tqdm
import os

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

    def map(self, observations, preprocess=None, **kwargs):
        last_potential = observations.potentials[-1].permute(3,2,1,0).unsqueeze(0)
        if preprocess is not None:
            last_potential = preprocess(last_potential)
        embedding = self.representation.calc_embedding(last_potential).squeeze()

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
        calc_dist_op = getattr(self.representation, "calc_distance", None)
        if callable(calc_dist_op):
            return self.representation.calc_distance(embedding_a, embedding_b, **kwargs)
        else:
            #L2 by default
            dist = (embedding_a - embedding_b).pow(2).sum(-1).sqrt()
            return dist

    def sample(self):
        return BoxSpace.sample(self).to(self.representation.config.device)


class IMGEP_OGL_Explorer(Explorer):
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
        default_config.num_of_random_initialization = 10  # number of random runs at the beginning of exploration to populate the IMGEP memory

        # Pi: source policy parameters config
        default_config.source_policy_selection = Dict()
        default_config.source_policy_selection.type = 'optimal'  # either: 'optimal', 'random'

        # Opt: Optimizer to reach goal
        default_config.reach_goal_optim_steps = 10

        # R: Goal space Training
        default_config.goalspace_preprocess = None

        default_config.goalspace_training = Dict()
        default_config.goalspace_training.dataset_append_trajectory = False #True: loads all states (in time), False: loads only last state
        default_config.goalspace_training.dataset_augment = True
        default_config.goalspace_training.dataset_preprocess = None
        default_config.goalspace_training.train_batch_size = 64
        default_config.goalspace_training.valid_batch_size = 32
        default_config.goalspace_training.frequency = 100
        default_config.goalspace_training.n_epochs = 100 #train n_epochs epochs every frequency runs
        default_config.goalspace_training.importance_sampling_last = 0.3 # importance of the last <frequency> runs when training representation


        return default_config

    def __init__(self, system, explorationdb, goal_space, config={}, **kwargs):
        super().__init__(system=system, explorationdb=explorationdb, config=config, **kwargs)

        self.goal_space = goal_space

        # initialize policy library
        self.policy_library = []

        # initialize goal library
        self.goal_library = torch.empty((0,) + self.goal_space.shape)

    def get_source_policy_idx(self, target_goal):

        if self.config.source_policy_selection.type == 'optimal':
            # get distance to other goals
            goal_distances = self.goal_space.calc_distance(target_goal, self.goal_library)

            # select goal with minimal distance
            isnan_distances = torch.isnan(goal_distances)
            canditates = torch.where(goal_distances == goal_distances[~isnan_distances].min())[0]
            source_policy_idx = random.choice(canditates)

        elif self.config.source_policy_selection.type == 'random':
            source_policy_idx = sample_value(('discrete', 0, len(self.goal_library) - 1))

        else:
            raise ValueError('Unknown source policy selection type {!r} in the configuration!'.format(
                self.config.source_policy_selection.type))

        return source_policy_idx

    def run(self, n_exploration_runs, continue_existing_run=False):

        print('Exploration: ')
        progress_bar = tqdm(total=n_exploration_runs)

        # prepare train and valid datasets
        train_dataset = EvocraftDataset(config=self.config.goalspace_training.dataset_config)
        weights_train_dataset = [1.]
        weighted_sampler = WeightedRandomSampler(weights_train_dataset, 1)
        train_loader = DataLoader(train_dataset, batch_size=self.config.goalspace_training.train_batch_size,
                                  sampler=weighted_sampler, num_workers=0)

        self.config.goalspace_training.dataset_config.data_augmentation = False
        valid_dataset = EvocraftDataset(config=self.config.goalspace_training.dataset_config)
        valid_loader = DataLoader(valid_dataset, self.config.goalspace_training.valid_batch_size, num_workers=0)

        if continue_existing_run:
            run_idx = len(self.policy_library)
            progress_bar.update(run_idx)

            # recreate train/valid datasets from saved exploration_db
            for run_idx, run_data in self.db.runs.items():
                # only add non-dead data
                if self.config.goalspace_training.dataset_append_trajectory:
                    timepoints_to_add = list(range(run_data.observations.potentials))
                else:
                    timepoints_to_add = [-1]
                for timepoint in timepoints_to_add:
                    potential = run_data.observations.potentials[timepoint].permute(3,2,1,0)
                    discrete_potential = potential.detach().argmax(0)
                    is_dead = torch.all(discrete_potential.eq(discrete_potential[0, 0, 0]))
                    if not is_dead:
                        if not is_dead:
                            if (train_loader.dataset.n_images + valid_loader.dataset.n_images) % 10 == 0:
                                valid_loader.dataset.add(potential.unsqueeze(0).cpu().detach().type(self.goal_space.representation.config.dtype),
                                                         torch.tensor([0]).unsqueeze(0))
                            else:
                                train_loader.dataset.add(potential.unsqueeze(0).cpu().detach().type(self.goal_space.representation.config.dtype),
                                                         torch.tensor([0]).unsqueeze(0))

        else:
            self.policy_library = []
            self.goal_library = torch.empty((0,) + self.goal_space.shape)
            run_idx = 0

        self.goal_library = self.goal_library.to(self.goal_space.representation.config.device)


        while run_idx < n_exploration_runs:

            # Initial Random Sampling of Parameters
            if len(self.policy_library) < self.config.num_of_random_initialization:

                target_goal = None
                source_policy_idx = None

                policy_parameters = self.system.sample_policy_parameters()
                self.system.reset(policy=policy_parameters)

                with torch.no_grad():
                    observations = self.system.run()
                    reached_goal = self.goal_space.map(observations, preprocess=self.config.goalspace_preprocess)

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
                                                                                                  self.goal_space.map(obs, preprocess=self.config.goalspace_preprocess)))
                    print(train_losses)
                    policy_parameters['initialization'] = self.system.initialization_parameters
                    policy_parameters['update_rule'] = self.system.update_rule_parameters

                with torch.no_grad():
                    observations = self.system.run()
                    reached_goal = self.goal_space.map(observations, preprocess=self.config.goalspace_preprocess)
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
            if self.config.goalspace_training.dataset_append_trajectory:
                timepoints_to_add = list(range(observations.potentials))
            else:
                timepoints_to_add = [-1]

            for timepoint in timepoints_to_add:
                potential = observations.potentials[timepoint].permute(3, 2, 1, 0)
                discrete_potential = potential.detach().argmax(0)
                is_dead = torch.all(discrete_potential.eq(discrete_potential[0, 0, 0]))
                if not is_dead:
                    if not is_dead:
                        if (train_loader.dataset.n_images + valid_loader.dataset.n_images) % 10 == 0:
                            valid_loader.dataset.add(potential.unsqueeze(0).cpu().detach().type(self.goal_space.representation.config.dtype),
                                                     torch.tensor([0]).unsqueeze(0))
                        else:
                            train_loader.dataset.add(potential.unsqueeze(0).cpu().detach().type(self.goal_space.representation.config.dtype),
                                                     torch.tensor([0]).unsqueeze(0))



            # training stage
            if len(self.policy_library) % self.config.goalspace_training.frequency == 0:
                # Importance Sampling
                stage_idx = len(self.policy_library) // self.config.goalspace_training.frequency
                if stage_idx >= 1:
                    weights = [1.0 / train_loader.dataset.n_images] * (train_loader.dataset.n_images)
                    train_loader.sampler.num_samples = len(weights)
                    train_loader.sampler.weights = torch.tensor(weights, dtype=torch.double)
                elif stage_idx <= 1:
                    weights = [(1.0 - self.config.goalspace_training.importance_sampling_last) / (
                            train_loader.dataset.n_images - self.config.goalspace_training.frequency)] * (
                                      train_loader.dataset.n_images - self.config.goalspace_training.frequency)
                    weights += ([self.config.goalspace_training.importance_sampling_last / self.config.goalspace_training.frequency] * self.config.goalspace_training.frequency)
                    train_loader.sampler.num_samples = len(weights)
                    train_loader.sampler.weights = torch.tensor(weights, dtype=torch.double)

                # Training
                self.goal_space.representation.train()
                self.goal_space.representation.run_training(train_loader=train_loader, training_config=self.config.goalspace_training, valid_loader=valid_loader)

                # Save the trained representation
                representation_filepath = os.path.join(self.goal_space.representation.config.checkpoint.folder,
                                              'stage_{:06d}_weight_model.pth'.format(stage_idx))
                self.goal_space.representation.save(representation_filepath)

                # Update goal library (and goal space extent when autoexpand)
                self.goal_space.representation.eval()
                with torch.no_grad():
                    for old_run_idx in range(len(self.goal_library)):
                        self.goal_library[old_run_idx] = self.goal_space.map(self.db.runs[old_run_idx].observations, preprocess=self.config.goalspace_preprocess).detach()

            # increment run_idx
            run_idx += 1
            progress_bar.update(1)


    def save(self, filepath):
        del self.goal_space.representation.logger
        Explorer.save(self, filepath)


    @staticmethod
    def load(explorer_filepath, load_data=True, run_ids=None, map_location='cuda'):
        explorer = Explorer.load(explorer_filepath, load_data=load_data, run_ids=run_ids, map_location=map_location)

        # deal the config of the goal space representation
        if explorer.goal_space.representation.config.device == 'cuda' and torch.cuda.is_available():
            explorer.goal_space.representation.to('cuda')

        return explorer

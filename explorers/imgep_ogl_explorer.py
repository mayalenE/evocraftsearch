import torch
from addict import Dict
from evocraftsearch import Explorer
from evocraftsearch.spaces import BoxSpace
from evocraftsearch.utils import sample_value
from image_representation.datasets.torch_dataset import LENIADataset
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

    def map(self, observations, **kwargs):
        last_state = observations.states[-1]
        embedding = self.representation.calc_embedding(last_state.unsqueeze(0).unsqueeze(0)).squeeze()

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
        default_config.reach_goal_optimizer = Dict()
        default_config.reach_goal_optimizer.optim_steps = 10
        default_config.reach_goal_optimizer.name = "Adam"
        default_config.reach_goal_optimizer.initialization_cppn.parameters.lr =  1e-3
        default_config.reach_goal_optimizer.potential_ca_step.K.parameters.lr = 1e-2

        # R: Goal space Training
        default_config.goalspace_training = Dict()
        default_config.goalspace_training.dataset_augment = True
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

        # reach goal optimizer
        self.reach_goal_optimizer = None

    def get_source_policy_idx(self, target_goal):

        if self.config.source_policy_selection.type == 'optimal':
            # get distance to other goals
            goal_distances = self.goal_space.calc_distance(target_goal, self.goal_library)

            # select goal with minimal distance
            source_policy_idx = torch.argmin(goal_distances)

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
        dataset_config = Dict()
        dataset_config.img_size = (self.system.config.SX, self.system.config.SY)
        dataset_config.data_augmentation = self.config.goalspace_training.dataset_augment
        train_dataset = LENIADataset(config=dataset_config)
        weights_train_dataset = [1.]
        weighted_sampler = WeightedRandomSampler(weights_train_dataset, 1)
        train_loader = DataLoader(train_dataset, batch_size=self.config.goalspace_training.train_batch_size,
                                  sampler=weighted_sampler, num_workers=0)

        dataset_config.data_augmentation = False
        valid_dataset = LENIADataset(config=dataset_config)
        valid_loader = DataLoader(valid_dataset, self.config.goalspace_training.valid_batch_size, num_workers=0)

        if continue_existing_run:
            run_idx = len(self.policy_library)
            progress_bar.update(run_idx)

            # recreate train/valid datasets from saved exploration_db
            for run_idx, run_data in self.db.runs.items():
                # only add non-dead data
                run_last_state = run_data.observations.states[-1]
                is_dead = torch.all(run_last_state == 1) or torch.all(run_last_state == 0)
                if not is_dead:
                    if (train_loader.dataset.n_images + valid_loader.dataset.n_images) % 10 == 0:
                        valid_loader.dataset.images = torch.cat([valid_loader.dataset.images, run_last_state.unsqueeze(0).unsqueeze(0).cpu().detach()])
                        valid_loader.dataset.labels = torch.cat([valid_loader.dataset.labels, torch.tensor([-1]).unsqueeze(0)])
                        valid_loader.dataset.n_images += 1
                    else:
                        train_loader.dataset.images = torch.cat([train_loader.dataset.images, run_last_state.unsqueeze(0).unsqueeze(0).cpu().detach()])
                        train_loader.dataset.labels = torch.cat([train_loader.dataset.labels, torch.tensor([-1]).unsqueeze(0)])
                        train_loader.dataset.n_images += 1

        else:
            self.policy_library = []
            self.goal_library = torch.empty((0,) + self.goal_space.shape)
            run_idx = 0

        self.goal_library = self.goal_library.to(self.goal_space.representation.config.device)


        while run_idx < n_exploration_runs:

            policy_parameters = Dict.fromkeys(
                ['initialization', 'update_rule'])  # policy parameters (output of IMGEP policy)

            # Initial Random Sampling of Parameters
            if len(self.policy_library) < self.config.num_of_random_initialization:

                target_goal = None
                source_policy_idx = None

                policy_parameters['initialization'] = self.system.initialization_space.sample()
                policy_parameters['update_rule'] = self.system.update_rule_space.sample()
                self.system.reset(initialization_parameters=policy_parameters['initialization'],
                                  update_rule_parameters=policy_parameters['update_rule'])

                with torch.no_grad():
                    observations = self.system.run()
                    reached_goal = self.goal_space.map(observations)


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
                policy_parameters['initialization'] = self.system.initialization_space.mutate(source_policy['initialization'])
                policy_parameters['update_rule'] = self.system.update_rule_space.mutate(source_policy['update_rule'])
                self.system.reset(initialization_parameters=policy_parameters['initialization'],
                                  update_rule_parameters=policy_parameters['update_rule'])

                # Optimization toward target goal
                if isinstance(self.system, torch.nn.Module) and self.config.reach_goal_optimizer.optim_steps > 0:

                    optimizer_class = eval(f'torch.optim.{self.config.reach_goal_optimizer.name}')
                    self.reach_goal_optimizer = optimizer_class([{'params': self.system.initialization_cppn.parameters(), **self.config.reach_goal_optimizer.initialization_cppn.parameters},
                                                                {'params': self.system.potential_ca_step.K.parameters(), **self.config.reach_goal_optimizer.potential_ca_step.K.parameters}])

                    print(f'Run {run_idx}, optimisation toward goal: ')
                    for optim_step_idx in tqdm(range(1, self.config.reach_goal_optimizer.optim_steps)):

                        # run system with IMGEP's policy parameters
                        observations = self.system.run()
                        reached_goal = self.goal_space.map(observations)

                        # compute error between reached_goal and target_goal
                        loss = self.goal_space.calc_distance(target_goal.detach(), reached_goal)
                        print(f'step {optim_step_idx}: distance to target={loss.item():0.2f}')

                        # optimisation step
                        self.reach_goal_optimizer.zero_grad()
                        loss.backward()
                        self.reach_goal_optimizer.step()


                        if optim_step_idx > 5 and abs(old_loss - loss.item()) < 1e-4:
                            break
                        old_loss = loss.item()

                    # gather back the trained parameters
                    self.system.update_initialization_parameters()
                    self.system.update_update_rule_parameters()
                    policy_parameters['initialization'] = self.system.initialization_parameters
                    policy_parameters['update_rule'] = self.system.update_rule_parameters

                    dist_to_target = loss.item()

                    #self.system.render()

                else:
                    with torch.no_grad():
                        observations = self.system.run()
                        reached_goal = self.goal_space.map(observations)
                        loss = self.goal_space.calc_distance(target_goal, reached_goal)
                    optim_step_idx = 0
                    dist_to_target = loss.item()

            # save results
            self.db.add_run_data(id=run_idx,
                                 policy_parameters=policy_parameters,
                                 observations=observations,
                                 source_policy_idx=source_policy_idx,
                                 target_goal=target_goal,
                                 reached_goal=reached_goal,
                                 n_optim_steps_to_reach_goal=optim_step_idx,
                                 dist_to_target=dist_to_target)

            # add policy and reached goal into the libraries
            # do it after the run data is saved to not save them if there is an error during the saving
            self.policy_library.append(policy_parameters)
            self.goal_library = torch.cat([self.goal_library, reached_goal.reshape(1, -1).detach()])
            if torch.any(torch.isnan(reached_goal)):
                print('break')

            # append new discovery to train/valid dataset
            is_dead = torch.all(observations.states[-1] == 1) or torch.all(observations.states[-1] == 0)
            if not is_dead:
                if (train_loader.dataset.n_images + valid_loader.dataset.n_images) % 10 == 0:
                    valid_loader.dataset.images = torch.cat([valid_loader.dataset.images, observations.states[-1].unsqueeze(0).unsqueeze(0).cpu().detach()])
                    valid_loader.dataset.labels = torch.cat([valid_loader.dataset.labels, torch.tensor([-1]).unsqueeze(0)])
                    valid_loader.dataset.n_images += 1
                else:
                    train_loader.dataset.images = torch.cat([train_loader.dataset.images, observations.states[-1].unsqueeze(0).unsqueeze(0).cpu().detach()])
                    train_loader.dataset.labels = torch.cat([train_loader.dataset.labels, torch.tensor([-1]).unsqueeze(0)])
                    train_loader.dataset.n_images += 1


            # training stage
            if len(self.policy_library) % self.config.goalspace_training.frequency == 0:
                print(run_idx)
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

                # save updated results
                ## /!\ we dont save it in the outputs run_data_* files where the reached_goal corresponds to the one in the current goal space at the time of exploration
                self.goal_space.representation.eval()
                with torch.no_grad():
                    for run_idx in range(len(self.goal_library)):
                        self.goal_library[run_idx] = self.goal_space.map(self.db.runs[run_idx].observations).detach()

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

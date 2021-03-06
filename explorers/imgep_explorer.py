import torch
import random
import os
from addict import Dict
from evocraftsearch import Explorer
from evocraftsearch.spaces import BoxSpace
from evocraftsearch.utils import sample_value
import numbers
from tqdm import tqdm
from exputils.seeding import set_seed

class BoxGoalSpace(BoxSpace):
    def __init__(self, representation, autoexpand=True, low=0., high=0., shape=None, dtype=torch.float32):
        self.representation = representation
        self.autoexpand = autoexpand
        if shape is not None:
            if isinstance(shape, list) or isinstance(shape, tuple):
                assert len(shape) == 1 and shape[0] == self.representation.n_latents
            elif isinstance(shape, numbers.Number):
                assert shape == self.representation.n_latents
        BoxSpace.__init__(self, low=low, high=high, shape=(self.representation.n_latents,), dtype=dtype)


    def map(self, observations, **kwargs):
        embedding = self.representation.calc(observations, **kwargs)
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
        scale[torch.where(scale == 0.0)] = 1.0
        low = self.low.to(self.representation.config.device)
        # L2 by default
        embedding_a = (embedding_a - low) / scale
        embedding_b = (embedding_b - low) / scale
        dist = (embedding_a - embedding_b).pow(2).sum(-1).sqrt()
        return dist

    def sample(self):
        return BoxSpace.sample(self).to(self.representation.config.device)

    def save(self, filepath):
        torch.save(self, filepath)

    @staticmethod
    def load(filepath, map_location='cpu'):
        goal_space = torch.load(filepath, map_location='cpu')
        return goal_space




class IMGEPExplorer(Explorer):
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
        default_config.frequency_of_random_initialization = 10 # number of random runs at during exploration

        # Pi: source policy parameters config
        default_config.source_policy_selection = Dict()
        default_config.source_policy_selection.type = 'optimal'
        # default_config.source_policy_selection.type = 'kNN_elite'
        # default_config.source_policy_selection.k = 50

        # Opt: Optimizer to reach goal
        default_config.reach_goal_optim_steps = 10

        return default_config

    def __init__(self, system, explorationdb, goal_space, config={}, score_function=None, **kwargs):
        super().__init__(system=system, explorationdb=explorationdb, config=config, **kwargs)

        self.goal_space = goal_space

        # initialize policy library
        self.policy_library = []

        # initialize goal library
        self.goal_library = torch.empty((0,) + self.goal_space.shape).to(self.goal_space.representation.config.device)

        # save score function f(observations) = score
        self.score_function = score_function

        # initialize policy scores
        self.policy_scores = []


    def get_source_policy_idx(self, target_goal):

        if self.config.source_policy_selection.type == 'optimal':
            # get distance to other goals
            goal_distances = self.goal_space.calc_distance(target_goal, self.goal_library)

            # select goal with minimal distance
            isnan_distances = torch.isnan(goal_distances)
            if (~isnan_distances).sum().item() == 0:
                source_policy_idx = sample_value(('discrete', 0, len(self.goal_library) - 1))
            else:
                canditates = torch.where(goal_distances == goal_distances[~isnan_distances].min())[0]
                source_policy_idx = random.choice(canditates)

        elif self.config.source_policy_selection.type == "kNN_elite":
            # get distance to other goals
            goal_distances = self.goal_space.calc_distance(target_goal, self.goal_library)

            # select k closest reached goals
            isnan_distances = torch.isnan(goal_distances)
            if (~isnan_distances).sum().item() == 0:
                source_policy_idx = sample_value(('discrete', 0, len(self.goal_library) - 1))
            else:
                notnan_inds = torch.where(~isnan_distances)[0]
                notnan_distances = goal_distances[~isnan_distances]
                _, rel_candidates = notnan_distances.topk(min(self.config.source_policy_selection.k, len(notnan_distances)), largest=False)
                candidates = notnan_inds[rel_candidates]

                # select elite among those k
                candidate_scores = torch.tensor(self.policy_scores)[candidates]
                isnan_scores = torch.isnan(candidate_scores)
                source_policy_candidates = torch.where(candidate_scores == candidate_scores[~isnan_scores].max())[0]
                source_policy_idx = random.choice(candidates[source_policy_candidates])


        elif self.config.source_policy_selection.type == 'random':
            source_policy_idx = sample_value(('discrete', 0, len(self.goal_library) - 1))

        else:
            raise ValueError('Unknown source policy selection type {!r} in the configuration!'.format(
                self.config.source_policy_selection.type))

        return source_policy_idx

    def run(self, n_exploration_runs, continue_existing_run=False, save_frequency=None, save_filepath=None):

        print('Exploration: ')
        progress_bar = tqdm(total=n_exploration_runs)
        if continue_existing_run:
            run_idx = len(self.policy_library)
            progress_bar.update(run_idx)
        else:
            self.policy_library = []
            self.goal_library = torch.empty((0,) + self.goal_space.shape)
            run_idx = 0

        self.goal_library = self.goal_library.to(self.goal_space.representation.config.device)

        while run_idx < n_exploration_runs:

            set_seed(100000 * self.config.seed + run_idx)

            # Initial Random Sampling of Parameters
            if (run_idx < self.config.num_of_random_initialization) or (run_idx % self.config.frequency_of_random_initialization == 0):

                target_goal = None
                source_policy_idx = None

                policy_parameters = self.system.sample_policy_parameters()
                self.system.reset(policy=policy_parameters)

                with torch.no_grad():
                    observations = self.system.run()
                    reached_goal = self.goal_space.map(observations)
                    if self.score_function is not None:
                        policy_score = self.score_function.map(observations).item()
                    else:
                        policy_score = 0.0

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

                    train_losses = self.system.optimize(self.config.reach_goal_optim_steps, lambda obs: self.goal_space.calc_distance(target_goal, self.goal_space.map(obs)))
                    print(train_losses)
                    policy_parameters['initialization'] = self.system.initialization_parameters
                    policy_parameters['update_rule'] = self.system.update_rule_parameters

                with torch.no_grad():
                    observations = self.system.run()
                    reached_goal = self.goal_space.map(observations)
                    loss = self.goal_space.calc_distance(target_goal, reached_goal)
                    dist_to_target = loss.item()
                    if self.score_function is not None:
                        policy_score = self.score_function.map(observations).item()
                    else:
                        policy_score = 0.0

            # save results
            self.db.add_run_data(id=run_idx,
                                 policy_parameters=policy_parameters,
                                 observations=observations,
                                 source_policy_idx=source_policy_idx,
                                 target_goal=target_goal,
                                 reached_goal=reached_goal,
                                 dist_to_target=dist_to_target,
                                 policy_score=policy_score)

            if self.db.config.save_rollout_render:
                self.system.render_rollout(observations, filepath=os.path.join(self.db.config.db_directory,
                                                                                   f'run_{run_idx}_rollout'))


            # add policy and reached goal into the libraries
            # do it after the run data is saved to not save them if there is an error during the saving
            self.policy_library.append(policy_parameters)
            self.policy_scores.append(policy_score)
            self.goal_library = torch.cat([self.goal_library, reached_goal.reshape(1, -1)])

            if (save_frequency is not None) and (run_idx % save_frequency == 0):
                if (save_filepath is not None) and (os.path.exists(save_filepath)):
                    self.save(save_filepath)

            # increment run_idx
            run_idx += 1
            progress_bar.update(1)

    def save(self, filepath):
        self.goal_space.save(filepath+"goal_space.pickle")
        tmp_goal_space = self.goal_space
        self.goal_space = None
        tmp_score_function = self.score_function
        self.score_function = None
        Explorer.save(self, filepath+"explorer.pickle")
        self.goal_space = tmp_goal_space
        self.score_function = tmp_score_function
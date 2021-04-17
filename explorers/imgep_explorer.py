import torch
from addict import Dict
from evocraftsearch import Explorer
from evocraftsearch.spaces import BoxSpace
from evocraftsearch.utils import sample_value
import numbers
from tqdm import tqdm

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
        return self.representation.calc_distance(embedding_a, embedding_b, **kwargs)

    def sample(self):
        return BoxSpace.sample(self)


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
        if continue_existing_run:
            run_idx = len(self.policy_library)
            progress_bar.update(run_idx)
        else:
            self.policy_library = []
            self.goal_library = torch.empty((0,) + self.goal_space.shape)
            run_idx = 0
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
                                                                {'params': self.system.potential_ca_step.K.parameters(), **self.config.reach_goal_optimizer.potential_ca_step.K.parameters}],
                                                                **self.config.reach_goal_optimizer.parameters
                                                                )
                    for optim_step_idx in tqdm(range(1, self.config.reach_goal_optimizer.optim_steps)):

                        print(f'Run {run_idx}, optimisation toward goal: ')

                        # run system with IMGEP's policy parameters
                        observations = self.system.run()
                        reached_goal = self.goal_space.map(observations)

                        # compute error between reached_goal and target_goal
                        loss = self.goal_space.calc_distance(target_goal, reached_goal)
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
            self.goal_library = torch.cat([self.goal_library, reached_goal.reshape(1, -1)])

            # increment run_idx
            run_idx += 1
            progress_bar.update(1)

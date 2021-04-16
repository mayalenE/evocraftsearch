from addict import Dict
from morphosearch.core import Explorer
from tqdm import tqdm


class RandomExplorer(Explorer):
    """Performs random explorations of a system."""

    def run(self, n_exploration_runs):

        print('Exploration: ')
        for run_idx in tqdm(range(n_exploration_runs)):

            if run_idx not in self.data:
                policy_parameters = Dict.fromkeys(
                    ['initialization', 'update_rule'])  # policy parameters (output of IMGEP policy)

                policy_parameters['initialization'] = self.system.initialization_space.sample()
                policy_parameters['update_rule'] = self.system.update_rule_space.sample()

                observations = self.system.run(initialization_parameters=policy_parameters['initialization'],
                                               update_rule_parameters=policy_parameters['update_rule'])

                # save results
                self.db.add_run_data(id=run_idx,
                                     run_parameters=policy_parameters,
                                     observations=observations)

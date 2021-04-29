import torch
from morphosearch.core import Explorer
from tqdm import tqdm


class RandomExplorer(Explorer):
    """Performs random explorations of a system."""

    def run(self, n_exploration_runs):

        print('Exploration: ')
        for run_idx in tqdm(range(n_exploration_runs)):

            if run_idx not in self.data:
                policy_parameters = self.system.sample_policy_parameters()
                self.system.reset(policy=policy_parameters)

                with torch.no_grad():
                    observations = self.system.run()

                # save results
                self.db.add_run_data(id=run_idx,
                                     policy_parameters=policy_parameters,
                                     observations=observations)

                if self.db.config.save_gifs:
                    self.system.render_rollout(observations, filepath=os.path.join(self.db.config.db_directory,
                                                                                   f'run_{run_idx}_rollout'))

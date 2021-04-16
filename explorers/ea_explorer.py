import neat
from evocraftsearch import Explorer


class EAExplorer(Explorer):
    """Performs NEAT+SGD optimisation of a system."""

    @staticmethod
    def default_config():
        default_config = Explorer.default_config()

        # Pi: policy parameters config
        default_config.policy_parameters = []  # config to init and mutate the run parameters

        return default_config

    def eval_genomes(self, genomes, neat_config):
        raise NotImplementedError

    def run(self, n_generations, neat_config):
        pop = neat.Population(neat_config)
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        reporter = neat.StdOutReporter(True)
        pop.add_reporter(reporter)

        pop.run(self.eval_genomes, n_generations)

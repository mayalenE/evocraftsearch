from copy import deepcopy
from evocraftsearch.spaces import Space

class CPPNSpace(Space):

    def __init__(self, neat_config):
        self.neat_config = neat_config

        super().__init__(shape=None, dtype=None)

    def sample(self):
        genome = self.neat_config.genome_type(0)
        genome.configure_new(self.neat_config.genome_config)
        return genome

    def mutate(self, genome):
        new_genome = deepcopy(genome)
        new_genome.mutate(self.neat_config.genome_config)
        return new_genome

    def crossover(self, genome_1, genome_2):
        genome_1 = deepcopy(genome_1)
        genome_2 = deepcopy(genome_2)
        if genome_1.fitness is None:
            genome_1.fitness = 0.0
        if genome_2.fitness is None:
            genome_2.fitness = 0.0
        child_1 = self.neat_config.genome_type(0)
        child_1.configure_crossover(genome_1, genome_2, self.neat_config.genome_config)
        child_2 = self.neat_config.genome_type(0)
        child_2.configure_crossover(genome_1, genome_2, self.neat_config.genome_config)
        return child_1, child_2

    def contains(self, x):
        # TODO from neat config max connections/weights
        return True

    def clamp(self, x):
        # TODO from neat config max connections/weights
        return x
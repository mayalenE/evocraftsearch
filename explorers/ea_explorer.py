from addict import Dict
from evocraftsearch import Explorer
from deap import creator, base, tools
from tqdm import tqdm
import torch

class EAExplorer(Explorer):
    """
    Basic EA explorer which uses a simple GA. Subclass it for using more complex algorithms.
    The individual sampling, mutation and crossover spaces are defined by the system (with internal probabilities for mutation and crossover).
    """

    # Set these in ALL subclasses
    goal_space = None  # defines the obs->goal representation and the goal sampling strategy (self.goal_space.sample())
    fitness_optimizer = None

    @staticmethod
    def default_config():
        default_config = Dict()
        # base config

        # EA:
        default_config.population_size = 50
        default_config.selection_size = 10

        # Opt: Optimizer to reach goal
        default_config.fitness_optim_steps = 10

        return default_config


    def __init__(self, system, explorationdb, output_fitness, config={}, **kwargs):
        super().__init__(system=system, explorationdb=explorationdb, config=config, **kwargs)

        self.output_fitness = output_fitness

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.system.sample_policy_parameters)
        self.toolbox.register("mate", self.system.crossover_policy_parameters)
        self.toolbox.register("mutate", self.system.mutate_policy_parameters)

        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual
        )

        self.toolbox.register("select", tools.selTournament, tournsize=self.config.population_size)

        self.toolbox.register("evaluate", self.evaluate_policy_parameters)


    def evaluate_policy_parameters(self, ind_id, ind_policy):

        # system rollout with the individual's policy
        self.system.reset(ind_policy)

        # Optimization toward target goal
        if isinstance(self.system, torch.nn.Module) and self.config.fitness_optim_steps > 0:

            train_losses = self.system.optimize(self.config.fitness_optimizer.optim_steps, lambda obs: - self.output_fitness.calc(obs))
            ind_policy['initialization'] = self.system.initialization_parameters
            ind_policy['update_rule'] = self.system.update_rule_parameters
            fitness = - train_losses[-1]

        else:
            with torch.no_grad():
                observations = self.system.run()
                fitness = self.output_fitness.calc(observations).item()

        # update ind_policy fitness
        ind_policy["fitness"] = fitness

        # store results in the database
        self.db.add_run_data(id=ind_id,
                             policy_parameters=ind_policy,
                             observations=observations,
                             fitness=fitness)
        return ind_policy


    def run(self, n_exploration_runs, continue_existing_run=False):
        # n_exploration_runs is number of generations
        # self.config.population_size is number of individuals per generation

        print('Exploration: ')
        progress_bar = tqdm(total=n_exploration_runs)
        if continue_existing_run:
            run_idx = len(self.policy_library)
            progress_bar.update(run_idx)
        else:
            self.population = self.toolbox.population(self.config.population_size)
            self.policy_library = []
            run_idx = 0

        while run_idx < n_exploration_runs:

            print(f"generation number {run_idx}")
            # select the next generation
            offspring = self.toolbox.select(self.population, self.config.selection_size)

            # clone selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # apply crossover
            print("apply crossover")
            for i, (x1, x2) in enumerate(zip(offspring[::2], offspring[1::2])):
                child1, child2 = self.toolbox.mate(x1, x2)
                child1.fitness = None
                child2.fitness = None
                offspring[2*i] = child1
                offspring[2*i+1] = child2


            # apply mutation
            print("apply mutation")
            for i, x in enumerate(offspring):
                mutant = self.toolbox.mutate(x)
                mutant.fitness = None
                offspring[i] = mutant

            # evaluate new individuals
            print("evaluate individuals")
            for ind_idx, ind_policy in enumerate(offspring):
                if ind_policy.fitness is not None:
                    continue
                else:
                    ind_policy = self.toolbox.evaluate(ind_idx, ind_policy)
                    offspring[ind_idx] = ind_policy

            # update population and pass to next generation
            self.population[:] = offspring
            run_idx += 1
            progress_bar.update(1)

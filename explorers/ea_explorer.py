from addict import Dict
from copy import deepcopy
from evocraftsearch import Explorer
from tqdm import tqdm
import torch
import random
import sys

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
        default_config.tournament_size = 5

        # Opt: Optimizer to reach goal
        default_config.fitness_optim_steps = 10

        return default_config


    def __init__(self, system, explorationdb, output_fitness, config={}, **kwargs):
        super().__init__(system=system, explorationdb=explorationdb, config=config, **kwargs)

        self.output_fitness = output_fitness


    def select_tournament(self, tournsize):
        """Select the best individual among *tournsize* randomly chosen
        individuals from the population, *len(population)* times. The list returned contains
        references to the input *individuals*.
        :param k: The number of individuals to select.
        :param tournsize: The number of individuals participating in each tournament.
        """
        chosen = []
        for _ in range(len(self.population)):
            aspirants = [random.choice(self.population) for _ in range(tournsize)]
            max_fitness = max([aspirant.fitness for aspirant in aspirants])
            selected_aspirants = [aspirant for aspirant in aspirants if aspirant.fitness == max_fitness]
            selected_individual = random.choice(selected_aspirants)
            chosen.append(deepcopy(selected_individual))
        return chosen


    def evaluate_policy_parameters(self, ind_policy):

        # system rollout with the individual's policy
        self.system.reset(ind_policy)

        # Optimization toward target goal
        if isinstance(self.system, torch.nn.Module) and self.config.fitness_optim_steps > 0:

            train_losses = self.system.optimize(self.config.fitness_optim_steps, lambda obs: - self.output_fitness.calc(obs))
            print(train_losses)
            ind_policy['initialization'] = self.system.initialization_parameters
            ind_policy['update_rule'] = self.system.update_rule_parameters

        with torch.no_grad():
            observations = self.system.run()
            fitness = self.output_fitness.calc(observations).item()

        return ind_policy, observations, fitness


    def run(self, n_exploration_runs, continue_existing_run=False):
        # n_exploration_runs is number of generations
        # self.config.population_size is number of individuals per generation

        print('Exploration: ')
        progress_bar = tqdm(total=n_exploration_runs)
        if continue_existing_run:
            run_idx = len(self.policy_library)
            progress_bar.update(run_idx)
        else:
            self.population = []
            for ind_idx in range(self.config.population_size):
                ind_policy = self.system.sample_policy_parameters()
                ind = Dict(idx=ind_idx, policy=ind_policy, fitness=sys.float_info.max) #positive exploration by encouraging all members to be tested
                self.population.append(ind)
            self.policy_library = []
            run_idx = 0


        while run_idx < n_exploration_runs:

            print(f"generation number {run_idx}")
            # select the next generation
            offspring = self.select_tournament(tournsize=self.config.tournament_size)

            # apply crossover
            print("apply crossover")
            for i, (x1, x2) in enumerate(zip(offspring[::2], offspring[1::2])):
                child1_policy, child2_policy = self.system.crossover_policy_parameters(x1.policy, x2.policy)
                offspring[2*i].policy = child1_policy
                offspring[2*i+1].policy = child2_policy


            # apply mutation
            print("apply mutation")
            for i, x in enumerate(offspring):
                mutant_policy = self.system.mutate_policy_parameters(x.policy)
                offspring[i].policy = mutant_policy

            # evaluate new individuals
            print("evaluate individuals")
            for relative_idx, individual in enumerate(offspring):
                if run_idx >= n_exploration_runs:
                    break
                ind_policy, observations, fitness = self.evaluate_policy_parameters(individual.policy)
                # append to policy library and to database
                self.policy_library.append(ind_policy)
                self.db.add_run_data(id=run_idx,
                                     source_policy_idx=individual.idx,
                                     policy_parameters=ind_policy,
                                     observations=observations,
                                     fitness=fitness)
                evaluated_individual = Dict(idx=run_idx, policy=ind_policy, fitness=fitness)
                offspring[relative_idx] = evaluated_individual
                run_idx += 1
                progress_bar.update(1)


            # The population is entirely replaced by the offspring
            self.population[:] = offspring


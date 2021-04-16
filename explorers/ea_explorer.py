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
        default_config.fitness_optimizer = Dict()
        default_config.fitness_optimizer.optim_steps = 10
        default_config.fitness_optimizer.name = "Adam"
        default_config.fitness_optimizer.initialization_cppn.parameters.lr =  1e-3
        default_config.fitness_optimizer.cppn_potential_ca_step.K.parameters.lr = 1e-2

        return default_config


    def __init__(self, system, explorationdb, output_fitness, config={}, **kwargs):
        super().__init__(system=system, explorationdb=explorationdb, config=config, **kwargs)

        self.output_fitness = output_fitness

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register(
            "individual",
            self.generate_policy_parameters,
            #creator.Individual,
        )

        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual
        )

        self.toolbox.register("select", tools.selTournament, tournsize=self.config.population_size)
        self.toolbox.register("mate", self.crossover_policy_parameters)
        self.toolbox.register("mutate", self.mutate_policy_parameters)
        self.toolbox.register("evaluate", self.evaluate_policy_parameters)


    def generate_policy_parameters(self):
        policy = Dict()
        policy['initialization'] = self.system.initialization_space.sample()
        policy['update_rule'] = self.system.update_rule_space.sample()
        policy['fitness'] = 0.0
        return policy

    def crossover_policy_parameters(self, policy_1, policy_2):
        child_1_policy, child_2_policy = Dict(), Dict()
        child_1_policy['initialization'], child_2_policy['initialization'] = self.system.initialization_space.crossover(policy_1['initialization'], policy_2['initialization'])
        child_1_policy['update_rule'], child_2_policy['update_rule'] = self.system.update_rule_space.crossover(policy_1['update_rule'], policy_2['update_rule'])
        return child_1_policy, child_2_policy

    def mutate_policy_parameters(self, policy):
        new_policy = Dict()
        new_policy['initialization'] = self.system.initialization_space.mutate(policy['initialization'])
        new_policy['update_rule'] = self.system.update_rule_space.mutate(policy['update_rule'])
        return new_policy


    def evaluate_policy_parameters(self, policy):
        individual_idx = 0

        # system rollout with the individual's policy
        self.system.reset(initialization_parameters=policy['initialization'],
                          update_rule_parameters=policy['update_rule'])

        # Optimization toward target goal
        if isinstance(self.system, torch.nn.Module) and self.config.fitness_optimizer.optim_steps > 0:

            optimizer_class = eval(f'torch.optim.{self.config.fitness_optimizer.name}')
            self.fitness_optimizer = optimizer_class([{'params': self.system.initialization_cppn.parameters(),
                                                          **self.config.fitness_optimizer.initialization_cppn.parameters},
                                                         {'params': self.system.cppn_potential_ca_step.K.parameters(),
                                                          **self.config.fitness_optimizer.cppn_potential_ca_step.K.parameters}],
                                                        **self.config.fitness_optimizer.parameters)
            
            for optim_step_idx in tqdm(range(1, self.config.fitness_optimizer.optim_steps)):

                # run system with IMGEP's policy parameters
                observations = self.system.run()

                # compute error between reached_goal and target_goal
                fitness = self.output_fitness.calc(observations)
                loss = -fitness
                print(f'step {optim_step_idx}: fitness={fitness.item():0.2f}')

                # optimisation step
                self.fitness_optimizer.zero_grad()
                loss.backward()
                self.fitness_optimizer.step()

                if optim_step_idx > 5 and abs(old_loss - loss.item()) < 1e-4:
                    break
                old_loss = loss.item()

            # gather back the trained parameters
            self.system.update_initialization_parameters()
            self.system.update_update_rule_parameters()
            policy['initialization'] = self.system.initialization_parameters
            policy['update_rule'] = self.system.update_rule_parameters


        else:
            with torch.no_grad():
                observations = self.system.run()
                fitness = self.output_fitness.calc(observations)
                optim_step_idx = 0

        # save results in the database
        fitness = fitness.item()
        if fitness > 2.5:
            self.system.render()
        self.db.add_run_data(id=individual_idx,
                             policy_parameters=policy,
                             observations=observations,
                             fitness=fitness,
                             n_optim_steps=optim_step_idx)
        return fitness


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
            invalid_inds = [ind for ind in offspring if ind.fitness is None]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_inds)
            for ind, fit in zip(invalid_inds, fitnesses):
                ind.fitness = fit


            # update population and pass to next generation
            self.population[:] = offspring
            run_idx += 1
            progress_bar.update(1)

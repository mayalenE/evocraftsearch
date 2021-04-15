import warnings

import torch
from addict import Dict
from evocraftsearch import ExplorationDB
from evocraftsearch.spaces import DictSpace


# Inspired from OpenAI's Gym API: https://github.com/openai/gym/blob/master/gym/core.py

class System:
    """The main System class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.
    The main API methods that users of this class need to know are:
        reset
        step
        render
        close
        #TODO: seed
     And set the following attributes:
        initialization_space: The DictSpace object corresponding to valid system initialisation genome's parameters
        update_rule_space: The DictSpace object corresponding to valid system update rule genome's parameters
        intervention_space: The DictSpace object corresponding to valid system intervention parameters
    """

    # Set these in ALL subclasses
    initialization_space = DictSpace()
    update_rule_space = DictSpace()
    intervention_space = DictSpace()

    @staticmethod
    def default_config():
        default_config = Dict()
        return default_config

    def __init__(self, config={}, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        # set randomly the system's parameters
        if self.initialization_space is not None:
            self._initialization_parameters = self.initialization_space.sample()
        else:
            self._initialization_parameters = None
        if self.update_rule_space is not None:
            self._update_rule_parameters = self.update_rule_space.sample()
        else:
            self._update_rule_parameters = None
        if self.intervention_space is not None:
            self._intervention_parameters = self.intervention_space.sample()
        else:
            self._intervention_parameters = None

    @property
    def initialization_parameters(self):
        return self._initialization_parameters

    @property
    def update_rule_parameters(self):
        return self._update_rule_parameters

    @property
    def intervention_parameters(self):
        return self._intervention_parameters

    @initialization_parameters.setter
    def initialization_parameters(self, new_initialization_parameters):
        if not self.initialization_space.contains(new_initialization_parameters):
            new_initialization_parameters = self.initialization_space.clamp(new_initialization_parameters)
            warnings.warn('provided parameters are not in the space range and are therefore clamped')
        self._initialization_parameters = new_initialization_parameters

    @update_rule_parameters.setter
    def update_rule_parameters(self, new_update_rule_parameters):
        if not self.update_rule_space.contains(new_update_rule_parameters):
            new_update_rule_parameters = self.update_rule_space.clamp(new_update_rule_parameters)
            warnings.warn('provided parameters are not in the space range and are therefore clamped')
        self._update_rule_parameters = new_update_rule_parameters

    @intervention_parameters.setter
    def intervention_parameters(self, new_intervention_parameters):
        if not self.intervention_space.contains(new_intervention_parameters):
            new_intervention_parameters = self.intervention_space.clamp(new_intervention_parameters)
            warnings.warn('provided parameters are not in the space range and are therefore clamped')
        self._intervention_parameters = new_intervention_parameters

    def reset(self, initialization_parameters=None, update_rule_parameters=None):
        """Resets the environment to an initial state and returns an initial
        observation.
        Args:
            initialization_parameters (Dict): the input parameters of the system provided by the agent
            update_rule_parameters (Dict): the parameters of the system's update rule provided by the agent
        Returns:
            observation (Dict): the initial observation.
        """
        raise NotImplementedError

    def step(self, intervention_parameters=None):
        """Run one timestep of the system's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            :param intervention_parameters: an action provided by the agent
        Returns:
            observation (Dict): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (Dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError

    def render(self, **kwargs):
        """Renders the environment.
        """
        raise NotImplementedError

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    def save(self, filepath):
        """
        Saves the system object using torch.save function in pickle format
        Can be used if the system state's change over exploration and we want to dump it
        """
        torch.save(self, filepath)


class OutputRepresentation:
    """ Base class to map the observations of a system to an embedding vector (BC characterization)
    """

    @staticmethod
    def default_config():
        default_config = Dict()
        return default_config

    def __init__(self, config={}, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

    def calc(self, observations, **kwargs):
        """ Maps the observations of a behavioral descriptor
            Args:
                observations (Dict): observations received after one environment run
            Returns
                embeddings (Dict): generally vector but we might need Dict structures, for instance for IMGEP-HOLMES
        """
        raise NotImplementedError

    def calc_distance(self, embedding_a, embedding_b, **kwargs):
        """ Compute the distance between 2 embedding
        """
        raise NotImplementedError


class OutputFitness:
    """ Base class to map the observations of a system to a behavioral descriptor
    """

    @staticmethod
    def default_config():
        default_config = Dict()
        return default_config

    def __init__(self, config={}, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

    def calc(self, observations, **kwargs):
        """ Maps the observations of a system to a fitness score
            Args:
                observations (Dict): observations received after one environment run
            Returns
                embeddings (torch.Tensor): generally vector but we might need Dict structures, for instance for IMGEP-HOLMES
        """
        raise NotImplementedError


class Explorer:
    """
    Base class for exploration experiments.
    Allows to save and load exploration results
    """

    @staticmethod
    def default_config():
        default_config = Dict()
        return default_config

    def __init__(self, system, explorationdb, config={}, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        self.system = system
        self.db = explorationdb

    def save(self, filepath):
        """
        Saves the explorer object using torch.save function in pickle format
        /!\ We intentionally empty explorer.db from the pickle
        because the database is already automatically saved in external files each time the explorer call self.db.add_run_data
        """
        # do not pickle the data as already saved in extra files
        tmp_data = self.db
        self.db.reset_empty_db()

        # pickle exploration object
        torch.save(self, filepath)

        # attach db again to the exploration object
        self.db = tmp_data

    @staticmethod
    def load(explorer_filepath, load_data=True, run_ids=None, map_location='cuda'):

        explorer = torch.load(explorer_filepath, map_location=map_location)

        # loop over policy parameters to coalesce sparse tensors (not coalesced by default)
        def coalesce_parameter_dict(d, has_coalesced_tensor=False):
            for k, v in d.items():
                if isinstance(v, Dict):
                    d[k], has_coalesced_tensor = coalesce_parameter_dict(v, has_coalesced_tensor=has_coalesced_tensor)
                elif isinstance(v, torch.Tensor) and v.is_sparse and not v.is_coalesced():
                    d[k] = v.coalesce()
                    has_coalesced_tensor = True
            return d, has_coalesced_tensor

        for policy_idx, policy in enumerate(explorer.policy_library):
            explorer.policy_library[policy_idx], has_coalesced_tensor = coalesce_parameter_dict(policy)
            if not has_coalesced_tensor:
                break

        if load_data:
            explorer.db = ExplorationDB(config=explorer.db.config)
            explorer.db.load(run_ids=run_ids, map_location=map_location)

        return explorer

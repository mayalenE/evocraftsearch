import os
import re
import warnings
from collections import OrderedDict
from glob import glob
from copy import deepcopy
import torch
from addict import Dict
from tqdm import tqdm

class ExplorationDB:
    """
    Base of all Database classes.
    """

    @staticmethod
    def default_config():

        default_config = Dict()
        default_config.db_directory = "database"
        default_config.save_observations = True
        default_config.memory_size_run_data = 'infinity'  # number of runs that are kept in memory: 'infinity' - no imposed limit, int - number of runs saved in memory
        default_config.load_observations = True  # if set to false observations are not loaded in the load() function

        return default_config

    def __init__(self, config={}, **kwargs):

        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        if self.config.memory_size_run_data != 'infinity':
            assert isinstance(self.config.memory_size_run_data,
                              int) and self.config.memory_size_run_data > 0, "config.memory_size_run_data must be set to infinity or to an integer >= 1"

        self.reset_empty_db()

    def reset_empty_db(self):
        self.runs = OrderedDict()
        self.run_ids = set()  # list with run_ids that exist in the db
        self.run_data_ids_in_memory = []  # list of run_ids that are hold in memory

    def get_run_data(self, run_id, map_location=None):

        try:
            return self.runs[run_id]

        except:
            if run_id not in self.run_ids:
                raise KeyError(f'Run with ID {run_id} does not exists in the database!')

            elif run_id not in self.run_data_ids_in_memory:
                # load from database but do not add it to memory
                filename = 'run_{:07d}_data.pickle'.format(run_id)
                filepath = os.path.join(self.config.db_directory, filename)

                if os.path.exists(filepath):
                    run_data_kwargs = torch.load(filepath, map_location=map_location)
                else:
                    run_data_kwargs = {'id': None, 'policy_parameters': None}

                if self.config.load_observations:
                    filename_obs = 'run_{:07d}_observations.pickle'.format(run_id)
                    filepath_obs = os.path.join(self.config.db_directory, filename_obs)

                    # load observations
                    if os.path.exists(filepath_obs):
                        observations = torch.load(filepath_obs, map_location=map_location)
                    else:
                        observations = None
                else:
                    observations = None

                # create run data and add it to memory
                run_data = Dict(observations=observations, **run_data_kwargs)

                return run_data


    def add_run_data(self, id, policy_parameters, observations, **kwargs):

        run_data_entry = Dict(db=self, id=id, policy_parameters=policy_parameters, observations=observations, **kwargs)
        if id not in self.run_ids:
            self.add_run_data_to_memory(id, run_data_entry)
            self.run_ids.add(id)

        else:
            warnings.warn(f'/!\ id {id} already in the database: overwriting it with new run data !!!')
            self.add_run_data_to_memory(id, run_data_entry, replace_existing=True)

        self.save([id])  # TODO: modify if we do not want to automatically save after each run


    def add_run_data_to_memory(self, id, run_data, replace_existing=False):
        self.runs[id] = run_data
        if not replace_existing:
            self.run_data_ids_in_memory.insert(0, id)

        # remove last item from memory when not enough size
        if self.config.memory_size_run_data != 'infinity' and len(self.run_data_ids_in_memory) > self.config.memory_size_run_data:
            del (self.runs[self.run_data_ids_in_memory[-1]])
            del (self.run_data_ids_in_memory[-1])


    def save(self, run_ids=None):
        # the run data entry is save in 2 files: 'run_*_data*' (general data dict such as run parameters -> for now json) and ''run_*_observations*' (observation data dict -> for now npz)
        if run_ids is None:
            run_ids = []

        for run_id in run_ids:
            self.save_run_data_to_db(run_id)
            if self.config.save_observations:
                self.save_observations_to_db(run_id)


    def save_run_data_to_db(self, run_id):
        run_data = self.runs[run_id]

        # add all data besides the observations
        save_dict = dict()
        for data_name, data_value in run_data.items():
            if data_name not in ['observations', 'db']:
                save_dict[data_name] = data_value
        filename = 'run_{:07d}_data.pickle'.format(run_id)
        filepath = os.path.join(self.config.db_directory, filename)

        torch.save(save_dict, filepath)

    def save_observations_to_db(self, run_id):
        run_data = self.runs[run_id]

        filename = 'run_{:07d}_observations.pickle'.format(run_id)
        filepath = os.path.join(self.config.db_directory, filename)

        torch.save(run_data.observations, filepath)


    def load(self, run_ids=None, map_location="cpu"):
        """
        Loads the data base.
        :param run_ids:  IDs of runs for which the data should be loaded into the memory.
                        If None is given, all ids are loaded (up to the allowed memory size).
        :param map_location: device on which the database is loaded
        """

        if run_ids is not None:
            assert isinstance(run_ids, list), "run_ids must be None or a list"

        self.runs = OrderedDict()
        self.run_data_ids_in_memory = []

        if run_ids is None:
            run_ids = self.load_run_ids_from_db() # set run_ids from the db directory and empty memory
        else:
            run_ids = set(run_ids)

        self.run_ids = run_ids

        if len(run_ids) > 0:

            if self.config.memory_size_run_data != 'infinity' and len(run_ids) > self.config.memory_size_run_data:
                # only load the maximum number of run_data into the memory
                run_ids_to_load_from_db = list(deepcopy(run_ids))[-self.config.memory_size_run_data:]

            else:
                run_ids_to_load_from_db = deepcopy(run_ids)

            self.load_run_data_from_db(run_ids_to_load_from_db, map_location=map_location)


    def load_run_ids_from_db(self):
        run_ids = set()

        file_matches = glob(os.path.join(self.config.db_directory, 'run_*_data*'))
        for match in file_matches:
            id_as_str = re.findall('_(\d+).', match)
            if len(id_as_str) > 0:
                run_ids.add(int(id_as_str[
                                    -1]))  # use the last find, because ther could be more number in the filepath, such as in a directory name

        return run_ids

    def load_run_data_from_db(self, run_ids, map_location="cpu"):
        """Loads the data for a list of runs and adds them to the memory."""

        if not os.path.exists(self.config.db_directory):
            raise Exception('The directory {!r} does not exits! Cannot load data.'.format(self.config.db_directory))

        print('Loading Data: ')
        for run_id in tqdm(run_ids):

            # load general data (run parameters and others)
            filename = 'run_{:07d}_data.pickle'.format(run_id)
            filepath = os.path.join(self.config.db_directory, filename)

            if os.path.exists(filepath):
                run_data_kwargs = torch.load(filepath, map_location=map_location)
            else:
                run_data_kwargs = {'id': None, 'policy_parameters': None}

            if self.config.load_observations:
                filename_obs = 'run_{:07d}_observations.pickle'.format(run_id)
                filepath_obs = os.path.join(self.config.db_directory, filename_obs)

                # load observations
                if os.path.exists(filepath_obs):
                    observations = torch.load(filepath_obs, map_location=map_location)
                else:
                    observations = None
            else:
                observations = None

            # create run data and add it to memory
            run_data = Dict(observations=observations, **run_data_kwargs)
            self.add_run_data_to_memory(run_id, run_data)


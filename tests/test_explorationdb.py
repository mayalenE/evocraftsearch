import os
from unittest import TestCase

import numpy as np
from addict import Dict
from evocraftsearch import ExplorationDB


class TestExplorationDB(TestCase):

    def test_explorationdb(self):
        db_config = Dict()
        db_config.db_directory = '/tmp/exploration_db'
        db_config.save_observations = True
        db_config.load_observations = True

        if not os.path.exists(db_config.db_directory):
            os.makedirs(db_config.db_directory)

        n_runs = 10

        for test_idx in range(10):
            db_config.keep_saved_runs_in_memory = bool(np.random.randint(0, 1))
            if np.random.rand() < 0.3:
                db_config.memory_size_run_data = 'infinity'
            else:
                db_config.memory_size_run_data = np.random.randint(1, n_runs + 1)

            db = ExplorationDB(config=db_config)

            for run_id in range(n_runs):
                policy_parameters = Dict()
                policy_parameters.R = np.random.randint(1, 10)
                policy_parameters.T = np.random.randint(1, 100)
                policy_parameters.cppn_params.sigma = np.random.rand()

                observations = Dict()
                observations.graph_states = [np.random.rand(4, 2) for i in range(10)]
                observations.env_frames = [np.random.rand(256, 256) for i in range(10)]

                if run_id < 10:  # random exploration
                    db.add_run_data(id=run_id,
                                    policy_parameters=policy_parameters,
                                    observations=observations,
                                    target_goal=None,
                                    source_policy_idx=None)
                else:  # goal based exploration
                    db.add_run_data(id=run_id, policy_parameters=policy_parameters, observations=observations,
                                    target_goal=np.random.rand(2), source_policy_idx=np.random.randint(0, run_id))

            # print summary:
            assert db.run_ids.__len__() == n_runs, "Test Error: all the runs where not added to the database"
            if not db_config.keep_saved_runs_in_memory:
                assert db.runs.__len__() == 0, "Test Error: len of run ids in memory > 0"
            else:
                if db_config.memory_size_run_data == "infinity":
                    assert db.runs.__len__() == n_runs, "Test Error: all runs should be stored in memory"
                elif isinstance(db_config.memory_size_run_data, int):
                    assert db.runs.__len__() == db_config.memory_size_run_data, "Test Error: the number of runs stored in memory is not corresponding to the one in the config"

            # erase db and reload from files WITH observations
            del db
            db = ExplorationDB(config=db_config)
            db.load()
            # print summary:
            assert db.run_ids.__len__() == n_runs, "Test Error: all the runs where not added to the database"
            if not db_config.keep_saved_runs_in_memory:
                assert db.runs.__len__() == 0, "Test Error: len of run ids in memory > 0"
            else:
                if db_config.memory_size_run_data == "infinity":
                    assert db.runs.__len__() == n_runs, "Test Error: all runs should be stored in memory"
                elif isinstance(db_config.memory_size_run_data, int):
                    assert db.runs.__len__() == db_config.memory_size_run_data, "Test Error: the number of runs stored in memory is not corresponding to the one in the config"

            # erase db and reload from files WITHOUT observations
            del db
            db_config.load_observations = False
            db = ExplorationDB(config=db_config)
            db.load()
            # print summary:
            assert db.run_ids.__len__() == n_runs, "Test Error: all the runs where not added to the database"
            if not db_config.keep_saved_runs_in_memory:
                assert db.runs.__len__() == 0, "Test Error: len of run ids in memory > 0"
            else:
                if db_config.memory_size_run_data == "infinity":
                    assert db.runs.__len__() == n_runs, "Test Error: all runs should be stored in memory"
                elif isinstance(db_config.memory_size_run_data, int):
                    assert db.runs.__len__() == db_config.memory_size_run_data, "Test Error: the number of runs stored in memory is not corresponding to the one in the config"

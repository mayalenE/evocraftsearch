# Table of Contents

* [Install and Run](#evocraftsearch-install-and-run)
  
* [Code Skeleton](#evocraftsearch-code-skeleton)
  

---

# EvocraftSearch: Install and Run

## Step 1: Installation
1. If you do not already have it, please install [Conda](https://www.anaconda.com/)
2. Create *autodisc* conda environment: `conda create --name autodisc python=3.8`
3. Activate *autodisc* conda environment: `conda activate autodisc`
4. If you do not already have it, please create a package folder that you will link to your conda env: `mkdir <path_to_packages_folder>`
5. Into your package folder, clone the following packages:  
    a. `git clone git@github.com:mayalenE/imagerepresentation.git`  
    b. `git clone git@github.com:mayalenE/pytorchneat.git`   
    c. `git clone git@github.com:mayalenE/evocraftsearch.git`
5. Include thos packages in the conda environment:  
   `echo "<path_to_packages_folder>" > "$HOME/miniconda3/envs/autodisc/lib/python3.8/site-packages/my_packages.pth"`
6. Install the required conda packages in the environment (*requirements.txt* file can be found in evocraftsearch directory):  
   `while read requirement; do conda install --yes $requirement --channel default --channel anaconda --channel conda-forge --channel pytorch || pip install $requirement; done < requirements.txt`


## Step 2: Run the experiments
You can reproduce the experiments by going in the *examples* folder of the evocraftsearch repository and do:
```buildoutcfg
python run_experiment.py
```
You can change the experiment configuration by selecting one of the proposed configs (`experiment_config_64_2D_growth.py`, etc)
or by changing the hyperparameters yourself in the config file.


# EvocraftSearch: Code Skeleton
The general structure of the code is inspired from [OpenAI's Gym](https://github.com/openai/gym/tree/master/gym) library.  
The main classes (***System***, ***OutputRepresentation***, ***OutputFitness*** and ***Explorer***) are implemented in `core.py`.


## evocraftsearch.spaces.Space

Class used to define valid spaces for (i) input parameters of the Evocraft systems and (ii) IMGEP explorer goal space.
Implemented space classes are: BoxSpace, DiscreteSpace, MultiDiscreteSpace, MultiBinarySpace and CppnSpace.  
Implemented container classes are: TupleSpace anb DictSpace.  
Most use-cases should be covered by those classes but custom Space can be defined and must implement the following API methods:
* **sample**(self) - randomly sample an element of this space
* **mutate**(self, x) - randomly mutate an element of this space
* **crossover**(self, x1, x2) - randomly crossover two elements of this space
* **contains**(self, x) - return boolean specifying if x is a valid
* **clamp**(self, x) - return a valid clamped value of x inside space's bounds


## evocraftsearch.ExplorationDB

Class used to store and save data runs of an exploration

* **config**
    * **db_directory** - directory into which the ExplorationDB will save the *run_\*_data.pickle* and *run_\*_
      observations.pickle* files
    * save_observations - boolean which indicates if observations should be saved (i.e. if *run_\*_observations.pickle*
      files should be generated)
    * keep_saved_runs_in_memory - boolean which indicates if run data entries should be kept in memory after being saved
      to external files
    * memory_size_run_data - integer which indicates the maximum number of run data entries that are stored in memory
    * load_observations - boolean which indicates if observations should be loaded in memory when loading from files
* run_ids - ids of all runs in the database
* run_data_ids_in_memory - is of runs stored in memory
* **runs** - Ordered dictionary with single exploration runs (keys: ids, values: run_data), a run_data is an Dict with:
    * id - integer identifer of the run
        * **policy_parameters** - Dict with the input parameters for the system
        * **observations** - Dict with the observations for the systems, for eg
            * states - evocraft states 
            * timepoints - timepoints of the observed states

## evocraftsearch.System

The main system's attributes are:

* **initialization_space** - DictSpace object corresponding to Evocraft initialisation genome's parameters
* **update_rule_space** - DictSpace object corresponding to Evocraft update rule genome's parameters
* **intervention_space** - DictSpace object corresponding to valid intervention parameters in the step() function

The main API methods that this class needs to implement are:

* **reset**(self,  initialization_parameters, update_rule_parameters) - resets the environment to an initial state and returns an initial observation
* **step**(self, intervention_parameters) - run one timestep of the system's dynamics
* **render**(self, **kwargs) - renders the environment
* **close**(self) - cleanup the environment
* **save**(self, filepath) - save the system object using torch.save function in pickle format


### evocraftsearch.system.torch_nn.TorchNNSystem

Base class for the torch_nn system. Inherits from evocraftsearch.System and torch.nn.Module.  
Aditionnally to System's main API methods, torch.nn.Module's main API methods are callable.

### evocraftsearch.system.LeniaChem
Class for the LeniaChem system (implemented as a differentiable torch module).

## evocraftsearch.OutputRepresentation 

Class used to convert observations (outputs of the system) to an embedding vector (eg: used for the goal space mapping
representation of a goal-based explorer).  
The main API methods that this class needs to implement are:

* **calc**(self, observations, **kwargs) - maps the observations of a system to an embedding (embedding space can take
  different forms)


## evocraftsearch.OutputFitness

Class used to convert observations (outputs of the system) to a fitness score (eg: used for the selection in neat-sgd
explorer).  
The main API methods that this class needs to implement are:

* **calc**(self, observations, **kwargs) - maps the observations of a system to a fitness score

## evocraftsearch.Explorer

Base class for exploration experiments.

* **system** - system explored
* **db** - ExplorationDB used to store and save exploration results

Allows to save and load exploration results via the ExplorationDB class. The main API methods that this class needs to
implement are:

* **save**(self, filepath) - Saves the explorer object using torch.save function in pickle format
* **load**(explorer_filepath, load_data=True, run_ids=None, map_location='cuda') - static method to load an explorer
  from a saved pickle

### evocraftsearch.explorers.IMGEPExplorer 

Basic goal-based explorer that samples goals in a goalspace and uses a policy library to generate parameters to reach
the goal.

* **config**
    * **num_of_random_initialization** - number of random initializations before starting the goal-directed exploration
    * **frequency_of_random_initialization** - frequency of random initialization during the goal-directed exploration
    * **source_policy_selection** - config used to sample policy parameters given a target goal
    * **reach_goal_optimizer** - config used to optimize policy parameters toward a target goal
* **policy_library** - episodic memory of policy parameters
* **goal_library** - episodic memory of reached goals


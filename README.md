# Table of Contents

* [Install and Run](#evocraftsearch-install-and-run)
  
* [Code Skeleton](#evocraftsearch-code-skeleton)
  
* [RoadMap](#evocraftsearch-roadmap)

---

# EvocraftSearch: Install and Run

## Step 1: Installation
1. If you do not already have it, please install [Conda](https://www.anaconda.com/)
2. Create *morphosearch* conda environment: `conda create --name morphosearch python=3.6`
3. Activate *morphosearch* conda environment: `conda activate morphosearch`
4. If you do not already have it, please create a package folder that you will link to your conda env: `mkdir <path_to_packages_folder>`
5. Into your package folder, clone the following packages:  
    a. `git clone git@github.com:mayalenE/exputils.git`  
    b. `git clone git@github.com:mayalenE/pytorchneat.git`   
    c. `git clone git@github.com:mayalenE/evocraftsearch.git`
5. Include thos packages in the conda environment:  
   `echo <path_to_packages_folder> "$HOME/miniconda3/envs/morphosearch/lib/python3.6/site-packages/my_packages.pth"`
6. Install the required conda packages in the environment (*requirements.txt* file can be found in evocraftsearch directory):  
   `while read requirement; do conda install --yes $requirement --channel default --channel anaconda --channel conda-forge --channel pytorch; done < requirements.txt`
7. For jupyter (see [link](https://github.com/Anaconda-Platform/nb_conda_kernels)):
    * in base: conda install nb_conda_kernels widgetsnbextension
    * in morphosearch: conda install ipykernel nbformat
    


## Step 2: Prepare the experiment folder structures
Experiments are stored in a specific folder structure which allows to save and load experimental data in a structured manner.
Please note that  it represents a default structure which can be adapted if required.
Elements in brackets (\<custom name>\) can have custom names.   
Folder structure:
      
        <path_to_packages_folder>/ 
        ├── evocraftsearch
        ├── image_representation
        ├── pytorchneat
        └── exputils  

        <experimental campaign>/  
        ├── analyze                                 # Scripts such as Jupyter notebooks to analyze the different experiments in this experimental campaign.  
        ├── experiment_configurations.ods           # ODS file that contains the configuration parameters of the different experiments in this campaign.  
        ├── code                                    # Holds code templates of the experiments.  
        │   ├── <repetition code>                   # Code templates that are used under the repetition folders of th experiments. These contain the acutal experimental code that should be run.  
        │   ├── <experiment code>                   # Code templates that are used under the experiment folder of the experiment. These contain usually code to compute statistics over all repetitions of an experiment.  
        ├── generate_code.sh                        # Script file that generates the experimental code under the experiments folder using the configuration in the experiment_configurations.ods file and the code under the code folder.          
        ├── experiments folder                      # Contains generated code for experiments and the collected experimental data.
        │   ├── experiment_{id}
        |   │    ├── repetition_{id}
        │   │    │    ├── data                      # Experimental data for the single repetitions, such as logs.
        │   │    │    └── code                      # Generated code and resource files.
        |   │    ├── data                           # Experimental data for the whole experiment, e.g. statistics that are calculated over all repetitions.   
        |   │    └── <code>                         # Generated code and resource files.  
        └── <run scripts>.sh                        # Various shell scripts to run experiments and calculate statistics locally or on clusters.

## Step 3: Run the experiments on Jeanzay
### Installation of morphosearch on the clusters
* install Miniconda on Jeanzay:
```bash
  cd /tmp
  wget <https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh>
  chmod +x Miniconda-latest-Linux-x86_64.sh
  ./Miniconda-latest-Linux-x86_64.sh 
  # follow the installation instruction (path for the installation, conda init: yes)
  source ~/.bashrc # activate the installation
```
* follow the same steps to setup the morphosearch conda environments
* create folders <DESTINATION_CODE> and <DESTINATION_EXPERIMENT> on jeanzay and modify accordingly <run_scripts>.sh 


### Useful commands on the clusters
```
---------
Slurm

- See running jobs: 
	squeue -u <USERNAME>

- Detect status of experiments and calculation of statistics:
	for f in $(find . -name "run_experiment.slurm.status"); do STATUS=$(tail -1 $f); echo $STATUS - $f ;done
	for f in $(find . -name "run_calc_statistics_per_experiment.slurm.status"); do STATUS=$(tail -1 $f); echo $STATUS - $f ;done
	for f in $(find . -name "run_calc_statistics_per_repetition.slurm.status"); do STATUS=$(tail -1 $f); echo $STATUS - $f ;done

- Deleting specific files:
	find -name <FILENAME> -delete
```

---

# EvocraftSearch: Code Skeleton
The general structure of the code is inspired from [OpenAI's Gym](https://github.com/openai/gym/tree/master/gym) library.  
The main classes (***System***, ***OutputRepresentation***, ***OutputFitness*** and ***Explorer***) are implemented in `core.py`.


## addict.Dict

Class which implements a dictionary that provides attribute-style access.  
This class is used to implement configurations of all morphosearch classes, typically initialized in the
class `__init__` function with:

```
self.config = self.__class__.default_config()
self.config.update(config)
self.config.update(kwargs)
```

## evocraftsearch.spaces.Space

Class used to define valid spaces for (i) input parameters of the Evocraft systems and (ii) IMGEP explorer goal space.
Implemented space classes are: BoxSpace, DiscreteSpace, MultiDiscreteSpace and MultiBinarySpace.  
Implemented container classes are: TupleSpace anb DictSpace.  
Most use-cases should be covered by those classes but custom Space can be defined and must implement the following API methods:
* **sample**(self) - randomly sample an element of this space
* **mutate**(self, x) - randomly mutate an element of this space
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
* **intervention_space** - DictSpace object corresponding to valid intervention parameters in Evocraft step() function

The main API methods that this class needs to implement are:

* **reset**(self,  initialization_parameters, update_rule_parameters) - resets the environment to an initial state and returns an initial observation
* **step**(self, intervention_parameters) - run one timestep of the system's dynamics
* **render**(self, **kwargs) - renders the environment
* **close**(self) - cleanup the environment
* **save**(self, filepath) - save the system object using torch.save function in pickle format


### evocraftsearch.system.torch_nn.TorchNNSystem

Base class for the torch_nn system. Inherits from evocraftsearch.System and torch.nn.Module.  
Aditionnally to System's main API methods, torch.nn.Module's main API methods are callable.



## evocraftsearch.OutputRepresentation 

Class used to convert observations (outputs of the system) to an embedding vector (eg: used for the goal space mapping
representation of a goal-based explorer).  
The main API methods that this class needs to implement are:

* **calc**(self, observations, **kwargs) - maps the observations of a system to an embedding (embedding space can take
  different forms)
* **calc_distance**(self, embedding_a, embedding_b, **kwargs) - computes the distance between 2 embeddings

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
    * **source_policy_selection** - config used to sample policy parameters given a target goal
    * **reach_goal_optimizer** - config used to optimize policy parameters toward a target goal
* **policy_library** - episodic memory of policy parameters
* **goal_library** - episodic memory of reached goals

# EvocraftSearch: RoadMap

### Random seed

* [optional] sequence of seeds per run_id instead of single seed
    * at each run:
        * set parameter sampler seed with run_id_seed
        * set env seed with run_id_seed
        * save the seed in exploration_db run data entries

### Exploration database

* [optional] Save in one big file with h5py instead of many torch pickles


### Evocraft
* parameter b: differentiable?
* Evocraft NDKC: integrate Gautier code

### OutputFitness
* integrate Gautier code


### OutputRepresentation
* integrate VAE, HOLMES

### Explorers
* implement EA explorer (see DEAP library) 
* implement IMGEPOGL explorer, IMGEP-HOLMES explorer

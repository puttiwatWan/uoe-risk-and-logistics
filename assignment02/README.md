## Dependencies
To install dependencies, simply run:
```commandline
pip install -r requirements.txt
```

---
## Run
### Heuristics Run
To run the deterministic heuristics, run the file `heuristic_main.py` 
To run the stochastic heuristics, run the file `sto_heuristic_main.py` 

### MINLP Run
To run the deterministic MINLP, run the file `minlp_deter.py` 
To run the stochastic MINLP, run the file `minlp_sto.py` 

---
## Heuristics Algorithm
The heuristics algorithm is stored in the files under `heuristics/` folder.
The files are separated into deterministic and stochastic version, 
where the stochastic version reuses the base method in the deterministic version.
---
## Pre-Processing Logic
`atcs.py` is the main file used to process the given data and extract the desired values.

---
## Config
The main configuration is in `config.yaml` file.

### Select Subset of Robots
To run only few samples of robots, simply set `use_subset_robot` to `true`.

Additionally, set `n_samples` to the desired amount of robots.
`seed` can also be changed in order to change the random behavior.

### Random Start
To use random start logic in the construction heuristics, simply set `use_random_start` to `true`.
However, noted that the random start only works with deterministic version. 
Setting these values will not affect the stochastic version.

Note: setting `use_random_start` to `true` will disable the use of `default_starting_robot`.

### Fixed Start
The fixed start logic is the default option for the construction heuristics (`use_random_start` as `false`).
The default robot chosen for starting point is 1.
The value can be updated by setting `default_starting_robot` to the desired robot. 
Noted that this value affects both deterministic and stochastic version.
---
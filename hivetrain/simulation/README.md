# Simulation Project

## Overview

This project implements a simulation of a network with miners and validators, using the Yuma consensus algorithm. The simulation allows for the evaluation of miner performance and the distribution of rewards based on the consensus mechanism.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/hivetrain.git
   ```
2. Navigate to the project directory:
   ```
   cd hivetrain
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the simulation:

```python
from hivetrain.simulation import Simulation, SimulationConfig

config = SimulationConfig()
sim = Simulation(config)
sim.run_simulation()
```

Or, from the command line:

```
python -m hivetrain.simulation.main
```

## Configuration

The simulation parameters can be configured in the `simulation_config.ini` file. Here's an example configuration:

```ini
[SIMULATION]
num_miners = 100
num_validators = 10
num_subnets = 5
miners_per_validator = 20
num_rounds = 50

[GLOBAL_STORE]
initial_mean = 10.0
initial_std = 2.0
decay_rate = 0.99

[YUMA_CONSENSUS]
trust_threshold = 0.0
consensus_kappa = 0.5
consensus_rho = 10.0
```

## Module Descriptions

### global_store.py

This module contains the `GlobalStore` class, which manages the global state of the simulation, including miner losses and the Yuma consensus parameters.

### yuma_consensus.py

Implements the Yuma consensus algorithm, including functions for calculating trust, rank, consensus, and emission.

### miner.py

Defines the `Miner` class, representing individual miners in the network. Each miner has a unique ID and maintains its loss history.

### validator.py

Contains the `Validator` class, representing validators in the network. Validators are responsible for evaluating miners and assigning weights based on their performance.

### config.py

Implements the `SimulationConfig` class, which handles loading and managing simulation parameters from the configuration file.

### simulation.py

The main `Simulation` class that orchestrates the entire simulation process, including initializing components, running simulation rounds, and collecting results.

### main.py

Provides an entry point for running the simulation from the command line.

## Example Code

Here's an example of how to use the main components of the simulation:

```python
from hivetrain.simulation import Simulation, SimulationConfig, Miner, Validator, GlobalStore

# Initialize configuration
config = SimulationConfig()

# Create a simulation instance
sim = Simulation(config)

# Run the simulation
sim.run_simulation()

# Access simulation results
results = sim.get_results()

# Analyze and visualize results
# (Add your analysis and visualization code here)
```

## Contributing

Contributions to this project are welcome. Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add some feature'`)
5. Push to the branch (`git push origin feature/your-feature-name`)
6. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
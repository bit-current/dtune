import configparser
import os
from typing import Any, Dict

class SimulationConfig:
    def __init__(self, config_file: str = 'simulation_config.ini'):
        self.config = configparser.ConfigParser()
        self.config_file = config_file
        self.load_config()

    def load_config(self):
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
        else:
            self.set_default_config()

    def set_default_config(self):
        self.config['SIMULATION'] = {
            'num_miners': '100',
            'num_validators': '10',
            'num_subnets': '5',
            'miners_per_validator': '20',
            'num_rounds': '50'
        }
        self.config['GLOBAL_STORE'] = {
            'initial_mean': '10.0',
            'initial_std': '2.0',
            'decay_rate': '0.99'
        }
        self.config['YUMA_CONSENSUS'] = {
            'trust_threshold': '0.0',
            'consensus_kappa': '0.5',
            'consensus_rho': '10.0'
        }
        self.save_config()

    def save_config(self):
        with open(self.config_file, 'w') as configfile:
            self.config.write(configfile)

    def get(self, section: str, key: str, fallback: Any = None) -> Any:
        return self.config.get(section, key, fallback=fallback)

    def getint(self, section: str, key: str, fallback: int = None) -> int:
        return self.config.getint(section, key, fallback=fallback)

    def getfloat(self, section: str, key: str, fallback: float = None) -> float:
        return self.config.getfloat(section, key, fallback=fallback)

    def set(self, section: str, key: str, value: Any):
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, key, str(value))

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        return {section: dict(self.config[section]) for section in self.config.sections()}

# Example usage
if __name__ == "__main__":
    config = SimulationConfig()

    # Print all configurations
    print("Current configuration:")
    for section, params in config.get_all().items():
        print(f"\n[{section}]")
        for key, value in params.items():
            print(f"{key} = {value}")

    # Modify a configuration
    config.set('SIMULATION', 'num_miners', '200')
    config.set('GLOBAL_STORE', 'initial_mean', '8.0')

    # Save the modified configuration
    config.save_config()

    print("\nAfter modification:")
    print(f"Number of miners: {config.getint('SIMULATION', 'num_miners')}")
    print(f"Initial mean: {config.getfloat('GLOBAL_STORE', 'initial_mean')}")


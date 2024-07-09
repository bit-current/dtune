import os
import torch
from typing import List, Dict
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from tqdm import tqdm
# Assuming we have already implemented these classes
from hivetrain.simulation import GlobalStore, YumaConsensus, Miner, Validator, SimulationConfig, NewConsensusMechanism


class DataCollector:
    def __init__(self):
        self.round_data = []
        self.miner_data = []
        self.validator_data = []

    def collect_round_data(self, round_num: int, global_mean: float, consensus_results: Dict[str, torch.Tensor]):
        self.round_data.append({
            'round': round_num,
            'global_mean_loss': global_mean,
            'mean_similarity': consensus_results['similarities'].mean().item(),
            'mean_penalty': consensus_results['penalties'].mean().item(),
            'mean_emission': consensus_results['emissions'].mean().item(),
            'mean_validator_trust': consensus_results['validator_trust'].mean().item()
        })

    def collect_miner_data(self, round_num: int, miners: List, consensus_results: Dict[str, torch.Tensor]):
        for i, miner in enumerate(miners):
            self.miner_data.append({
                'round': round_num,
                'miner_id': miner.uid,
                'current_loss': miner.current_loss,
                'emission': consensus_results['emissions'][i].item(),
            })

    def collect_validator_data(self, round_num: int, validators: List, consensus_results: Dict[str, torch.Tensor]):
        for i, validator in enumerate(validators):
            self.validator_data.append({
                'round': round_num,
                'validator_id': validator.uid,
                'mean_weight': np.mean(list(validator.weights.values())),
                'similarity': consensus_results['similarities'][i].item(),
                'penalty': consensus_results['penalties'][i].item(),
                'validator_trust': consensus_results['validator_trust'][i].item()
            })

    def get_dataframes(self):
        return {
            'round': pd.DataFrame(self.round_data),
            'miner': pd.DataFrame(self.miner_data),
            'validator': pd.DataFrame(self.validator_data)
        }
    
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import pandas as pd
import numpy as np



class DataVisualizer:
    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data
        sns.set(style="whitegrid")
        self.output_dir = "simulation_plots"
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_global_metrics(self):
        fig, axes = plt.subplots(3, 2, figsize=(20, 24))
        fig.suptitle('Global Metrics Over Time', fontsize=16)

        metrics = [
            ('global_mean_loss', 'Global Mean Loss'),
            ('mean_similarity', 'Mean Similarity'),
            ('mean_penalty', 'Mean Penalty'),
            ('mean_emission', 'Mean Emission'),
            ('mean_validator_trust', 'Mean Validator Trust')
        ]

        for i, (metric, title) in enumerate(metrics):
            row, col = divmod(i, 2)
            sns.lineplot(data=self.data['round'], x='round', y=metric, ax=axes[row, col])
            axes[row, col].set_title(title)
            if 'penalty' in metric or 'trust' in metric:
                axes[row, col].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'global_metrics.png'))
        plt.close()

    def plot_miner_performance(self):
        plt.figure(figsize=(20, 10))
        sns.lineplot(data=self.data['miner'], x='round', y='current_loss', hue='miner_id', 
                     legend=False, alpha=0.3)
        sns.lineplot(data=self.data['miner'].groupby('round').mean().reset_index(), 
                     x='round', y='current_loss', color='red', linewidth=2)
        plt.title('Miner Losses Over Time')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(self.output_dir, 'miner_performance.png'))
        plt.close()

    def plot_miner_emissions(self):
        plt.figure(figsize=(20, 10))
        sns.lineplot(data=self.data['miner'], x='round', y='emission', hue='miner_id', 
                     legend=False, alpha=0.3)
        sns.lineplot(data=self.data['miner'].groupby('round').mean().reset_index(), 
                     x='round', y='emission', color='red', linewidth=2)
        plt.title('Miner Emissions Over Time')
        plt.xlabel('Round')
        plt.ylabel('Emission')
        plt.savefig(os.path.join(self.output_dir, 'miner_emissions.png'))
        plt.close()

    def plot_validator_weights(self):
        plt.figure(figsize=(20, 10))
        sns.boxplot(data=self.data['validator'], x='round', y='mean_weight')
        plt.title('Distribution of Validator Weights Over Time')
        plt.xlabel('Round')
        plt.ylabel('Mean Weight')
        plt.savefig(os.path.join(self.output_dir, 'validator_weights.png'))
        plt.close()



    def plot_validator_similarities(self):
        plt.figure(figsize=(20, 10))
        sns.lineplot(data=self.data['validator'], x='round', y='similarity', hue='validator_id', 
                     legend=False, alpha=0.3)
        sns.lineplot(data=self.data['round'], x='round', y='mean_similarity', color='red', linewidth=2)
        plt.title('Validator Similarities Over Time')
        plt.xlabel('Round')
        plt.ylabel('Similarity')
        plt.savefig(os.path.join(self.output_dir, 'validator_similarities.png'))
        plt.close()

    def plot_validator_trust(self):
        plt.figure(figsize=(20, 10))
        sns.lineplot(data=self.data['validator'], x='round', y='validator_trust', hue='validator_id', 
                     legend=False, alpha=0.3)
        sns.lineplot(data=self.data['round'], x='round', y='mean_validator_trust', color='red', linewidth=2)
        plt.title('Validator Trust Over Time')
        plt.xlabel('Round')
        plt.ylabel('Validator Trust')
        plt.ylim(0, 1)
        plt.savefig(os.path.join(self.output_dir, 'validator_trust.png'))
        plt.close()

    def plot_validator_penalties(self):
        plt.figure(figsize=(20, 10))
        sns.lineplot(data=self.data['validator'], x='round', y='penalty', hue='validator_id', 
                     legend=False, alpha=0.3)
        sns.lineplot(data=self.data['round'], x='round', y='mean_penalty', color='red', linewidth=2)
        plt.title('Validator Penalties Over Time')
        plt.xlabel('Round')
        plt.ylabel('Penalty')
        plt.ylim(0, 1)
        plt.savefig(os.path.join(self.output_dir, 'validator_penalties.png'))
        plt.close()

    def plot_top_miners(self, top_n=10):
        last_round = self.data['miner']['round'].max()
        top_miners = self.data['miner'][self.data['miner']['round'] == last_round].nsmallest(top_n, 'current_loss')['miner_id']
        
        top_miner_data = self.data['miner'][self.data['miner']['miner_id'].isin(top_miners)]
        
        plt.figure(figsize=(20, 10))
        sns.lineplot(data=top_miner_data, x='round', y='current_loss', hue='miner_id')
        plt.title(f'Performance of Top {top_n} Miners')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.legend(title='Miner ID', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'top_{top_n}_miners_performance.png'))
        plt.close()

    def plot_correlation_heatmap(self):
        # Prepare data
        round_data = self.data['round'].copy()
        miner_means = self.data['miner'].groupby('round').mean().reset_index()
        validator_means = self.data['validator'].groupby('round').mean().reset_index()
        
        combined_data = pd.merge(round_data, miner_means, on='round', suffixes=('', '_miner'))
        combined_data = pd.merge(combined_data, validator_means, on='round', suffixes=('', '_validator'))
        
        # Select relevant columns for correlation
        corr_columns = ['global_mean_loss', 'mean_similarity', 'mean_penalty', 'mean_emission', 
                        'mean_validator_trust', 'current_loss', 'emission', 'mean_weight', 
                        'similarity', 'penalty', 'validator_trust']
        corr_data = combined_data[corr_columns]
        
        # Compute correlation matrix
        corr_matrix = corr_data.corr()
        
        # Plot heatmap
        plt.figure(figsize=(15, 12))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Correlation Heatmap of Simulation Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'correlation_heatmap.png'))
        plt.close()

    def generate_all_plots(self):
        print("Generating plots...")
        self.plot_global_metrics()
        self.plot_miner_performance()
        self.plot_miner_emissions()
        self.plot_validator_weights()
        self.plot_validator_trust()
        self.plot_validator_similarities()
        self.plot_validator_penalties()
        self.plot_top_miners()
        self.plot_correlation_heatmap()
        print(f"All plots saved in directory: {self.output_dir}")

class Simulation:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.num_miners = config.getint('SIMULATION', 'num_miners')
        self.num_validators = config.getint('SIMULATION', 'num_validators')
        self.num_subnets = config.getint('SIMULATION', 'num_subnets')
        self.miners_per_validator = config.getint('SIMULATION', 'miners_per_validator')
        
        initial_mean = config.getfloat('GLOBAL_STORE', 'initial_mean')
        initial_std = config.getfloat('GLOBAL_STORE', 'initial_std')
        decay_rate = config.getfloat('GLOBAL_STORE', 'decay_rate')
        
        self.global_store = GlobalStore(initial_mean=initial_mean,
                                        initial_std=initial_std,
                                        decay_rate=decay_rate)
        
        self.miners = [Miner(i) for i in range(self.num_miners)]
        self.validators = [Validator(i, self.num_miners) for i in range(self.num_validators)]
        
        self.consensus_mechanism = NewConsensusMechanism(
            penalty_factor=config.getfloat('CONSENSUS', 'penalty_factor', fallback=1.0)
        )

        self.data_collector = DataCollector()
        
        # Initialize stake vector
        self.global_store.stake_vector = torch.ones(size=(self.num_validators,),dtype=torch.float32)
        ##self.global_store.stake_vector /= self.global_store.stake_vector.sum()
        self.miners_per_validator = config.getint('SIMULATION', 'miners_per_validator', fallback=5)

    @staticmethod
    def get_total_runs(num_miners, miners_per_validator):
        return math.ceil(num_miners / miners_per_validator)

    @staticmethod
    def get_current_run(counter, total_runs):
        return counter % total_runs

    def assign_miners_to_validators(self, assign_all = False):
        if assign_all:
            miner_uids = [miner.uid for miner in self.miners]
            validator_uids = [validator.uid for validator in self.validators]
            for validator in self.validators:
                validator.assign_miners(miner_uids)
            return {validator_uid:miner_uids for validator_uid in validator_uids}
        else:

            miner_uids = [miner.uid for miner in self.miners]
            validator_uids = [validator.uid for validator in self.validators]
            
            current_step = self.round  # Using counter_pushed as the current step
            
            total_runs = self.get_total_runs(len(miner_uids), self.miners_per_validator)
            current_run = self.get_current_run(current_step, total_runs)
            
            # Calculate the ratio based on the fixed number of miners per validator
            ratio = self.miners_per_validator / len(miner_uids)
            
            miner_assignments = {vali_uid: [] for vali_uid in validator_uids}

            for i, vali_uid in enumerate(validator_uids):
                start_index = (i + current_run * len(validator_uids)) % len(miner_uids)
                assigned_miners = [miner_uids[(start_index + j) % len(miner_uids)] for j in range(self.miners_per_validator)]
                miner_assignments[vali_uid] = assigned_miners
            
            # Log the current assignment details
            # print(f"Step: {current_step}, Run: {current_run}, Total Runs: {total_runs}")
            # print(f"Miners per validator: {self.miners_per_validator}")
            # print(f"Total miners: {len(miner_uids)}, Total validators: {len(validator_uids)}")
            # print(f"Ratio: {ratio:.2f}, Miners assigned this run: {self.miners_per_validator * len(validator_uids)}")
            
            # Update validators with their assigned miners
            for validator in self.validators:
                validator.assign_miners(miner_assignments[validator.uid])
        
            return miner_assignments
        
    def update_miner_losses(self):
        for miner in self.miners:
            self.global_store.update_miner_loss(miner.uid)
            miner.update_loss(self.global_store.get_miner_losses(miner.uid)[-1])
        self.global_store.calculate_round_statistics()

    def run_validator_evaluations(self, clear):
        for validator in self.validators:
            validator.evaluate_miners(self.global_store.losses, self.global_store.current_round_avg_loss, clear)

    def run_consensus(self):
        W = torch.zeros(self.num_validators, self.num_miners)
        for i, validator in enumerate(self.validators):
            weights = validator.get_weights()
            for j, weight in weights.items():
                W[i][j] = weight

        S = self.global_store.stake_vector
        
        return self.consensus_mechanism.run_consensus(W, S)

    def run_simulation(self):
        num_rounds = self.config.getint('SIMULATION', 'num_rounds')
        for round in tqdm(range(num_rounds)):
            
            self.round = round

            #if round % 20 == 0:
            self.global_store.calculate_round_statistics()
            #self.run_validator_evaluations(clear=True)

            self.assign_miners_to_validators()
            self.update_miner_losses()
            self.run_validator_evaluations(clear=False)
            consensus_results = self.run_consensus()
            
            self.global_store.update_global_mean()
            
            # Collect data
            #if round % 20 == 0:
            self.data_collector.collect_round_data(round, self.global_store.mean, consensus_results)
            self.data_collector.collect_miner_data(round, self.miners, consensus_results)
            self.data_collector.collect_validator_data(round, self.validators, consensus_results)

        # After the simulation, generate all plots
        data = self.data_collector.get_dataframes()
        visualizer = DataVisualizer(data)
        visualizer.generate_all_plots()

        
    def get_collected_data(self):
        return self.data_collector.get_dataframes()


    def print_round_summary(self, round: int, consensus_results: Dict[str, torch.Tensor]):
        print(f"Round {round + 1} Summary:")
        print(f"Global mean loss: {self.global_store.mean:.4f}")
        print("Top 5 miners by emission:")
        top_miners = torch.topk(consensus_results['emission'], 5)
        for i, (index, value) in enumerate(zip(top_miners.indices, top_miners.values)):
            print(f"  {i+1}. Miner {index}: {value:.4f}")

if __name__ == "__main__":
    config = SimulationConfig()
    sim = Simulation(config)
    sim.run_simulation()

    data = sim.get_collected_data()
    visualizer = DataVisualizer(data)
    
    visualizer.generate_all_plots()
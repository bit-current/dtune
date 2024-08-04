import numpy as np
from typing import Dict, List

class GlobalGaussian:
    def __init__(self, initial_mean, initial_std, decay_rate):
        self.mean = initial_mean
        self.std = initial_std
        self.decay_rate = decay_rate

    def sample(self):
        return max(0, np.random.normal(self.mean, self.std))

    def update_mean(self):
        self.mean *= self.decay_rate

# Global Gaussian instance
global_gaussian = GlobalGaussian(initial_mean=10, initial_std=0.01, decay_rate=0.99)


class GlobalStore:
    def __init__(self, initial_mean: float = 10.0, initial_std: float = 2.0, decay_rate: float = 0.99):
        self.losses: Dict[int, List[float]] = {}
        self.simulator = GlobalGaussian(initial_mean, initial_std, decay_rate)
        self.stake_vector = None  # This will be initialized in the Simulation class
        self.current_round_losses: List[float] = []
        self.current_round_avg_loss: float = float("inf")
        self.current_round_min_loss: float = float("inf")

    def update_miner_loss(self, miner_uid: int):
        if miner_uid not in self.losses:
            self.losses[miner_uid] = []
        
        new_loss = self.simulator.sample()
        self.losses[miner_uid].append(new_loss)
        self.current_round_losses.append(new_loss)

    def get_miner_losses(self, miner_uid: int) -> List[float]:
        return self.losses.get(miner_uid, [])

    def update_global_mean(self):
        self.simulator.update_mean()

    def calculate_round_statistics(self):
        if self.current_round_losses:
            below_average_losses = [loss for loss in self.current_round_losses if loss < self.current_round_avg_loss]
            if len(below_average_losses) > 0:
                self.current_round_avg_loss = np.mean(below_average_losses)
                self.current_round_min_loss = np.min(below_average_losses)
            else:
                pass
        else:
            pass

        self.current_round_losses = []  # Reset for the next round

    @property
    def mean(self):
        return self.simulator.mean

    def __str__(self):
        return (f"GlobalStore(miners: {len(self.losses)}, mean: {self.simulator.mean:.4f}, "
                f"std: {self.simulator.std:.4f}, current_round_min_loss: {self.current_round_min_loss:.4f})")
    
class Miner:
    def __init__(self, uid):
        self.uid = uid
        self.losses = []
        self.current_loss = None

    def update_loss(self, new_loss):
        """
        Update the miner's current loss with a new value.

        Args:
            new_loss (float): The new loss value to be added.
        """
        self.current_loss = max(0, new_loss)  # Ensure non-negative loss
        self.losses.append(self.current_loss)

    def get_loss_reduction(self):
        if len(self.losses) < 2:
            return 0
        return max(0, self.losses[-2] - self.losses[-1])

    def __str__(self):
        return f"Miner(uid={self.uid}, current_loss={self.current_loss:.4f if self.current_loss is not None else 'N/A'})"
    
# Example usage
if __name__ == "__main__":
    # Create a few miners
    miners = [Miner(i) for i in range(5)]

    # Simulate a few rounds
    for round in range(10):
        print(f"\nRound {round + 1}")
        global_gaussian.update_mean()
        for miner in miners:
            miner.update_loss()
            print(f"{miner}, loss reduction: {miner.get_loss_reduction():.4f}")
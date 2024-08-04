import numpy as np
from typing import List, Callable, Dict

class Validator:
    def __init__(self, uid: int, num_miners: int, incentive_function: Callable = None):
        self.uid = uid
        self.assigned_miners: List[int] = []
        self.num_miners = num_miners
        self.weights: Dict[int, float] = {i: 0.0 for i in range(num_miners)}  # Initialize weights for all miners
        self.incentive_function = incentive_function or self.default_incentive_function

    def assign_miners(self, miner_uids: List[int]):
        self.assigned_miners = miner_uids
        # No need to reset weights here, as we're maintaining weights for all miners

    def evaluate_miners(self, global_store_losses: Dict[int, List[float]], best_average_loss: float, clear = False):
        selected_miner_losses = []
        selected_miner_uids = []
        if clear:
            print("Clearing weights")
            self.weights = {i: 0.5 for i in range(self.num_miners)}  # Initialize weights for all miners

        for uid in self.assigned_miners:
            losses = global_store_losses.get(uid, [])
            
            if losses:
                loss = losses[-1]
                selected_miner_losses.append(self.incentive_function(loss, best_average_loss))
                selected_miner_uids.append(uid)
        try:
            max_weight_idx = selected_miner_losses.index(min(selected_miner_losses))
            max_weight_uid = selected_miner_uids[max_weight_idx]
        except:
            max_weight_idx = 999
            max_weight_uid = 999
        

        for uid, loss in zip(selected_miner_uids, selected_miner_losses):
            if uid == max_weight_uid:
                self.weights[uid] = 1.0
            else:
                if loss > 0:
                    self.weights[uid] = 0.5
                else:
                    self.weights[uid] = 0.0


        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for uid in self.weights:
                self.weights[uid] /= total_weight
        #print(f"Validator: {self.uid} Weights: {self.weights}")
    def read_miner_losses(self, global_store: dict):
        return {uid: global_store.get(uid, []) for uid in self.assigned_miners}

    @staticmethod
    def default_incentive_function(final_loss: float, best_average_loss: float) -> float:
        """
        Default incentive function.
        Rewards lower final loss and considers the trend of recent losses.
        """

        if final_loss < best_average_loss:
            return np.exp(best_average_loss)-np.exp(final_loss)
        else:
            return 0.0
        

    def set_incentive_function(self, func: Callable):
        self.incentive_function = func

    def get_weights(self) -> Dict[int, float]:
        return self.weights

    def __str__(self):
        return f"Validator(uid={self.uid}, assigned_miners={len(self.assigned_miners)}, total_miners={len(self.weights)})"
    
# Example usage
if __name__ == "__main__":
    # Create a global store to simulate miner losses
    global_store = {
        0: [10, 9, 8],
        1: [10, 11, 10],
        2: [10, 9, 9.5],
        3: [10, 8, 7],
        4: [10, 10, 10]
    }

    # Create a validator and assign miners
    validator = Validator(uid=0)
    validator.assign_miners([0, 1, 2, 3, 4])

    # Evaluate miners
    validator.evaluate_miners(global_store)

    # Print results
    print(validator)
    for miner_uid, weight in validator.get_weights().items():
        print(f"Miner {miner_uid}: weight = {weight:.4f}")

    # Example of setting a custom incentive function
    def custom_incentive(losses: List[float]) -> float:
        if len(losses) < 2:
            return 0.0
        recent_avg = sum(losses[-3:]) / len(losses[-3:])  # Average of last 3 losses
        improvement = losses[0] - recent_avg  # Improvement from initial loss
        return np.tanh(improvement)

    validator.set_incentive_function(custom_incentive)
    print("\nAfter setting custom incentive function:")
    validator.evaluate_miners(global_store)
    for miner_uid, weight in validator.get_weights().items():
        print(f"Miner {miner_uid}: weight = {weight:.4f}")
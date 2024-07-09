import torch
import torch.nn.functional as F

class YumaConsensus:
    def __init__(self, threshold=0, kappa=0.5, rho=10):
        self.threshold = threshold
        self.kappa = kappa
        self.rho = rho

    def trust(self, W: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        Wn = (W > self.threshold).float()
        return Wn.T @ S

    def rank(self, W: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        R = W.T @ S
        return R / R.sum()

    def consensus(self, T: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.rho * (T - self.kappa))

    def emission(self, C: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        E = C * R
        return E / E.sum()

    def validator_trust(self, W: torch.Tensor) -> torch.Tensor:
        """
        Calculate validator trust by summing the rows of the weight matrix.

        Args:
            W (torch.Tensor): Weight matrix

        Returns:
            torch.Tensor: Validator trust vector
        """
        return torch.sum(W, dim=1)

    def weighted_median(self, values: torch.Tensor, weights: torch.Tensor) -> float:
        sorted_indices = torch.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]
        cumulative_weight = torch.cumsum(sorted_weights, dim=0)
        total_weight = cumulative_weight[-1]
        median_weight = total_weight / 2
        median_index = torch.searchsorted(cumulative_weight, median_weight, right=True)
        return sorted_values[median_index].item()

    def weighted_median_col(self, S: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        n_cols = W.shape[1]
        consensus = torch.zeros(n_cols, device=W.device)
        for j in range(n_cols):
            col_weights = W[:, j]
            non_zero_mask = col_weights != 0
            if torch.any(non_zero_mask):
                consensus[j] = self.weighted_median(col_weights[non_zero_mask], S[non_zero_mask])
        return consensus

    def clip_weights(self, W: torch.Tensor, consensus: torch.Tensor) -> torch.Tensor:
        clipped_W = W.clone()
        for j in range(W.shape[1]):
            col = W[:, j]
            clip_value = consensus[j]
            sum_above = torch.sum(col[col > clip_value])
            sum_below = torch.sum(col[col <= clip_value])
            if sum_above > 0:
                scaling_factor = (1 - sum_below) / sum_above
                clipped_W[:, j] = torch.where(col > clip_value, col * scaling_factor, col)
        return clipped_W

    def run_consensus(self, W: torch.Tensor, S: torch.Tensor) -> dict:
        print(f"Weights: {W}")
        print(f"Stakes: {S}")
        self.validate_inputs(W, S)
        Sn = S / S.sum()  # Normalize stake vector

        # Calculate preranks (which is the same as our current rank calculation)
        preranks = self.rank(W, Sn)

        # Calculate consensus using weighted median
        consensus = self.weighted_median_col(Sn, W)

        # Clip weights
        W_clipped = self.clip_weights(W, consensus)

        # Calculate trust, rank, consensus, and emission using clipped weights
        T = self.trust(W, Sn)
        R = self.rank(W, Sn)
        C = self.consensus(T)
        E = self.emission(C, R)

        print(f"Emissions: {E}")

        # Calculate validator trust
        V = self.validator_trust(W_clipped)

        return {
            "preranks": preranks,
            "consensus": consensus,
            "trust": T,
            "rank": R,
            "consensus_sigmoid": C,
            "emission": E,
            "validator_trust": V,
            "clipped_weights": W_clipped
        }

    @staticmethod
    def validate_inputs(W: torch.Tensor, S: torch.Tensor):
        if not isinstance(W, torch.Tensor) or not isinstance(S, torch.Tensor):
            raise ValueError("Weight matrix and stake vector must be PyTorch tensors.")
        
        if W.dim() != 2 or S.dim() != 1:
            raise ValueError("Weight matrix must be 2D and stake vector must be 1D.")
        
        if W.shape[0] != S.shape[0]:
            raise ValueError("Number of rows in weight matrix must match length of stake vector.")
        
        if torch.any(W < 0) or torch.any(W > 1):
            raise ValueError("Weight values must be between 0 and 1.")
        
        if torch.any(S < 0):
            raise ValueError("Stake values must be non-negative.")

class NewConsensusMechanism:
    def __init__(self, penalty_factor: float = 1.0):
        self.penalty_factor = penalty_factor

    def stake_weighted_average(self, W: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        return (W.T @ S) / S.sum()

    def cosine_similarity(self, W: torch.Tensor, avg_W: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(W, avg_W.unsqueeze(0))

    def calculate_penalties(self, similarities: torch.Tensor) -> torch.Tensor:
        """
        Calculate penalties based on cosine similarities, constrained between 0 and 1.
        
        Args:
            similarities (torch.Tensor): Cosine similarity scores (num_validators)
        
        Returns:
            torch.Tensor: Penalty factors (num_validators) between 0 and 1
        """
        # Use sigmoid function to constrain penalties between 0 and 1
        return (1 - similarities)

    def calculate_emissions(self, W: torch.Tensor, S: torch.Tensor, penalties: torch.Tensor) -> torch.Tensor:
        adjusted_stakes = S * (1 - penalties)  # Invert penalties for stake adjustment
        emissions = (W.T @ adjusted_stakes) / adjusted_stakes.sum()
        return emissions / emissions.sum()

    def run_consensus(self, W: torch.Tensor, S: torch.Tensor) -> dict:
        self.validate_inputs(W, S)
        
        avg_W = self.stake_weighted_average(W, S)
        similarities = self.cosine_similarity(W, avg_W)
        penalties = self.calculate_penalties(similarities)
        
        return {
            "avg_weights": avg_W,
            "similarities": similarities,
            "penalties": penalties,
            "emissions": torch.zeros(size=avg_W.shape),
            "validator_trust": 1 - penalties  # Invert penalties to represent trust
        }

    @staticmethod
    def validate_inputs(W: torch.Tensor, S: torch.Tensor):
        if not isinstance(W, torch.Tensor) or not isinstance(S, torch.Tensor):
            raise ValueError("Weight matrix and stake vector must be PyTorch tensors.")
        
        if W.dim() != 2 or S.dim() != 1:
            raise ValueError("Weight matrix must be 2D and stake vector must be 1D.")
        
        if W.shape[0] != S.shape[0]:
            raise ValueError("Number of rows in weight matrix must match length of stake vector.")
        
        if torch.any(W < 0) or torch.any(W > 1):
            raise ValueError("Weight values must be between 0 and 1.")
        
        if torch.any(S < 0):
            raise ValueError("Stake values must be non-negative.")

# Example usage
if __name__ == "__main__":
    # Create example weight matrix and stake vector
    W = torch.rand(64, 33)  # 64 validators, 33 subnets (including root)
    S = torch.rand(64)  # 64 validators

    # Initialize YumaConsensus
    yuma = YumaConsensus(threshold=0.1, kappa=0.6, rho=12)

    # Run consensus
    results = yuma.run_consensus(W, S)

    # Print results
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            print(f"{key.capitalize()}:", value.shape, value[:5])  # Print shape and first 5 elements
        else:
            print(f"{key.capitalize()}:", value)
from typing import Optional, Dict, List, NamedTuple, Tuple
import numpy as np
import torch
from .utils import build_mlp
from .utils import SerializedBuffer_Rl

class GailRewardProvider(torch.nn.Module):
    def __init__(self, state_space) -> None:
        super().__init__()
        self._ignore_done = False
        self.learning_rate = 3e-4
        self.state_space = state_space
        
        self._discriminator_network = GAILDiscrim(state_space)
        self._discriminator_network.to("cuda:0")
        self._demo_buffer = SerializedBuffer_Rl()
        params = list(self._discriminator_network.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate )
        print("load gail")

    def evaluate(self, obs_batch) -> torch.tensor:
        with torch.no_grad():
            estimates = self._discriminator_network(
                obs_batch
            )
            return -torch.log(
                1.0
                - estimates.squeeze(dim=1)
                * (1.0 - self._discriminator_network.EPSILON)
            )


    def update(self, obs_batch) -> Dict[str, np.ndarray]:

        expert_batch_size = obs_batch.shape[0]
        expert_batch = self._demo_buffer.sample(
            expert_batch_size
        )

        loss, stats_dict = self._discriminator_network.compute_loss(
            obs_batch, expert_batch
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return stats_dict

    def get_modules(self):
        return {f"Module:{self.name}": self._discriminator_network}



class GAILDiscrim(torch.nn.Module):

    def __init__(self, state_shape, hidden_units=(100, 100),
                 hidden_activation=torch.nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        
        self.EPSILON = 1e-7
        self.gradient_penalty_weight = 10.0

    def forward(self, states):
        return self.net(states)
    
    def compute_loss(
        self, policy_batch, expert_batch
    ) -> torch.Tensor:
        """
        Given a policy mini_batch and an expert mini_batch, computes the loss of the discriminator.
        """
        total_loss = torch.zeros(1,device="cuda:0")
        stats_dict: Dict[str, np.ndarray] = {}
        policy_estimate = self.forward(
            policy_batch
        )
        expert_estimate = self.forward(
            expert_batch
        )
        
        stats_dict["Policy/GAIL Policy Estimate"] = policy_estimate.mean().item()
        stats_dict["Policy/GAIL Expert Estimate"] = expert_estimate.mean().item()
        
        discriminator_loss = -(
            torch.log(expert_estimate + self.EPSILON)
            + torch.log(1.0 - policy_estimate + self.EPSILON)
        ).mean()
        
        stats_dict["Losses/GAIL Loss"] = discriminator_loss.item()
        
        total_loss += discriminator_loss
        
        if self.gradient_penalty_weight > 0.0:
            gradient_magnitude_loss = (
                self.gradient_penalty_weight
                * self.compute_gradient_magnitude(policy_batch, expert_batch)
            )
            stats_dict["Policy/GAIL Grad Mag Loss"] = gradient_magnitude_loss.item()
            total_loss += gradient_magnitude_loss
        
        return total_loss, stats_dict

    def compute_gradient_magnitude(
        self, policy_batch, expert_batch
    ) -> torch.Tensor:
        """
        Gradient penalty from https://arxiv.org/pdf/1704.00028. Adds stability esp.
        for off-policy. Compute gradients w.r.t randomly interpolated input.
        """


        obs_epsilon = torch.rand(policy_batch.shape,device=policy_batch.device)
        interp_inputs = obs_epsilon * policy_batch + (1 - obs_epsilon) * expert_batch
        interp_inputs.requires_grad = True  # For gradient calculation
        
        estimate = self.forward(interp_inputs).squeeze(1).sum()

        gradient = torch.autograd.grad(estimate, interp_inputs, create_graph=True)[0]
        # Norm's gradient could be NaN at 0. Use our own safe_norm
        safe_norm = (torch.sum(gradient**2, dim=1) + self.EPSILON).sqrt()
        gradient_mag = torch.mean((safe_norm - 1) ** 2)
        return gradient_mag

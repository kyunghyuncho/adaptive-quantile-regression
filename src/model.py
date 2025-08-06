import torch
import torch.nn as nn
import pytorch_lightning as pl
from .loss import pinball_loss

class QuantileRegressor(pl.LightningModule):
    """
    A PyTorch Lightning module for adaptive quantile regression.

    This model takes an input feature `x` and a target quantile `tau` and
    predicts the corresponding conditional quantile of the output `y`.
    It is trained by sampling `tau` uniformly from [0, 1] at each step.
    """
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        learning_rate: float = 1e-3,
        num_hidden_layers: int = 2,
    ):
        """
        Args:
            input_size (int): The number of input features (usually 1 for this problem).
            hidden_size (int): The number of neurons in each hidden layer.
            learning_rate (float): The learning rate for the optimizer.
            num_hidden_layers (int): The number of hidden layers in the network.
        """
        super().__init__()
        # This saves the arguments to self.hparams and makes them available
        # to the trainer and for checkpointing.
        self.save_hyperparameters()

        # The input to the network is the feature `x` concatenated with the quantile `tau`.
        # So, the effective input size is input_size + 1.
        layers = [
            nn.Linear(self.hparams.input_size + 1, self.hparams.hidden_size),
            nn.ReLU()
        ]

        # Add the specified number of hidden layers
        for _ in range(self.hparams.num_hidden_layers - 1):
            layers.extend([
                nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size),
                nn.ReLU()
            ])

        # The output layer predicts the single quantile value.
        layers.append(nn.Linear(self.hparams.hidden_size, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Args:
            x (torch.Tensor): The input features. Shape: (batch_size, input_size).
            tau (torch.Tensor): The target quantiles. Shape: (batch_size, 1).

        Returns:
            torch.Tensor: The predicted quantiles. Shape: (batch_size, 1).
        """
        # Concatenate the input features and the quantiles along the dimension 1.
        input_tensor = torch.cat([x, tau], dim=1)
        return self.net(input_tensor)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Performs a single training step.
        """
        x, y = batch
        
        # --- The core of adaptive training ---
        # For each sample in the batch, draw a random quantile from a uniform distribution.
        tau = torch.rand(x.size(0), 1, device=self.device)
        
        # Get the quantile prediction
        y_hat = self(x, tau)
        
        # Calculate the pinball loss
        loss = pinball_loss(y_hat, y, tau)
        
        # Log the training loss for monitoring (e.g., with wandb)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int):
        """
        Performs a single validation step.
        """
        x, y = batch
        
        # For validation, we check a few fixed, representative quantiles
        # to get a stable measure of performance across epochs.
        quantiles_to_check = torch.tensor([0.10, 0.50, 0.90], device=self.device)
        total_loss = 0
        
        for q in quantiles_to_check:
            # Create a tensor of the current quantile for the whole batch
            tau = torch.full((x.size(0), 1), fill_value=q, device=self.device)
            y_hat = self(x, tau)
            loss = pinball_loss(y_hat, y, tau)
            
            # Log the loss for this specific quantile
            self.log(f'val_loss_q{int(q*100)}', loss, on_epoch=True, logger=True)
            total_loss += loss
            
        # Log the average validation loss across the checked quantiles
        avg_loss = total_loss / len(quantiles_to_check)
        self.log('val_loss', avg_loss, on_epoch=True, prog_bar=True, logger=True)
        return avg_loss

    def test_step(self, batch: tuple, batch_idx: int):
        """
        Performs a single test step.
        """
        # The logic for the test step is often the same as the validation step.
        x, y = batch
        quantiles_to_check = torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95], device=self.device)
        total_loss = 0
        
        for q in quantiles_to_check:
            tau = torch.full((x.size(0), 1), fill_value=q, device=self.device)
            y_hat = self(x, tau)
            loss = pinball_loss(y_hat, y, tau)
            self.log(f'test_loss_q{int(q*100)}', loss, logger=True)
            total_loss += loss
            
        avg_loss = total_loss / len(quantiles_to_check)
        self.log('test_loss', avg_loss, logger=True)
        return avg_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
import torch
import torch.nn as nn
import pytorch_lightning as pl
from .loss import pinball_loss

class ResidualBlock(nn.Module):
    """
    A simple residual block for an MLP.
    It consists of two linear layers with a ReLU activation in between.
    The input to the block is added to the output of the second linear layer.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the residual block.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The output tensor after the residual connection.
        """
        # The output of the block is added to the original input (the skip connection)
        residual = self.block(x)
        return self.relu(x + residual)

class QuantileRegressor(pl.LightningModule):
    """
    A PyTorch Lightning module for adaptive quantile regression using residual blocks.
    """
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        learning_rate: float = 1e-3,
        num_hidden_blocks: int = 2, # Changed from num_hidden_layers
    ):
        """
        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of neurons in each hidden layer.
            learning_rate (float): The learning rate for the optimizer.
            num_hidden_blocks (int): The number of residual blocks to use.
        """
        super().__init__()
        self.save_hyperparameters()

        # Input layer: projects concatenated (x, tau) to the hidden dimension
        self.input_layer = nn.Sequential(
            nn.Linear(self.hparams.input_size + 1, self.hparams.hidden_size),
            nn.ReLU()
        )

        # A sequence of residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(self.hparams.hidden_size) for _ in range(self.hparams.num_hidden_blocks)]
        )

        # Output layer: projects from the hidden dimension to the final prediction
        self.output_layer = nn.Linear(self.hparams.hidden_size, 1)

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.
        """
        # Concatenate the input features and the quantiles
        input_tensor = torch.cat([x, tau], dim=1)
        
        # Pass through the layers
        hidden = self.input_layer(input_tensor)
        hidden = self.residual_blocks(hidden)
        output = self.output_layer(hidden)
        
        return output

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Performs a single training step.
        """
        x, y = batch
        tau = torch.rand(x.size(0), 1, device=self.device)
        y_hat = self(x, tau)
        loss = pinball_loss(y_hat, y, tau)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int):
        """
        Performs a single validation step.
        """
        x, y = batch
        quantiles_to_check = torch.tensor([0.10, 0.50, 0.90], device=self.device)
        total_loss = 0
        
        for q in quantiles_to_check:
            tau = torch.full((x.size(0), 1), fill_value=q, device=self.device)
            y_hat = self(x, tau)
            loss = pinball_loss(y_hat, y, tau)
            self.log(f'val_loss_q{int(q*100)}', loss, on_epoch=True, logger=True)
            total_loss += loss
            
        avg_loss = total_loss / len(quantiles_to_check)
        self.log('val_loss', avg_loss, on_epoch=True, prog_bar=True, logger=True)
        return avg_loss

    def test_step(self, batch: tuple, batch_idx: int):
        """
        Performs a single test step.
        """
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
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
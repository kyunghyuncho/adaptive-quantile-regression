import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader, random_split

class PolynomialDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for generating and serving data from a
    polynomial function with heteroscedastic noise.

    The noise is heteroscedastic, meaning its variance is not constant.
    Specifically, the noise magnitude increases as the absolute value of x increases.
    This creates a more interesting and realistic scenario for quantile regression.
    """
    def __init__(
        self,
        degree: int = 3,
        noise_scale: float = 0.8,
        n_samples: int = 2000,
        x_range: tuple = (-4, 4),
        batch_size: int = 64,
        val_split: float = 0.2,
        test_split: float = 0.2,
        num_workers: int = 4
    ):
        """
        Args:
            degree (int): The degree of the polynomial to generate.
            noise_scale (float): A scaling factor for the noise.
            n_samples (int): The total number of samples to generate.
            x_range (tuple): The range from which to sample x values.
            batch_size (int): The batch size for the DataLoaders.
            val_split (float): The fraction of data to use for validation.
            test_split (float): The fraction of data to use for testing.
            num_workers (int): Number of workers for the DataLoader.
        """
        super().__init__()
        # save_hyperparameters() allows us to access these arguments
        # via self.hparams, and it saves them to the checkpoint.
        self.save_hyperparameters()

        self.coeffs = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # This method is called once per node. Good for downloads.
        # Since we generate data, we don't need to do anything here.
        pass

    def setup(self, stage: str = None):
        """
        This method is called on every GPU in a distributed setup.
        It's responsible for generating the data and splitting it.
        """
        # Ensure data is generated only once
        if self.train_dataset is not None:
            return

        # 1. Generate the true polynomial function
        # The coefficients are sampled from a standard normal distribution.
        self.coeffs = np.random.randn(self.hparams.degree + 1)

        # 2. Generate x values
        x_min, x_max = self.hparams.x_range
        x = np.random.uniform(x_min, x_max, self.hparams.n_samples)

        # 3. Generate y values with heteroscedastic noise
        y_true = np.polyval(self.coeffs, x)
        
        # The noise level is proportional to the absolute value of x.
        # This means the data points are more spread out further from the origin.
        noise = self.hparams.noise_scale * (1 + np.abs(x)) * np.random.randn(self.hparams.n_samples)
        y = y_true + noise

        # 4. Convert to PyTorch Tensors
        # We add an unsqueeze(1) to make them [n_samples, 1] instead of just [n_samples].
        # This is the standard shape for single-feature regression tasks.
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        # 5. Create a full TensorDataset
        dataset = TensorDataset(x_tensor, y_tensor)

        # 6. Split the dataset into training, validation, and test sets
        n_val = int(self.hparams.n_samples * self.hparams.val_split)
        n_test = int(self.hparams.n_samples * self.hparams.test_split)
        n_train = self.hparams.n_samples - n_val - n_test

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [n_train, n_val, n_test]
        )

    def train_dataloader(self):
        """Returns the DataLoader for the training set."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        """Returns the DataLoader for the validation set."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=True
        )

    def test_dataloader(self):
        """Returns the DataLoader for the test set."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=True
        )

# Example of how to use it (optional, for testing)
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Create an instance of the data module
    data_module = PolynomialDataModule(degree=4, noise_scale=1.0, n_samples=500)
    
    # The setup() method is called to generate the data
    data_module.setup()

    # Get the full dataset for plotting
    x_all = data_module.train_dataset.dataset.tensors[0].numpy()
    y_all = data_module.train_dataset.dataset.tensors[1].numpy()

    # Get the true function line
    x_line = np.linspace(data_module.hparams.x_range[0], data_module.hparams.x_range[1], 400)
    y_line = np.polyval(data_module.coeffs, x_line)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(x_all, y_all, alpha=0.3, label='Generated Data')
    plt.plot(x_line, y_line, color='r', linestyle='--', label='True Polynomial Function')
    plt.title('Sample Generated Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

    # You can also inspect a batch
    train_loader = data_module.train_dataloader()
    x_batch, y_batch = next(iter(train_loader))
    print(f"Shape of a batch of x: {x_batch.shape}")
    print(f"Shape of a batch of y: {y_batch.shape}")
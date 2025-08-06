import os
import yaml
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Import our custom modules
from src.data import PolynomialDataModule
from src.model import QuantileRegressor

def train(config):
    """
    The main training routine.

    Args:
        config (dict): A dictionary containing all hyperparameters and settings.
    """
    # --- Setup ---
    # Set seed for reproducibility
    pl.seed_everything(config['seed'])

    # Initialize the WandbLogger. This will automatically log all metrics,
    # hyperparameters, and save the model checkpoint to W&B.
    wandb_logger = WandbLogger(
        project=config['wandb']['project'],
        entity=config['wandb'].get('entity'), # Optional: your W&B entity
        config=config
    )

    # --- Data and Model ---
    # Initialize the DataModule
    data_module = PolynomialDataModule(
        degree=config['data']['degree'],
        noise_scale=config['data']['noise_scale'],
        n_samples=config['data']['n_samples'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )

    # Initialize the Model
    model = QuantileRegressor(
        input_size=1, # Since our x is a single feature
        hidden_size=config['model']['hidden_size'],
        learning_rate=config['training']['learning_rate'],
        num_hidden_blocks=config['model']['num_hidden_blocks']
    )

    # --- Callbacks ---
    # 1. ModelCheckpoint: Saves the best model based on validation loss.
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=config['training']['checkpoint_dir'],
        filename='best-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )

    # 2. EarlyStopping: Stops training if the validation loss doesn't improve.
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=config['training']['early_stopping_patience'],
        verbose=True,
        mode='min'
    )

    # --- Training ---
    # Initialize the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=config['training']['max_epochs'],
        log_every_n_steps=config['training']['log_every_n_steps'],
        accelerator='auto' # Automatically selects GPU if available
    )

    # Start the training process
    trainer.fit(model, datamodule=data_module)

    # Run a final test loop after training is complete
    trainer.test(model, datamodule=data_module, ckpt_path='best')

    # Close the wandb run
    wandb_logger.experiment.finish()
    print("Training finished. Best model checkpoint saved.")


if __name__ == '__main__':
    # --- Configuration Loading ---
    # We use argparse to point to a configuration file.
    # This is more flexible than hardcoding parameters.
    parser = argparse.ArgumentParser(description="Train an adaptive quantile regressor.")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to the configuration file (e.g., config.yaml)'
    )
    args = parser.parse_args()

    # Load settings from the YAML file
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        print("Please create a 'config.yaml' file.")
        exit(1)

    # Create checkpoint directory if it doesn't exist
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)

    # Start training
    train(config)
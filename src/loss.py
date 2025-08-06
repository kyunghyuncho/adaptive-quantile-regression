import torch

def pinball_loss(y_hat: torch.Tensor, y: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Pinball Loss, which is a metric used in quantile regression.

    The loss function is defined as:
    L_tau(y, y_hat) = (y - y_hat) * tau      if y > y_hat
                    = (y_hat - y) * (1 - tau)  if y <= y_hat

    This can be written more compactly using a max operation.

    Args:
        y_hat (torch.Tensor): The predicted quantiles. Shape: (batch_size, 1).
        y (torch.Tensor): The true values. Shape: (batch_size, 1).
        tau (torch.Tensor): The target quantiles. Can be a scalar or a tensor
                            of shape (batch_size, 1) for adaptive training.

    Returns:
        torch.Tensor: The mean pinball loss for the batch (a scalar tensor).
    """
    # Calculate the error between true values and predictions
    error = y - y_hat

    # The core of the pinball loss calculation
    # torch.max(a, b) computes the element-wise maximum of a and b.
    # This elegantly implements the two cases of the loss function.
    loss = torch.max((tau - 1) * error, tau * error)

    # Return the average loss over the batch
    return torch.mean(loss)

# Example of how to use it (optional, for testing)
if __name__ == '__main__':
    # --- Test Case 1: Simple values ---
    print("--- Test Case 1: Simple values ---")
    y_hat_simple = torch.tensor([[1.0], [2.0], [3.0]])
    y_simple = torch.tensor([[1.5], [1.5], [3.5]])
    tau_median = torch.tensor([[0.5]]) # Median (L1 loss)
    
    loss_val = pinball_loss(y_hat_simple, y_simple, tau_median)
    # Expected: 
    # y > y_hat: (1.5-1.0)*0.5 = 0.25
    # y < y_hat: (2.0-1.5)*(1-0.5) = 0.25
    # y > y_hat: (3.5-3.0)*0.5 = 0.25
    # Mean = (0.25 + 0.25 + 0.25) / 3 = 0.25
    print(f"Median Loss (tau=0.5): {loss_val.item():.4f}")
    assert abs(loss_val.item() - 0.25) < 1e-6, "Median loss calculation is incorrect"

    # --- Test Case 2: Higher quantile ---
    print("\n--- Test Case 2: Higher quantile ---")
    tau_high = torch.tensor([[0.9]]) # 90th percentile
    loss_val_high = pinball_loss(y_hat_simple, y_simple, tau_high)
    # Expected:
    # y > y_hat: (1.5-1.0)*0.9 = 0.45
    # y < y_hat: (2.0-1.5)*(1-0.9) = 0.05
    # y > y_hat: (3.5-3.0)*0.9 = 0.45
    # Mean = (0.45 + 0.05 + 0.45) / 3 = 0.3167
    print(f"High Quantile Loss (tau=0.9): {loss_val_high.item():.4f}")
    assert abs(loss_val_high.item() - 0.316666) < 1e-6, "High quantile loss calculation is incorrect"

    # --- Test Case 3: Adaptive quantiles per sample ---
    print("\n--- Test Case 3: Adaptive quantiles ---")
    tau_adaptive = torch.tensor([[0.1], [0.5], [0.9]])
    loss_val_adaptive = pinball_loss(y_hat_simple, y_simple, tau_adaptive)
    # Expected:
    # y > y_hat, tau=0.1: (1.5-1.0)*0.1 = 0.05
    # y < y_hat, tau=0.5: (2.0-1.5)*(1-0.5) = 0.25
    # y > y_hat, tau=0.9: (3.5-3.0)*0.9 = 0.45
    # Mean = (0.05 + 0.25 + 0.45) / 3 = 0.25
    print(f"Adaptive Quantile Loss: {loss_val_adaptive.item():.4f}")
    assert abs(loss_val_adaptive.item() - 0.25) < 1e-6, "Adaptive quantile loss calculation is incorrect"

    print("\nAll tests passed!")
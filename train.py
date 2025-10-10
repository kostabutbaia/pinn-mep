import torch

from torch import nn

def train_nn(optim: str, model: nn.Module, loss_fn, num_epochs: int, lr: float) -> None:
    if optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    losses = []

    for epoch in range(num_epochs):
        # Forward pass
        loss = loss_fn(model)
        losses.append(loss.item())
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.12f}')

    return losses
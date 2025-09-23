import torch

from torch import nn
import numpy as np

def train_pinn(model: nn.Module, x_train: torch.Tensor, num_epochs: int, lr: float, bound_weight: float) -> None:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []

    for epoch in range(num_epochs):
        u_pred = model(x_train)

        u_x = torch.autograd.grad(u_pred, x_train, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_train, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        pde_loss = torch.mean((u_xx + np.pi**2 * torch.sin(np.pi * x_train))**2)
        bound_loss = torch.mean(model(torch.tensor([.0]).view(-1, 1))**2 + model(torch.tensor([2.]).view(-1, 1))**2)
        loss = pde_loss + bound_weight*bound_loss

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.7f}')
    
    return losses

def train_nn(optim: str, model: nn.Module, x_train: torch.Tensor, y_train: torch.Tensor, num_epochs: int, lr: float) -> None:
    criterion = nn.MSELoss()
    if optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    losses = []

    for epoch in range(num_epochs):
        # Forward pass
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        losses.append(loss.item())
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.12f}')

    return losses
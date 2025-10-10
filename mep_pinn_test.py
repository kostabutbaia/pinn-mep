import torch
import random
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from MLP import Model
from train import train_nn
from mep import get_path

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def NEB():
    x_train = torch.linspace(0, 2, 100).view(-1, 1)
    x_train.requires_grad = True

    y_target = torch.sin(np.pi*x_train)

    model1 = Model([4], 'tanh')
    model2 = Model([4], 'tanh')

    def exact_error(model: nn.Module):
        return torch.mean((y_target-model(x_train))**2)

    def loss_fn_pinn(model: nn.Module):
        u_pred = model(x_train)

        u_x = torch.autograd.grad(u_pred, x_train, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_train, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        pde_loss = torch.mean((u_xx + np.pi**2 * torch.sin(np.pi * x_train))**2)
        bound_loss = torch.mean(model(torch.tensor([.0]).view(-1, 1))**2 + model(torch.tensor([2.]).view(-1, 1))**2)
        loss = pde_loss + bound_loss
        return loss
    
    train_nn('sgd', model1, loss_fn_pinn, 15000, 1e-3)
    train_nn('sgd', model2, loss_fn_pinn, 15000, 1e-3)

    x_plot = torch.linspace(-5, 7, 100).view(-1, 1)
    plt.plot(x_plot.tolist(), model1(x_plot).tolist(), 'b', label='Model 1')
    plt.plot(x_plot.tolist(), model2(x_plot).tolist(), 'g', label='Model 2')
    plt.plot(x_train.tolist(), y_target.tolist(), 'r--', label='Target')
    plt.grid()
    plt.legend()
    plt.show()
    
    num_pivots = 35
    models = get_path(
        model1,
        model2,
        num_pivots,
        loss_fn_pinn,
        1,
        [
            [3000, 1e-3],
        ],
        True,
        'NEB Algorithm'
    )

    errors = [exact_error(model).item() for model in models]
    losses = [loss_fn_pinn(model).item() for model in models]
    plt.plot(losses, 'b', label='MEP')
    plt.plot(errors, 'g', label='Exact Error')

    for i, model in enumerate(models):
        print(f'Training Model: {i}')
        train_nn('sgd', model, loss_fn_pinn, 1500, 1e-3)

    trained_losses = [loss_fn_pinn(model).item() for model in models]
    plt.plot(trained_losses, '.', label='Trained from MEP')

    plt.vlines(x=np.arange(0, len(trained_losses), 1), ymin=trained_losses, ymax=losses, color='r', linestyle='--', alpha=0.5)

    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    NEB()
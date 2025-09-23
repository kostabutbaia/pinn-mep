import copy
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from MLP import Model, get_lin_interp
from train import train_nn, train_pinn
from mep import get_path
from eng import train_pinn_engd

def pinn():
    x_train = torch.linspace(0, 2, 100).view(-1, 1)
    x_train.requires_grad = True

    y_target = torch.sin(np.pi*x_train)

    model1 = Model([15, 15], 'tanh')
    model2 = copy.deepcopy(model1)

    train_pinn(model1, x_train, 500, 1e-3, 1)
    train_pinn_engd(model2, x_train, 100, 1e-4)
    
    # plt.plot(x_train.detach(), y_target.detach(), 'r--', label='Target')
    # plt.plot(x_train.detach(), model1(x_train).detach(), 'b', label='Model gd')
    # plt.plot(x_train.detach(), model2(x_train).detach(), 'g', label='Model ngd')

    # plt.legend()
    # plt.grid()
    # plt.show()

# ------------------------------ NEB

def NEB():
    x_train = torch.linspace(0, 2, 100).view(-1, 1)
    x_train.requires_grad = True

    y_target = torch.sin(np.pi*x_train)

    model1 = Model([15, 15], 'tanh')
    model2 = copy.deepcopy(model1)


    train_pinn(model1, x_train, 5000, 1e-3, 1)
    # train_pinn(model2, x_train, 50000, 1e-3, 1)
    train_pinn_engd(model2, x_train, 100, 1e-4)

    # def loss_fn(model: nn.Module):
    #     criterion = nn.MSELoss()
    #     y_pred = model(x_train)
    #     loss = criterion(y_pred, y_target)
    #     return loss

    def loss_fn_pinn(model: nn.Module):
        u_pred = model(x_train)

        u_x = torch.autograd.grad(u_pred, x_train, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_train, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        pde_loss = torch.mean((u_xx + np.pi**2 * torch.sin(np.pi * x_train))**2)
        bound_loss = torch.mean(model(torch.tensor([.0]).view(-1, 1))**2 + model(torch.tensor([2.]).view(-1, 1))**2)
        loss = pde_loss + bound_loss
        return loss
    
    num_pivots = 35
    models = get_path(
        model1,
        model2,
        num_pivots,
        loss_fn_pinn,
        1,
        [
            [300, 0.001],
            # [1000, 0.01],
            # [1000, 0.001],
        ],
        True,
        'NEB Algorithm'
    )

    alphas = np.linspace(0, 1, num_pivots+2)
    # lin_errors = get_lin_interp(loss_fn_pinn, model1, model2, num_pivots+2)

    errors = [loss_fn_pinn(model).item() for model in models]
    plt.plot(alphas, errors, 'b', label='MEP')
    # plt.plot(alphas, lin_errors, 'r', label='Linear Interpolation')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    NEB()
    # pinn()
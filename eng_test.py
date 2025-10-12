import random
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from MLP import Model
from eng import train_pinn_engd
from train import train_nn

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def main():
    x_train = torch.linspace(0, 2, 100).view(-1, 1)
    x_train.requires_grad = True

    y_target = torch.sin(2*np.pi*x_train)

    model = Model([15, 15], 'sin')

    def loss_fn_pinn(model: nn.Module):
        u_pred = model(x_train)

        u_x = torch.autograd.grad(u_pred, x_train, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_train, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        pde_loss = torch.mean((u_xx + (2*np.pi)**2 * torch.sin(2*np.pi * x_train))**2)
        bound_loss = torch.mean(model(torch.tensor([.0]).view(-1, 1))**2 + model(torch.tensor([2.]).view(-1, 1))**2)
        loss = pde_loss + bound_loss
        return loss
    
    train_pinn_engd(model, loss_fn_pinn, x_train, 300, 1e-4)
    # train_nn('adam', model, loss_fn_pinn, 25000, 1e-4)

    # x_plot = torch.linspace(-5, 7, 100).view(-1, 1)
    plt.plot(x_train.tolist(), model(x_train).tolist(), 'g', label='Model')
    plt.plot(x_train.tolist(), y_target.tolist(), 'r--', label='Target')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
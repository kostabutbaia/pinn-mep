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
torch.set_default_dtype(torch.float64)

def main():
    x_train = torch.linspace(0, 2, 100).view(-1, 1)
    x_train.requires_grad = True

    y_target = torch.sin(2*np.pi*x_train)

    model = Model([15, 15], 'tanh')

    def loss_fn_pinn(model: nn.Module):
        u_pred = model(x_train)

        u_x = torch.autograd.grad(u_pred, x_train, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_train, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        pde_loss = torch.mean((u_xx + (2*np.pi)**2 * torch.sin(2*np.pi * x_train))**2)
        bound_loss = torch.mean(model(torch.tensor([.0]).view(-1, 1))**2 + model(torch.tensor([2.]).view(-1, 1))**2)
        loss = pde_loss + bound_loss
        return loss
    
    losses = train_pinn_engd(model, loss_fn_pinn, x_train, 150, 1e-4)

    plt.plot(losses)
    plt.grid()
    plt.yscale('log')
    plt.title('E-NGD Training Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


if __name__ == '__main__':
    main()
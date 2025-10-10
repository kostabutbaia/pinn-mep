import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from mep import get_path, plot_distance_between_pivots

def loss_profile(x, y):
    return (x**2 - 1)**2 + y**2

class TwoMinimaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return torch.ones_like(x)*self.a**2
    
def two_minima_example():
    model1 = TwoMinimaModel()
    model2 = TwoMinimaModel()

    with torch.no_grad():
        model1.a.fill_(1.2)
        model1.b.fill_(1.2)

    with torch.no_grad():
        model2.a.fill_(-1.2)
        model2.b.fill_(1.2)

    def loss_fn(model: nn.Module):
        a = model.a
        b = model.b
        return loss_profile(a, b)
    
    num_pivots = 35
    models = get_path(
        model1,
        model2,
        num_pivots,
        loss_fn,
        10,
        [
            [3000, 1e-3],
        ],
        True,
        'NEB Algorithm'
    )

    alphas = np.linspace(0, 1, num_pivots+2)
    errors = [loss_fn(model).item() for model in models]
    plt.plot(alphas, errors, 'b', label='MEP')
    plt.legend()
    plt.grid()
    plt.show()

    plot_3d(models)

    plot_distance_between_pivots(models)

def plot_3d(models):
    x = np.linspace(-1.5, 1.5, 200)
    y = np.linspace(-1.6, 1.6, 200)
    X, Y = np.meshgrid(x, y)
    Z = loss_profile(X, Y)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    path_x = torch.stack([m.a for m in models]).squeeze()
    path_y = torch.stack([m.b for m in models]).squeeze()
    path_z = loss_profile(path_x, path_y)

    ax.plot(path_x.tolist(), path_y.tolist(), path_z.tolist(), color='red', linewidth=3, label='Path', alpha=1.0)
    # Surface
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

    ax.scatter([1.2, -1.2], [1.2, 1.2], loss_profile(np.array([1.2,-1.2]), np.array([1.2,1.2])), color='blue', marker='o', label='Endpoints', zorder=10)

    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('loss')
    ax.set_title('loss landscape')
    ax.legend()

    plt.show()

if __name__ == '__main__':
    two_minima_example()
from torch import nn
import torch
import numpy as np

def interpolate_model(model1: nn.Module, model2: nn.Module, t: float) -> nn.Module:
    model_interp = Model(model1._hidden_layers, model1._activation)
    with torch.no_grad():
        for p1, p2, p_interp in zip(model1.parameters(), model2.parameters(), model_interp.parameters()):
            p_interp.copy_((1 - t) * p1 + t * p2)
    return model_interp

def get_lin_interp(loss_fn, model1: nn.Module, model2: nn.Module, count: int) -> tuple[list[float], list[nn.Module]]:
    t_space = np.linspace(0, 1, count)
    losses = []
    models = []
    for t in t_space:
         model = interpolate_model(model1, model2, t)
         models.append(model)
         loss = loss_fn(model).item()
         losses.append(loss)
    return losses, models

class Model(nn.Module):
    def __init__(self, hidden_layers, activation, in_features=1, out_features=1):
        super().__init__()
        if activation == 'tanh':
            activation = nn.Tanh()
        elif activation == 'relu':
            activation = nn.ReLU()
        elif activation == 'sin':
            Sine = type('Sine', (nn.Module,), {'forward': lambda self, x: torch.sin(x)})
            activation = Sine()
        elif activation == 'sigmoid':
            activation = nn.Sigmoid()

        self._activation = activation
        self._hidden_layers = hidden_layers
        
        layer_sizes = [in_features] + hidden_layers + [out_features]
        layers = []

        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(activation)
        
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
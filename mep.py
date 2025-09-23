import copy
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from eng import build_energy_gram, line_search
from anim import create_anim_gif

def get_path(
        model1: nn.Module,
        model2: nn.Module,
        num_pivots: int,
        loss_fn,
        k_spring: float,
        cycles: list[tuple[int, int]],

        make_gif: bool,
        gif_name: str = None
) -> list[nn.Module]:
    
    vec_a = parameters_to_vector(model1.parameters()).detach()
    vec_b = parameters_to_vector(model2.parameters()).detach()

    frames = []

    pivots = []
    alphas = np.linspace(0, 1, num_pivots+2)
    alphas = alphas

    for alpha in alphas:
        pivots.append((1-alpha)*vec_a + alpha*vec_b)

    for i, cycle in enumerate(cycles):
        print(f"[Cycle {i+1}]  Epochs = {cycle[0]}  |  Learning Rate = {cycle[1]}")
        for _ in range(cycle[0]):
            losses = []
            grads = []
            for p in pivots:
                loss_val, grad_vec = get_loss_grad(p, loss_fn, model1)
                losses.append(loss_val)
                grads.append(grad_vec)
            
            new_pivots = []
            for i in range(1, num_pivots+1):
                tau = pivots[i+1] - pivots[i] if losses[i+1] > losses[i-1] else pivots[i] - pivots[i-1]
                tau = tau / (tau.norm() + 1e-12)

                grad_para = torch.dot(grads[i], tau) * tau
                F_L_perp = -(grads[i] - grad_para)

                d_prev = (pivots[i] - pivots[i-1]).norm()
                d_next = (pivots[i+1] - pivots[i]).norm()
                Fs = -k_spring * (d_prev - d_next)
                F_S_para = Fs * tau

                step = cycle[1] * (F_L_perp + F_S_para)
                new_pivots.append((pivots[i] + step).detach().clone())
            
            new_pivots = [pivots[0]] + new_pivots + [pivots[-1]]
            pivots = new_pivots
            if make_gif:
                alphas = np.linspace(0, 1, num_pivots+2)
                errors = [loss_fn(model).item() for model in pivots_to_models(pivots, model1)]
                frames.append((alphas, errors))
    if make_gif:
        create_anim_gif(gif_name, frames)
  
    return pivots_to_models(pivots, model1)

def get_loss_grad(vec: torch.Tensor, loss_fn, model: nn.Module) -> tuple[float, torch.Tensor]:
    temp_model = copy.deepcopy(model)
    vector_to_parameters(vec, temp_model.parameters())
    temp_model.zero_grad()
    loss = loss_fn(temp_model)
    loss.backward()
    grad_vec = parameters_to_vector([p.grad for p in temp_model.parameters()])
    return float(loss.item()), grad_vec.detach()

def pivots_to_models(pivots: list, model: nn.Module) -> list[nn.Module]:
    models = []
    for p in pivots:
        m = copy.deepcopy(model)
        vector_to_parameters(p, m.parameters())
        models.append(m)
    return models
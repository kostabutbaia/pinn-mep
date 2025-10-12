import torch
from torch import nn
import numpy as np

def train_pinn_engd(
    model: torch.nn.Module,
    loss_fn,
    x_train: torch.Tensor,
    num_epochs: int,
    lr: float
) -> list:

    losses = []
    params = [p for p in model.parameters() if p.requires_grad]
    P = torch.cat([p.reshape(-1) for p in params]).numel()

    for epoch in range(num_epochs):
        loss = loss_fn(model)
        losses.append(loss.item())

        # Get gradient of loss
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grad_vec = torch.cat([g.reshape(-1) for g in grads])

        # Gram matrix G_E
        G = build_energy_gram(model, x_train, torch.tensor([[0.0],[2.0]], dtype=x_train.dtype))
        # G = torch.eye(P) # for vanilla gradient descent set G to identity matrix

        # Solve least squares for pseudo-inverse: G phi = grad_vec
        natural_grad = torch.linalg.lstsq(G, grad_vec.unsqueeze(1)).solution
        natural_grad = natural_grad.squeeze()

        lr = line_search(model, loss_fn, natural_grad)

        # Update parameters with natural gradient
        with torch.no_grad():
            idx = 0
            for p in model.parameters():
                size = p.numel()
                p -= lr * natural_grad[idx:idx+size].view_as(p)
                idx += size

        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.7f}")

    return losses

def build_energy_gram(model: nn.Module, x_interior, x_boundary):
    params = [p for p in model.parameters() if p.requires_grad]
    P = torch.cat([p.reshape(-1) for p in params]).numel()
    G = torch.zeros((P, P))

    # Interior second derivative
    u = model(x_interior)
    u_x = torch.autograd.grad(u, x_interior, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_interior, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]

    # Boundary values
    u_b = model(x_boundary)

    for k in range(len(x_interior)):
        d_u_xx = torch.autograd.grad(u_xx[k], model.parameters(), retain_graph=True, create_graph=True, allow_unused=True)
        d_u_xx = torch.cat([t.reshape(-1) if t is not None else torch.tensor([0.]) for t in d_u_xx])

        G += 1/len(x_interior)*torch.outer(d_u_xx, d_u_xx)

    for m in range(len(x_boundary)):
        d_u_b = torch.autograd.grad(u_b[m], model.parameters(), retain_graph=True, create_graph=True, allow_unused=True)
        d_u_b = torch.cat([t.reshape(-1) if t is not None else torch.tensor([0.]) for t in d_u_b])

        G += 1/len(x_boundary)*torch.outer(d_u_b, d_u_b)

    return G

def line_search(model, loss_fn, natural_grad, steps=20):
    eta_candidates = torch.logspace(np.log10(1e-5), np.log10(1), steps=steps)
    best_loss = None
    best_eta = 1

    for eta in eta_candidates:
        idx = 0
        # Temporarily apply update
        with torch.no_grad():
            for p in model.parameters():
                size = p.numel()
                p -= eta * natural_grad[idx:idx+size].view_as(p)
                idx += size

        # Compute loss after update
        curr_loss = loss_fn(model)

        # Restore parameters
        idx = 0
        with torch.no_grad():
            for p in model.parameters():
                size = p.numel()
                p += eta * natural_grad[idx:idx+size].view_as(p)
                idx += size

        if best_loss is None or curr_loss < best_loss:
            best_loss = curr_loss
            best_eta = eta

    return best_eta
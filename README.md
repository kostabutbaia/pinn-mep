### Finding Minimum Energy Paths
In `mep.py` the function `get_path` finds minimum energy path between two minimizers $\theta_1$ and $\theta_2$ with NEB algorithm, it starts with linear interpolation

$$p=\{(1-\alpha)\theta_1 + \alpha\theta_2: \alpha\in(0,1) \}$$

pivots in $p$ are connected with springs with $k$ spring constant. for each $p_i\in p$ tangent vector is defined:

$$\tau_i=\begin{cases}
  p_{i+1}-p_i & \text{if } L(p_{i+1}) > L(p_{i-1}) \\
  p_i-p_{i-1}  & \text{else }
\end{cases}$$

each $p_i$ is updated with forces, spring force parallel to $\tau_i$ and loss gradient force perpendicular to $\tau_i$, the spring force is:

$$F_i^S=-k(||p_i-p_{i-1}||-||p_{i+1}-p_i||)\tau_i$$

for perpendicular force from loss
$$F_i^L=-(\nabla_\theta L(p_i)-(\nabla_\theta L(p_i)\cdot\tau_i)\tau_i)$$
Then we update pivots with with learning rate $\eta$

$$p_i\leftarrow p_i + \eta(F_i^L+F_i^S)$$

to get minimum energy path in function space, use natural gradient $\nabla_\theta\to\nabla_E$

### Natural Gradient Descent
In `eng.py` the function `train_pinn_engd` updates the parameters $\theta \in \mathbb{R}^P$ using natural gradient descent

$$\theta\leftarrow\theta-\eta\nabla_EL(\theta)$$

Where natural gradient is defined as:

$$\nabla_EL(\theta)=G^+_E(\theta)\nabla_\theta L(\theta)$$

Where $G_E\in\mathbb{R}^{P\times P}$ is energy gram matrix which for 1D poisson $u_{xx}=f$ solving on $\Omega$ takes the form

$$G_E(\theta)_{ij}=\int_\Omega(\partial_x^2\partial_{\theta_i}u_\theta(x))(\partial_x^2\partial_{\theta_j}u_\theta(x))dx+\int_{\partial \Omega}\partial_{\theta_i}u_\theta(s)\partial_{\theta_j}u_\theta(s)ds$$

We calculate this integral for the Gram matrix using the collocation points on the interior $\{x_i\}_{i=1}^N$ and on the boundary $\{x_i\}_{i=1}^M$

$$G_E(\theta)_{ij}\approx\frac{1}{N}\sum_{k=1}^N(\partial_x^2\partial_{\theta_i}u_\theta(x_k))(\partial_x^2\partial_{\theta_j}u_\theta(x_k))+\frac{1}{M}\sum_{m=1}^M\partial_{\theta_i}u_\theta(x_m)\partial_{\theta_j}u_\theta(x_m)$$

To get the pseudo inverse of $G_E$ we solve the following least squares with `torch.linalg.lstsq`

$$\nabla_EL(\theta)=\argmin_{\psi\in\mathbb{R}^P}||G_E(\theta)\psi-\nabla_\theta L(\theta)||^2$$

Constant learning rate $\eta$ may overshoot, leading to oscillations or divergence. Do line search for optimal $\eta$ on interval $[0,1]$, i.e. choose $\eta\in[0,1]$ which gives lowest error after parameter update

$$\eta^* ← \argmin_{\eta\in[0,1]} L(\theta − η\nabla_E L(\theta))$$
# Pseudocode

## Dependency Noise Model
```latex
\begin{lstlisting}[language=Python, caption=Implementation of dependency noise model in PyTorch-style., label=code: dependency noise model]
import torch
from typing import List


def dependency_noise_model(shape: List[int], lambda_i: float = 0.01, random_search_times: int = 10, linear_search_times: int = 15):
    """Dependency Noise Model for a batch.
    Args:
        shape: A list of int indicates shape as (batches, frames, ...).
        lambda_i: The hyper-parameters to control the dependency noise model. 
            If lambda_i = 0., all frames share the same noise, and a still video clip will be generated.
            If lambda_i -> +inf, all frames sample independency noise, a random video clip will be generated.
        random_search_times: The number of times to sample in the random search phase.
        linear_search_times: The number of times to update in the linear search phase.
    Returns: A sequence of dependency noise tensors with the shape of (batches, frames, ...)
    """
    batches, frames, *others = shape

    def get_progressive_noise(previous_x):
        noise_x = noise = torch.randn(batches, *others)
        
        def compute_error(noise):
            return (torch.kl_div(noise, previous_x, log_target=True).mean(dim=[1, 2, 3]) - lambda_i).abs()

        def compose_noise(previous, noise_x, lambda_i):
            return torch.sqrt(lambda_i) * previous + torch.sqrt(1-lambda_i) * noise_x

        error_bound = torch.tensor([torch.inf] * batches)
        for _ in range(random_search_times):
            # Coarse Random Search Phase
            error = compute_error(noise)
            index = error < error_bound
            noise_x[index] = noise[index]
            error_bound[index] = error[index]
            noise = torch.randn(batches, *others)

        alpha_i  = torch.tensor([0.1] * batches)
        step_size = torch.tensor([0.1] * batches)
        for _ in range(linear_search_times):
            # Linear Search Phase
            error = compute_error(compose_noise(previous_x, noise_x, alpha_i))
            index = error <= error_bound
            index_mask = torch.logical_not(index)
            alpha_i[index] = alpha_i[index] + step_size[index]
            alpha_i[index_mask] = alpha_i[index_mask] - 0.8 * step_size[index_mask]
            step_size[index_mask] = step_size[index_mask] * 0.2
            error_bound[index] = error[index]
        return compose_noise(previous_x, noise_x, alpha_i)
    
    noises = []
    for index in range(frames):
        noises.append(torch.randn(batches, *others) if index == 0 else get_progressive_noise(noises[-1]))
    return torch.stack(noises, dim=1)
\end{lstlisting}
```

## Temporal Momentum Attention


```latex
\begin{lstlisting}[language=Python, caption=Implementation of temporal momentum attention in PyTorch-style., label=code: temporal momentum attention]
import torch
from einops import einsum


def efficient_temporal_momentum_attention(x: torch.Tensor, momentum: float = 0.98):
    """Efficient Temporal Momentum Attention with Matrix Operation.
    Args:
        x: torch.Tensor with the shape of (frames, ...).
        momentum: The hyper-parameters to control temporal momentum attention.
            If momentum = 0.0, temporal momentum attention decays to self-attention.
            If momentum = 1.0, temporal momentum attention decays to cross-frame attention.
    Returns: The momentum shifted tensor with the shape of (frames, ...).
    """
    # Build U matrix.
    exp_mu = torch.pow(torch.tensor([momentum,] * len(x)), exponent=torch.arange(len(x)))
    exp_mu_matrix = torch.stack([torch.roll(exp_mu, i) for i in range(len(exp_mu))]).T
    U = torch.tril(exp_mu_matrix)
    
    # Matrix multiply
    x[1:] = x[1:] * (1 - momentum)
    
    return einsum("ff, fbnc -> fbnc", U, x) 

def temporal_momentum_attention(x: torch.Tensor, momentum: float = 0.98):
    """Momentum Attention with `for` loop.
    Args:
        x: torch.Tensor with the shape of (frames, ...).
        momentum: The hyper-parameters to control temporal momentum attention.
            If momentum = 0.0, temporal momentum attention decays to self-attention.
            If momentum = 1.0, temporal momentum attention decays to cross-frame attention.
    Returns: The momentum shifted tensor with the shape of (frames, ...).
    """
    return torch.stack([x[index] if index == 0 else x[index-1] * momentum + x[index] * (1-momentum) for index in range(len(x))])
\end{lstlisting}
```
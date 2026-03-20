"""
Capacity-inspired latency model: ℓ(θ, λ) = a(θ) + b(θ) / (c(θ) − λ).

Setup θ = (tensor_parallel_size, thread_percentage); load λ = load_rps.
a, b, c are positive and depend only on θ. This gives:
  - interpretable capacity c(θ) (max sustainable RPS)
  - closed-form ∂ℓ/∂λ = b(θ) / (c(θ) − λ)² for the routing optimizer

Regimes:
  - Stable (λ < c(θ), throughput ≈ load): Model is trained and accurate here.
    Use min_throughput_load_ratio when training so only these points are fitted.
  - Near capacity (λ → c(θ)): Latency rises; formula and gradients are valid.
  - Overload (λ ≥ c(θ)): Denominator is clamped to eps; predicted latency is
    very high and ∂ℓ/∂λ is zero. Do not rely on the model in this regime—either
    keep the optimizer in the stable region (e.g. constraint λ_i < c(θ_i)) or
    treat overload separately (e.g. penalty or feasibility gate).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CapacityInspiredLatencyModel(nn.Module):
    """
    Latency = a(θ) + b(θ) / (c(θ) − λ), with a, b, c from setup θ = (tp, thread_pct).
    Input x: [tp, thread_pct, load_rps]. Use raw load_rps (not log) so c and λ share units.

    Trust the model when λ < c(θ) (stable / near-capacity). When λ ≥ c(θ) (overload),
    output is clamped to a large value and gradients w.r.t. load are zero.
    """

    def __init__(self, setup_dim: int = 2, hidden: int = 32, eps: float = 1e-2, capacity_init: float = 50.0):
        super().__init__()
        self.eps = eps
        self.capacity_init = capacity_init  # so c starts high and learns down toward true capacity

        def make_positive_net():
            net = nn.Sequential(
                nn.Linear(setup_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
            return net

        self.a_net = make_positive_net()
        self.b_net = make_positive_net()
        self.c_net = make_positive_net()

        # Initialize c_net bias so c is large initially (above typical load)
        with torch.no_grad():
            self.c_net[-1].bias.fill_(capacity_init)

    def _positive(self, raw: torch.Tensor) -> torch.Tensor:
        return F.softplus(raw) + self.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 3) — [tp, thread_pct, load_rps]
        Returns: (N, 1) log(latency_ms) for stable training.
        """
        setup = x[:, :2]
        load = x[:, 2:3]

        a = self._positive(self.a_net(setup))
        b = self._positive(self.b_net(setup))
        c = self._positive(self.c_net(setup))

        denom = torch.clamp(c - load, min=self.eps)
        latency_ms = a + b / denom
        return torch.log(latency_ms.clamp(min=self.eps))

    def forward_latency_ms(self, x: torch.Tensor) -> torch.Tensor:
        """Return latency in ms (not log). Useful for evaluation and optimizer."""
        return torch.exp(self.forward(x))

    def d_latency_d_load(self, x: torch.Tensor) -> torch.Tensor:
        """
        ∂ℓ/∂λ in ms per RPS. x: (N, 3) [tp, thread_pct, load_rps].
        Returns (N, 1). Use with λ_i = λ * w_i for ∂L/∂w.
        """
        setup = x[:, :2]
        load = x[:, 2:3]

        a = self._positive(self.a_net(setup))
        b = self._positive(self.b_net(setup))
        c = self._positive(self.c_net(setup))

        denom = torch.clamp(c - load, min=self.eps)
        # ℓ = a + b/denom, denom = c−λ  =>  dℓ/dλ = (dℓ/d(denom)) * (d(denom)/dλ) = (-b/denom²)*(-1) = b/denom² when c>λ
        in_region = (c > load).float()
        d_latency_d_denom = -b / (denom ** 2)
        d_denom_d_load = -in_region  # d(c−λ)/dλ = -1
        d_latency_d_load = d_latency_d_denom * d_denom_d_load
        return d_latency_d_load

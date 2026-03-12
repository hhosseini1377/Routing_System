import torch
import torch.nn as nn
import torch.nn.functional as F

class QueueInspiredLogTTFTModel(nn.Module):
    def __init__(self, hidden: int = 16, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

        self.a_net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.b_net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.c_net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        setup = x[:, :2]
        load = x[:, 2:3]

        a = F.softplus(self.a_net(setup)) + self.eps
        b = F.softplus(self.b_net(setup)) + self.eps
        c = F.softplus(self.c_net(setup)) + self.eps

        denom = torch.clamp(c - load + self.eps, min=self.eps)
        ttft = a + b / denom
        pred_log_ttft = torch.log1p(ttft.clamp(min=0.0))

        return pred_log_ttft
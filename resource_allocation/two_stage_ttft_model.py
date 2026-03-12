import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden: int = 32, out_features: int = 1, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Linear(in_features, hidden),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers += [
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers += [nn.Linear(hidden, out_features)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoStageTTFTModel(nn.Module):
    """
    Input:
        x = [tp_level, thread_percentage, load]

    Gate predicts p_over (probability of overload).
    If p_over < threshold: large penalty (gate predicted stable when overloaded).
    If p_over >= threshold: use single MLP regressor to predict log1p(TTFT).
    """
    def __init__(
        self,
        in_features: int = 3,
        gate_hidden: int = 16,
        expert_hidden: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.gate = MLP(
            in_features=in_features,
            hidden=gate_hidden,
            out_features=1,
            dropout=dropout,
        )

        self.regressor = MLP(
            in_features=in_features,
            hidden=expert_hidden,
            out_features=1,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor):
        gate_logits = self.gate(x)
        p_over = torch.sigmoid(gate_logits)
        pred_log_ttft = self.regressor(x)
        return {
            "pred_log_ttft": pred_log_ttft,
            "p_over": p_over,
        }

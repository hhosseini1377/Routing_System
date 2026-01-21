from dataclasses import dataclass


@dataclass
class RouterModelConfig:
    dropout_rate: float = 0.1
    classifier_dropout: bool = True
    layers_to_freeze: int = 4
    freeze_layers: bool = False
    freeze_embedding: bool = False
    classifier_type: str = "linear"  # Options: "linear" or "mlp"
    mlp_hidden_size: int = 512  # Hidden layer size for MLP classifier
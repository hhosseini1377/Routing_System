import torch
import torch.nn as nn


def compute_losses(
    model_out: dict,
    y_log_ttft: torch.Tensor,          # [B, 1]
    throughput_load_ratio: torch.Tensor,  # [B, 1]
    feasibility_threshold: float = 0.98,
    p_over_threshold: float = 0.5,
    large_penalty: float = 100.0,
):
    """
    If p_over < threshold (gate predicted stable) when actually overloaded: large penalty.
    Otherwise (stable regime): compute TTFT regression loss with the single MLP.
    """
    pred_log_ttft = model_out["pred_log_ttft"]
    p_over = model_out["p_over"]

    overloaded = (throughput_load_ratio < feasibility_threshold).float()
    stable_mask = (throughput_load_ratio >= feasibility_threshold).float()

    # Penalty: overloaded but gate predicted stable (p_over < p_over_threshold)
    wrong_stable = overloaded * (p_over < p_over_threshold).float()
    penalty_loss = large_penalty * wrong_stable.mean()

    # Regression: only on stable samples (ratio >= feasibility_threshold)
    if stable_mask.sum() > 0:
        reg_loss_per_sample = nn.HuberLoss(delta=1.0, reduction="none")(pred_log_ttft, y_log_ttft)
        reg_loss = (reg_loss_per_sample * stable_mask).sum() / stable_mask.sum()
    else:
        reg_loss = torch.tensor(0.0, device=pred_log_ttft.device)

    total_loss = reg_loss + penalty_loss

    return {
        "total_loss": total_loss,
        "reg_loss": reg_loss,
        "penalty_loss": penalty_loss,
    }

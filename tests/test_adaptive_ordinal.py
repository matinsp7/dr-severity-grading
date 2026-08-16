import torch

from src.loss.adaptive_ordinal import (
    AdaptiveOrdinalLoss,
)


def main():

    batch_size = 4

    outputs = {
        "class_logits": torch.randn(
            batch_size,
            5,
            requires_grad=True,
        ),
        "ordinal_logits": torch.randn(
            batch_size,
            4,
            requires_grad=True,
        ),
        "global_ordinal_logits": torch.randn(
            batch_size,
            4,
            requires_grad=True,
        ),
        "local_ordinal_logits": torch.randn(
            batch_size,
            4,
            requires_grad=True,
        ),
    }

    targets = torch.tensor(
        [0, 1, 2, 4]
    )

    criterion = AdaptiveOrdinalLoss(
        lambda_ordinal=1.0,
        lambda_boundary=0.5,
        boundary_alpha=2.0,
    )

    loss = criterion(
        outputs,
        targets,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()

    for tensor in outputs.values():
        assert tensor.grad is not None

    print(
        "Adaptive ordinal loss: OK"
    )

    print(
        f"loss = {loss.item():.6f}"
    )


if __name__ == "__main__":
    main()
import torch

from src.models.dual_branch import DualBranchDR


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = DualBranchDR(
        global_backbone=(
            "swin_tiny_patch4_window7_224"
        ),
        local_backbone="efficientnet_b0",
        num_classes=5,
        image_size=384,
        fusion_dim=512,
        dropout=0.2,
        pretrained=True,
    ).to(device)

    model.train()

    images = torch.randn(
        2,
        3,
        384,
        384,
        device=device,
    )

    targets = torch.randint(
        0,
        5,
        (2,),
        device=device,
    )

    outputs = model(images)

    assert outputs["class_logits"].shape == (
        2,
        5,
    )

    assert outputs["ordinal_logits"].shape == (
        2,
        4,
    )

    criterion = __import__(
        "src.loss.adaptive_ordinal",
        fromlist=["AdaptiveOrdinalLoss"],
    ).AdaptiveOrdinalLoss()

    loss = criterion(
        outputs,
        targets,
    )

    assert torch.isfinite(loss)

    loss.backward()

    print("Dual-branch forward/backward: OK")
    print(
        "class:",
        outputs["class_logits"].shape,
    )
    print(
        "ordinal:",
        outputs["ordinal_logits"].shape,
    )
    print(
        "loss:",
        loss.item(),
    )


if __name__ == "__main__":
    main()
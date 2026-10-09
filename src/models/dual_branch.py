import torch
import torch.nn as nn
import timm


class DualBranchDR(nn.Module):
    """
    Global-local dual-branch network for diabetic retinopathy grading.

    Global branch:
        Swin-Tiny for global retinal context.

    Local branch:
        EfficientNet-B0 for fine-grained visual features.

    Heads:
        - 5-class classification head
        - fused ordinal head
        - global ordinal head
        - local ordinal head
    """

    def __init__(
        self,
        global_backbone: str,
        local_backbone: str,
        num_classes: int = 5,
        image_size: int = 384,
        fusion_dim: int = 512,
        dropout: float = 0.2,
        pretrained: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_boundaries = num_classes - 1

        # ---------------------------------------------------------
        # Global branch
        # ---------------------------------------------------------
        self.global_branch = timm.create_model(
            global_backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            img_size=image_size,
        )

        # ---------------------------------------------------------
        # Local branch
        # ---------------------------------------------------------
        self.local_branch = timm.create_model(
            local_backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        global_dim = self.global_branch.num_features
        local_dim = self.local_branch.num_features

        # ---------------------------------------------------------
        # Project both branches into a shared latent space
        # ---------------------------------------------------------
        self.global_projection = nn.Sequential(
            nn.Linear(global_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        self.local_projection = nn.Sequential(
            nn.Linear(local_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        # ---------------------------------------------------------
        # Adaptive cross-branch fusion
        # ---------------------------------------------------------
        self.gate = nn.Sequential(
            nn.Linear(
                fusion_dim * 2,
                fusion_dim,
            ),
            nn.GELU(),
            nn.Linear(
                fusion_dim,
                fusion_dim,
            ),
            nn.Sigmoid(),
        )

        self.fusion_norm = nn.LayerNorm(fusion_dim)

        self.dropout = nn.Dropout(dropout)

        # ---------------------------------------------------------
        # Classification head
        # ---------------------------------------------------------
        self.classifier = nn.Linear(
            fusion_dim,
            num_classes,
        )

        # ---------------------------------------------------------
        # Ordinal heads
        # ---------------------------------------------------------
        self.ordinal_head = nn.Linear(
            fusion_dim,
            self.num_boundaries,
        )

        self.global_ordinal_head = nn.Linear(
            fusion_dim,
            self.num_boundaries,
        )

        self.local_ordinal_head = nn.Linear(
            fusion_dim,
            self.num_boundaries,
        )

    def forward(self, x):
        global_features = self.global_branch(x)

        local_features = self.local_branch(x)

        global_features = self.global_projection(
            global_features
        )

        local_features = self.local_projection(
            local_features
        )

        branch_features = torch.cat(
            [
                global_features,
                local_features,
            ],
            dim=1,
        )

        gate = self.gate(branch_features)

        fused_features = (
            gate * global_features
            + (1.0 - gate) * local_features
        )

        fused_features = self.fusion_norm(
            fused_features
        )

        fused_features = self.dropout(
            fused_features
        )

        class_logits = self.classifier(
            fused_features
        )

        ordinal_logits = self.ordinal_head(
            fused_features
        )

        global_ordinal_logits = (
            self.global_ordinal_head(
                global_features
            )
        )

        local_ordinal_logits = (
            self.local_ordinal_head(
                local_features
            )
        )

        return {
            "class_logits": class_logits,
            "ordinal_logits": ordinal_logits,
            "global_ordinal_logits": global_ordinal_logits,
            "local_ordinal_logits": local_ordinal_logits,
        }
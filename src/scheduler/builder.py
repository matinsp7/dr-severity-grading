from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)

def build_scheduler(cfg, optimizer):
    schedulers = {
        "step": StepLR,
        "multistep": MultiStepLR,
        "exponential": ExponentialLR,
        "cosine": CosineAnnealingLR,
        "cosine_warm_restarts": CosineAnnealingWarmRestarts,
        "plateau": ReduceLROnPlateau,
    }

    name = cfg.scheduler.name.lower()
    
    if name not in schedulers:
        raise ValueError(
            f"Unknown scheduler: {name}. "
            f"Available schedulers: {list(schedulers)}"
        )

    scheduler_cls = schedulers[name]

    return scheduler_cls(
        optimizer,
        **cfg.scheduler.params,
    )

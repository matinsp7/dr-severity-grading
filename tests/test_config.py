from src.utils.config import load_config

cfg = load_config("configs/exp005_proposed.yaml")

print(cfg.experiment_name)

print(cfg.optimizer.lr)

print(cfg.trainer.epochs)

print(cfg.dataset.image_size)
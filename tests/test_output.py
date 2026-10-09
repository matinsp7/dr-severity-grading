from src.utils.paths import ExperimentPaths

paths = ExperimentPaths("exp001_baseline")

print(paths.root)
print(paths.best_model)
print(paths.metrics_csv)
print(paths.confusion_matrix)
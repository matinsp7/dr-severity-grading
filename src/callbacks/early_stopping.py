class EarlyStopping:

    def __init__(self, patience):
        self.patience = patience
        self.counter = 0

    def step(self, improved):
        if improved:
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience
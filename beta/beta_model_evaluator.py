

class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    def evaluate_model(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)[1]
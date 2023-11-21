from sklearn.metrics import f1_score


class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    def evaluate_model(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)[1]

    def calculate_f1_score(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        y_pred = [1 if pred >= 0.5 else 0 for pred in y_pred]
        return f1_score(y_test, y_pred)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier()

    def train_model(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def evaluate_model(self, X_test, Y_test):
        X_test_prediction = self.model.predict(X_test)
        return accuracy_score(X_test_prediction, Y_test)

    def save_model(self, filename):
        joblib.dump(self.model, filename)

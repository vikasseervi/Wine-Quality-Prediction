from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from data_preprocessor import DataPreprocessor  # Import the DataPreprocessor module
from data_loader import DataLoader  # Import the DataLoader module

class ModelTrainer:
    def __init__(self, data_file_path):
        self.model = RandomForestClassifier()
        self.data_loader = DataLoader(data_file_path)
        self.data = self.data_loader.load_data()
        self.preprocessor = DataPreprocessor(self.data)
        self.X, self.Y = self.preprocessor.separate_features_labels()
        self.X_train, self.X_test, self.Y_train, self.Y_test = self.preprocessor.split_data()

    def train_model(self):
        # Train the model on the training data
        self.model.fit(self.X_train, self.Y_train)

    def evaluate_model(self):
        # Evaluate the model on the test set
        X_test_prediction = self.model.predict(self.X_test)
        return accuracy_score(X_test_prediction, self.Y_test)

    def save_model(self, filename):
        # Save the trained model to the file
        joblib.dump(self.model, filename)
        print(f'Model saved to {filename}')

data_file_path = 'winequality.csv'
trainer = ModelTrainer(data_file_path)

trainer.train_model()
accuracy = trainer.evaluate_model()
print(f'Accuracy on test data: {accuracy * 100:.2f}%')

trainer.save_model('wine_quality_model.pkl')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.wine_dataset = None

    def load_data(self):
        self.wine_dataset = pd.read_csv(self.file_path)
        return self.wine_dataset

    def display_info(self):
        print(self.wine_dataset.shape)
        print(self.wine_dataset.head())
        print(self.wine_dataset.describe())

    def check_missing_values(self):
        return self.wine_dataset.isnull().sum()


class DataVisualizer:
    @staticmethod
    def plot_quality_distribution(data):
        sns.catplot(x='quality', data=data, kind='count')
        plt.show()

    @staticmethod
    def plot_feature_vs_quality(data, feature):
        plt.figure(figsize=(5, 5))
        sns.barplot(x='quality', y=feature, data=data, ci=None)
        plt.show()

    @staticmethod
    def plot_correlation_heatmap(data):
        correlation = data.corr()
        plt.figure(figsize=(10, 10))
        sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')
        plt.show()


class DataPreprocessor:
    def __init__(self, data):
        self.data = data
        self.X = None
        self.Y = None
        self.feature_names = None

    def separate_features_labels(self):
        self.X = self.data.drop('quality', axis=1)
        self.feature_names = self.X.columns  # Save feature names
        self.Y = self.data['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)
        return self.X, self.Y

    def split_data(self, test_size=0.2, random_state=3):
        return train_test_split(self.X, self.Y, test_size=test_size, random_state=random_state)


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


class Predictor:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def predict_quality(self, input_data):
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        input_df = pd.DataFrame(input_data_reshaped, columns=self.feature_names)  # Use feature names
        prediction = self.model.predict(input_df)
        return 'Good Quality Wine' if prediction[0] == 1 else 'Bad Quality Wine'


# Usage
data_loader = DataLoader('winequality.csv')
wine_dataset = data_loader.load_data()
data_loader.display_info()
missing_values = data_loader.check_missing_values()
print(missing_values)

visualizer = DataVisualizer()
visualizer.plot_quality_distribution(wine_dataset)
visualizer.plot_feature_vs_quality(wine_dataset, 'volatile acidity')
visualizer.plot_feature_vs_quality(wine_dataset, 'citric acid')
visualizer.plot_feature_vs_quality(wine_dataset, 'alcohol')
visualizer.plot_correlation_heatmap(wine_dataset)

preprocessor = DataPreprocessor(wine_dataset)
X, Y = preprocessor.separate_features_labels()
X_train, X_test, Y_train, Y_test = preprocessor.split_data()

trainer = ModelTrainer()
trainer.train_model(X_train, Y_train)
accuracy = trainer.evaluate_model(X_test, Y_test)
print('Accuracy:', accuracy)

trainer.save_model('model.pkl')

predictor = Predictor(trainer.model, preprocessor.feature_names)
input_data = (7.5, 0.5, 0.36, 6.1, 0.071, 17.0, 102.0, 0.9978, 3.35, 0.8, 10.5)
quality_prediction = predictor.predict_quality(input_data)
print(quality_prediction)

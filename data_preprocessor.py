from sklearn.model_selection import train_test_split

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

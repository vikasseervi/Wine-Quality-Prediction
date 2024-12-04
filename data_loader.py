import pandas as pd

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

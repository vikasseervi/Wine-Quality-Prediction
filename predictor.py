import numpy as np
import pandas as pd

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

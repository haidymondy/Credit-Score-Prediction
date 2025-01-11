import pickle
import pandas as pd
from flask import Flask, request, jsonify

from Experminral.test_data_preprocessing import DataPreprocessor, DataEncoder

# Create instances of the classes
preprocessor1 = DataPreprocessor()
preprocessor2 = DataEncoder()

# Load the test data
test_data = pd.read_csv(r'C:\Users\Hazem\Desktop\awt\Data\test.csv')

# Apply the first preprocessing step
test_data = preprocessor1.preprocess_data(test_data)

# Apply the second preprocessing step (data encoding)
data = preprocessor2.preprocess(test_data)

# Initialize Flask app
app = Flask(__name__)

# Load the model
with open(r'C:\Users\Hazem\Desktop\awt\rf2_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the prediction route
@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Make prediction
        prediction = model.predict(data)

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        # Handle any errors and return the error message
        return jsonify({'error': str(e)}), 400

# Start the Flask app
if __name__ == '__main__':
    app.run(port=5000)

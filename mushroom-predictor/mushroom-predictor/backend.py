from flask import Flask, request, jsonify, render_template
import pandas as pd
from pycaret.classification import load_model, predict_model

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = load_model('mushroom_classification_model')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json(force=True)
        
        # Convert JSON data into a DataFrame
        data_unseen = pd.DataFrame([data])
        
        # Generate prediction using the loaded model
        prediction = predict_model(model, data=data_unseen)
        
        # Extract the predicted label from the prediction DataFrame
        output = prediction['prediction_label'].iloc[0]
        
        # Return the prediction as a JSON response
        return jsonify({'prediction': output})
    
    except Exception as e:
        # Handle errors and return a message
        return jsonify({"error": str(e)})

# Define a route for the frontend
@app.route('/')
def home():
    return render_template('frontend.html')

if __name__ == '__main__':
    app.run(debug=True)

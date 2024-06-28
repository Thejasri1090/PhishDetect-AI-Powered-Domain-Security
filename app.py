import numpy as np
from flask import Flask, request, render_template
from data_loading import load_data
from feature_extraction import extract_features
from model_training import train_models
from model_evaluation import evaluate_models
from save_load_model import save_model, load_model
from sklearn.model_selection import train_test_split
# Flask application for deployment
app = Flask(__name__)

# Load best model for prediction
model = joblib.load(model_path)

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    features = np.array(extract_features(url)).reshape(1, -1)
    
    # Predict using the best model
    prediction = model.predict(features)[0]
    
    result = 'Phishing URL' if prediction == 1 else 'Legitimate URL'
    return render_template('result.html', url=url, result=result)
    # Run Flask app
if __name__ == '__main__':
    app.run(debug=True)


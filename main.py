#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:49:22 2024

@author: thejasri
"""

import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from flask import Flask, request, render_template
import joblib

# Function to extract features from URL
def extract_features(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path
    
    # Length-based features
    url_length = len(url)
    domain_length = len(domain)
    path_length = len(path)
    num_path_segments = len(path.split('/'))
    num_queries = len(parsed_url.query)
    
    # Count-based features
    num_dots = domain.count('.')
    num_hyphens = domain.count('-')
    num_digits = sum(c.isdigit() for c in domain)
    num_special_chars = len(re.findall(r'[^a-zA-Z0-9]', domain))
    num_subdomains = len(domain.split('.'))
    
    # Binary features
    has_https = 1 if parsed_url.scheme == 'https' else 0
    has_dot_com = 1 if domain.endswith('.com') else 0
    
    # Combine all features into a list
    features = [
        url_length, domain_length, path_length, num_path_segments, num_queries,
        num_dots, num_hyphens, num_digits, num_special_chars, num_subdomains,
        has_https, has_dot_com
    ]
    
    return features

# Load dataset
dataset_path = '/Users/thejasri/Downloads/urldata.csv'  # Replace with your actual path
df = pd.read_csv(dataset_path)

# Display dataset before feature extraction
print("Dataset Before Feature Extraction:")
print(df.head())

# Feature extraction
df['features'] = df['url'].apply(extract_features)
X = np.array(df['features'].tolist())
y = df['label']

# Display dataset after feature extraction
df_after_extraction = pd.DataFrame(X, columns=[
    'url_length', 'domain_length', 'path_length', 'num_path_segments', 'num_queries',
    'num_dots', 'num_hyphens', 'num_digits', 'num_special_chars', 'num_subdomains',
    'has_https', 'has_dot_com'
])
df_after_extraction['label'] = y
print("\nDataset After Feature Extraction:")
print(df_after_extraction.head())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train models
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train Random Forest classifier
rf_classifier.fit(X_train, y_train)

# Train Decision Tree classifier
dt_classifier.fit(X_train, y_train)

# Evaluate models
models = {
    'Random Forest Classifier': rf_classifier,
    'Decision Tree Classifier': dt_classifier
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'\n{name} Accuracy: {accuracy}')
    print(classification_report(y_test, y_pred))

# Select the best model based on accuracy
best_model_name = max(models, key=lambda k: accuracy_score(y_test, models[k].predict(X_test)))
best_model = models[best_model_name]
print(f'\nBest Model: {best_model_name}')

# Save best model
model_path = f'phishing_detection_{best_model_name.lower().replace(" ", "_")}_model.pkl'
joblib.dump(best_model, model_path)

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

# HTML templates
index_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PhishDetect - Phishing Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            padding: 20px;
        }
        h1 {
            text-align: center;
            margin-bottom: 20px;
        }
        form {
            max-width: 500px;
            margin: auto;
            background: #f9f9f9;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        label, input {
            display: block;
            margin-bottom: 10px;
        }
        input[type="submit"] {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin-top: 10px;
            cursor: pointer;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <h1>PhishDetect - Phishing Detection</h1>
    <form action="/predict" method="post">
        <label for="url">Enter URL:</label>
        <input type="text" id="url" name="url" required>
        <input type="submit" value="Check URL">
    </form>
</body>
</html>
'''

result_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PhishDetect - Phishing Detection Result</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            padding: 20px;
        }
        h1 {
            text-align: center;
            margin-bottom: 20px;
        }
        .result {
            max-width: 500px;
            margin: auto;
            background: #f9f9f9;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
    </style>
</head>
<body>
    <h1>PhishDetect - Phishing Detection Result</h1>
    <div class="result">
        <p>URL: {{ url }}</p>
        <p><strong>Result:</strong> {{ result }}</p>
    </div>
</body>
</html>
'''

# Save HTML templates to files
with open('templates/index.html', 'w') as f:
    f.write(index_html)

with open('templates/result.html', 'w') as f:
    f.write(result_html)

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)


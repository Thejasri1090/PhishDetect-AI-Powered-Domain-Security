from sklearn.metrics import classification_report, accuracy_score
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

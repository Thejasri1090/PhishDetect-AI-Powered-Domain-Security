import joblib
# Save best model
model_path = f'phishing_detection_{best_model_name.lower().replace(" ", "_")}_model.pkl'
joblib.dump(best_model, model_path)

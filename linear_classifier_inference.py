import pandas as pd
from joblib import load

# Load the model from file
model_filename = 'logistic_regression_model.joblib'
classifier = load(model_filename)
print("Model loaded successfully.")

# Load features from CSV
#features_filename = 'C:\\Users\\DELL\\Test_data_for_linear_classifier_features.csv'  # Replace with your actual file name
features_df = pd.read_csv('C:\\Users\\DELL\\abc.csv')
#features_df = pd.read_csv('abc.csv')

# Assuming the features do not include the target variable
new_observations = features_df.values

# Use the loaded model to predict the class of new observations
predictions = classifier.predict(new_observations)

# Add predictions as a new column to the DataFrame
features_df['Predictions'] = predictions

# Save the DataFrame with predictions to a new CSV file
output_filename = 'predictions.csv'  # Replace with your desired output file name
features_df.to_csv(output_filename, index=False)
print(f"Predictions saved to {output_filename}")

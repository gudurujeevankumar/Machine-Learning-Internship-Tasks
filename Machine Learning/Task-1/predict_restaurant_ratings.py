import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Load the dataset
# This function checks if the dataset file exists and loads it into a pandas DataFrame
# Raises an error if the file is not found
def load_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")
    return pd.read_csv(file_path)

# Preprocess the dataset
# Handles missing values and encodes categorical variables for machine learning
def preprocess_data(data):
    if data is None:
        print("Error: No data to preprocess.")
        return None, None, None

    # Handle missing values
    data = data.fillna(data.mean(numeric_only=True))
    data = data.fillna(data.mode().iloc[0])

    # Separate features and target
    X = data.drop(columns=['Aggregate rating'], errors='ignore')
    if 'Aggregate rating' not in data.columns:
        print("Error: 'Aggregate rating' column not found in dataset.")
        return None, None, None

    y = data['Aggregate rating']

    # Identify categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
        ], remainder='passthrough'
    )

    print("Data preprocessing completed.")
    return X, y, preprocessor

# Split the data into training and testing sets
# Splits the dataset into features (X) and target variable (y)
def split_data(X, y):
    if X is None or y is None:
        print("Error: No data to split.")
        return None, None, None, None
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
# Fits a regression model (e.g., Linear Regression) to the training data
def train_model(X_train, y_train, preprocessor):
    if X_train is None or y_train is None:
        print("Error: No training data available.")
        return None

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    model.fit(X_train, y_train)
    print("Model training completed.")
    return model

# Evaluate the model
# Calculates metrics like Mean Squared Error (MSE) and R-squared to measure performance
def evaluate_model(model, X_test, y_test):
    if model is None or X_test is None or y_test is None:
        print("Error: Model or test data is missing.")
        return None, None

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    return mse, r2

# Analyze feature importance
# Determines which features have the most influence on the predicted ratings
def analyze_features(model, preprocessor):
    if model is None or preprocessor is None:
        print("Error: Model or preprocessor is missing.")
        return None

    feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out()
    coefficients = model.named_steps['regressor'].coef_
    feature_importance = dict(zip(feature_names, coefficients))
    print("Feature Importance:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"{feature}: {importance}")
    return feature_importance

# Main script execution
if __name__ == "__main__":
    # File path to the dataset
    file_path = '/Users/jeevankumar/Desktop/Machine Learning/Dataset/Dataset.csv'
    DATASET_PATH = '/Users/jeevankumar/Desktop/Machine Learning/Dataset/Dataset.csv'

    try:
        # Load the dataset
        data = load_dataset(DATASET_PATH)
    except FileNotFoundError as e:
        # Handle missing dataset file
        print(e)
        print("Error: No data to preprocess.")
        print("Error: No data to split.")
        print("Error: No training data available.")
        print("Error: Model or test data is missing.")
        print("Error: Model or preprocessor is missing.")
        exit(1)

    # Load and preprocess the data
    data = load_dataset(file_path)  # Correct function name
    X, y, preprocessor = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train the model
    model = train_model(X_train, y_train, preprocessor)

    # Evaluate the model
    mse, r2 = evaluate_model(model, X_test, y_test)

    # Analyze feature importance
    analyze_features(model, preprocessor)

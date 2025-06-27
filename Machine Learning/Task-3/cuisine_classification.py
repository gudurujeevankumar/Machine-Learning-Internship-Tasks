# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, classification_report, roc_auc_score
)
from imblearn.over_sampling import SMOTE
import os

# Define constants for configuration
DATASET_PATH = '/Users/jeevankumar/Desktop/Machine Learning/Dataset/Dataset.csv'  # Path to the dataset
RARE_CLASS_THRESHOLD = 20  # Threshold for grouping rare classes
SUBSET_FRACTION = 0.5  # Fraction of the dataset to use for testing
MIN_SAMPLES_THRESHOLD = 5  # Minimum samples required for a class
SMOTE_K_NEIGHBORS = 2  # Number of neighbors for SMOTE oversampling
CV_SPLITS = 3  # Number of splits for cross-validation
RANDOM_STATE = 42  # Random state for reproducibility

# Load the dataset
def load_dataset(path):
    """
    Load the dataset from the specified path.

    Args:
        path (str): Path to the dataset file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at path: {path}")
    return pd.read_csv(path)

# Preprocess the dataset
def preprocess_data(data):
    """
    Preprocess the dataset by handling missing values and encoding categorical variables.

    Args:
        data (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    data.ffill(inplace=True)  # Forward fill missing values
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])
    return data

# Handle class imbalance
def handle_class_imbalance(y, threshold):
    """
    Group rare classes into a single category based on the specified threshold.

    Args:
        y (pd.Series): Target column.
        threshold (int): Minimum number of samples for a class to be considered non-rare.

    Returns:
        pd.Series: Updated target column.
    """
    return y.apply(lambda x: 'Rare' if y.value_counts()[x] < threshold else x)

# Filter classes with too few samples
def filter_classes(X, y, threshold):
    """
    Remove classes with fewer samples than the specified threshold.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target column.
        threshold (int): Minimum number of samples required for a class.

    Returns:
        tuple: Filtered feature matrix and target column.
    """
    y_filtered = y[y.map(y.value_counts()) >= threshold]
    X_filtered = X.loc[y_filtered.index]
    return X_filtered, y_filtered

# Perform SMOTE oversampling
def perform_smote(X, y, k_neighbors, random_state):
    """
    Apply SMOTE to oversample the minority classes.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target column.
        k_neighbors (int): Number of neighbors for SMOTE.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: Resampled feature matrix and target column.
    """
    smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    return smote.fit_resample(X, y)

# Train and evaluate models
def train_and_evaluate(models, hyperparameters, X_train, y_train, X_test, y_test, cv_splits, random_state):
    """
    Train and evaluate models using cross-validation and hyperparameter tuning.

    Args:
        models (dict): Dictionary of model names and instances.
        hyperparameters (dict): Dictionary of hyperparameters for each model.
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target column.
        X_test (pd.DataFrame): Testing feature matrix.
        y_test (pd.Series): Testing target column.
        cv_splits (int): Number of splits for cross-validation.
        random_state (int): Random state for reproducibility.

    Returns:
        sklearn.base.BaseEstimator: Best model based on evaluation.
    """
    best_model = None
    best_score = 0
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        grid_search = GridSearchCV(model, hyperparameters[model_name], cv=cv, scoring='roc_auc')
        grid_search.fit(X_train, y_train)

        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best score for {model_name}: {grid_search.best_score_}")

        if grid_search.best_score_ > best_score:
            best_model = grid_search.best_estimator_
            best_score = grid_search.best_score_

    y_pred = best_model.predict(X_test)
    roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test), multi_class='ovr')

    print("Best Model Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("ROC AUC:", roc_auc)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return best_model

# Main script
if __name__ == "__main__":
    # Load and preprocess the dataset
    data = load_dataset(DATASET_PATH)
    print("Dataset columns:", data.columns)
    data = preprocess_data(data)

    # Split the data into training and testing sets
    if 'Cuisines' not in data.columns:
        raise KeyError("'Cuisines' column not found in the dataset")
    X = data.drop('Cuisines', axis=1)
    y = data['Cuisines']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # Debugging: Analyze target column distribution
    print("Target column distribution:")
    print(y.value_counts())

    # Handle class imbalance by grouping rare classes
    y_train = handle_class_imbalance(y_train, RARE_CLASS_THRESHOLD)
    print("Updated target column distribution:")
    print(y_train.value_counts())

    # Filter classes with too few samples
    X_train, y_train = filter_classes(X_train, y_train, MIN_SAMPLES_THRESHOLD)

    # Perform SMOTE oversampling
    X_train, y_train = perform_smote(X_train, y_train, SMOTE_K_NEIGHBORS, RANDOM_STATE)
    print("Resampled target column distribution:")
    print(pd.Series(y_train).value_counts())

    # Define models and hyperparameters
    models = {
        'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE),
    }
    hyperparameters = {
        'RandomForest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    }

    # Train and evaluate models
    best_model = train_and_evaluate(models, hyperparameters, X_train, y_train, X_test, y_test, CV_SPLITS, RANDOM_STATE)

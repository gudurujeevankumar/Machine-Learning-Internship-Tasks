import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
from sklearn.decomposition import TruncatedSVD
import random

# Load the dataset from the specified file path
def load_dataset(file_path):
    return pd.read_csv(file_path)

# Preprocess the dataset to handle missing values, encode categorical variables, and normalize numerical features
def preprocess_data(df):
    # Handle missing values by replacing them with 0
    df.fillna(0, inplace=True)

    # Process the 'Cuisines' column by splitting and creating binary columns for each cuisine
    df['Cuisines'] = df['Cuisines'].fillna('').str.split(', ').apply(lambda x: ','.join(sorted(x)) if isinstance(x, list) else '')
    unique_cuisines = set(','.join(df['Cuisines']).split(','))
    cuisine_columns = pd.DataFrame({f'Cuisine_{cuisine}': df['Cuisines'].apply(lambda x: 1 if cuisine in x.split(',') else 0) for cuisine in unique_cuisines})
    df = pd.concat([df, cuisine_columns], axis=1)
    df.drop(columns=['Cuisines'], inplace=True)

    # Convert categorical columns to strings for consistent processing
    for column in df.select_dtypes(include=['object', 'int', 'float']).columns:
        df[column] = df[column].astype(str)

    # Encode categorical variables using LabelEncoder
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        label_encoders[column].fit(df[column])
        df[column] = label_encoders[column].transform(df[column])

    # Normalize numerical features using MinMaxScaler
    scaler = MinMaxScaler()
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    return df, label_encoders, scaler, unique_cuisines

# Define user preferences for restaurant recommendations
def get_user_preferences():
    return {
        'Cuisines': 'Italian',
        'Price range': '3',  # Example preference for medium price range
    }

# Recommend restaurants based on user preferences using content-based filtering
def recommend_restaurants(df, user_preferences, label_encoders, unique_cuisines):
    # Convert user preferences to encoded values
    encoded_preferences = {}
    for key, value in user_preferences.items():
        if key.startswith('Cuisine_'):
            encoded_preferences[key] = 1 if value in key else 0
        elif key in label_encoders:
            try:
                if value in label_encoders[key].classes_:
                    encoded_preferences[key] = label_encoders[key].transform([value])[0]
                else:
                    encoded_preferences[key] = 0
            except ValueError:
                encoded_preferences[key] = 0
        else:
            encoded_preferences[key] = 0

    # Create a DataFrame for user preferences and align it with dataset columns
    user_vector = pd.DataFrame([encoded_preferences])
    missing_columns = set(df.columns) - set(user_vector.columns)
    missing_data = pd.DataFrame({col: [0] for col in missing_columns})
    user_vector = pd.concat([user_vector, missing_data], axis=1)
    user_vector = user_vector[df.columns]

    # Ensure no NaN values remain in the DataFrame
    df = df.fillna(0)
    user_vector = user_vector.fillna(0)

    # Select relevant features for similarity calculation
    relevant_features = ['Price range', 'Aggregate rating', 'Votes'] + [f'Cuisine_{cuisine}' for cuisine in unique_cuisines]
    df_relevant = df[relevant_features].copy()
    user_vector_relevant = user_vector[relevant_features].copy()

    # Apply weights to features based on user preferences
    weights = {key: 2 for key in relevant_features}
    weights.update({f'Cuisine_{cuisine}': 1 for cuisine in unique_cuisines})
    for column in df_relevant.columns:
        df_relevant[column] *= weights.get(column, 1)
        user_vector_relevant[column] *= weights.get(column, 1)

    # Normalize the weighted features using MinMaxScaler
    scaler = MinMaxScaler()
    df_relevant[relevant_features] = scaler.fit_transform(df_relevant[relevant_features])
    user_vector_relevant[relevant_features] = scaler.transform(user_vector_relevant[relevant_features])

    # Calculate similarity using cosine similarity
    similarity_scores = cosine_similarity(df_relevant, user_vector_relevant)

    # Add similarity scores to the DataFrame
    df = pd.concat([df, pd.DataFrame({'similarity': similarity_scores.flatten()})], axis=1)

    # Return top recommendations sorted by similarity
    return df.sort_values(by='similarity', ascending=False).head(5)

# Perform collaborative filtering using Singular Value Decomposition (SVD)
def collaborative_filtering(df, user_id):
    # Create a user-item matrix for collaborative filtering
    user_item_matrix = df.pivot_table(index='User ID', columns='Restaurant ID', values='Rating', fill_value=0)

    # Dynamically adjust the number of SVD components based on matrix dimensions
    n_components = min(50, user_item_matrix.shape[1])
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    latent_matrix = svd.fit_transform(user_item_matrix)

    # Calculate similarity scores for the given user
    user_vector = latent_matrix[user_id]
    similarity_scores = np.dot(latent_matrix, user_vector)

    # Add collaborative similarity scores to the DataFrame
    df['collaborative_similarity'] = similarity_scores

    return df

# Combine content-based and collaborative filtering for hybrid recommendations
def hybrid_recommendation(df, user_preferences, label_encoders, unique_cuisines, user_id):
    # Perform content-based filtering
    df = recommend_restaurants(df, user_preferences, label_encoders, unique_cuisines)

    # Perform collaborative filtering
    df = collaborative_filtering(df, user_id)

    # Combine scores from both methods with weighted contributions
    content_weight = 0.6
    collaborative_weight = 0.4
    df['hybrid_score'] = content_weight * df['similarity'] + collaborative_weight * df['collaborative_similarity']

    # Return top recommendations sorted by hybrid score
    return df.sort_values(by='hybrid_score', ascending=False).head(5)

# Add synthetic ratings and user IDs for demonstration purposes
def add_synthetic_ratings(df):
    synthetic_data = pd.DataFrame({
        'Rating': [random.randint(1, 5) for _ in range(len(df))],
        'User ID': [random.randint(0, 100) for _ in range(len(df))]
    })
    return pd.concat([df, synthetic_data], axis=1)

# Main function to execute the recommendation system
def main():
    file_path = '/Users/jeevankumar/Desktop/Machine Learning/Dataset/Dataset.csv'
    df = load_dataset(file_path)
    df, label_encoders, scaler, unique_cuisines = preprocess_data(df)

    # Add synthetic ratings for collaborative filtering
    df = add_synthetic_ratings(df)

    # Define user preferences and user ID for recommendations
    user_preferences = get_user_preferences()
    user_id = 0  # Example user ID

    # Generate hybrid recommendations
    recommendations = hybrid_recommendation(df, user_preferences, label_encoders, unique_cuisines, user_id)

    # Display top recommendations
    print("Top Recommendations:")
    print(recommendations)

# Execute the main function when the script is run directly
if __name__ == "__main__":
    main()

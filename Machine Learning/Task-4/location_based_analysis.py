import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium

# Load the dataset
dataset_path = '/Users/jeevankumar/Desktop/Machine Learning/Dataset/Dataset.csv'
data = pd.read_csv(dataset_path)

# Ensure necessary columns exist
required_columns = ['Latitude', 'Longitude', 'City', 'Locality', 'Aggregate rating', 'Cuisines', 'Price range']
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"Dataset must contain the following columns: {required_columns}")

# Find the best restaurant based on the highest rating
best_restaurant = data.loc[data['Aggregate rating'].idxmax()]
map_center = [best_restaurant['Latitude'], best_restaurant['Longitude']]
restaurant_map = folium.Map(location=map_center, zoom_start=12)

for _, row in data.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"{row['Locality']}\nRating: {row['Aggregate rating']}\nCuisine: {row['Cuisines']}\nPrice Range: {row['Price range']}"
    ).add_to(restaurant_map)

restaurant_map.save('./Task-4/restaurant_distribution_map.html')
print("Map saved as restaurant_distribution_map.html")

# Step 2: Group by city/locality and analyze concentration
city_group = data.groupby('City')
locality_group = data.groupby('Locality')

city_stats = city_group.agg({
    'Aggregate rating': 'mean',
    'Cuisines': lambda x: x.mode()[0] if not x.mode().empty else None,
    'Price range': 'mean'
})

locality_stats = locality_group.agg({
    'Aggregate rating': 'mean',
    'Cuisines': lambda x: x.mode()[0] if not x.mode().empty else None,
    'Price range': 'mean'
})

print("City-level statistics:")
print(city_stats)

print("Locality-level statistics:")
print(locality_stats)

# Step 3: Visualize insights
plt.figure(figsize=(10, 6))
sns.histplot(data['City'], kde=False, color='blue')
plt.title('Distribution of Restaurants by City')
plt.xlabel('City')
plt.ylabel('Number of Restaurants')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./Task-4/city_distribution.png')
print("City distribution plot saved as city_distribution.png")

plt.figure(figsize=(10, 6))
sns.histplot(data['Locality'], kde=False, color='green')
plt.title('Distribution of Restaurants by Locality')
plt.xlabel('Locality')
plt.ylabel('Number of Restaurants')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./Task-4/locality_distribution.png')
print("Locality distribution plot saved as locality_distribution.png")

# Step 4: Identify insights
print("Interesting insights:")
print("- Cities with the highest average ratings:")
print(city_stats.sort_values(by='Aggregate rating', ascending=False).head())
print("- Localities with the highest average ratings:")
print(locality_stats.sort_values(by='Aggregate rating', ascending=False).head())

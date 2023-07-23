import pandas as pd
import matplotlib.pyplot as plt

# Loading the data
df = pd.read_csv('ratings_Electronics (1).csv', names=['userId', 'productId','Rating','timestamp'])

# print(df.head())

# Dropping the timestamp column
df.drop(['timestamp'], axis=1, inplace=True)

# Dropping null values
df.dropna(inplace=True)

# print(df.head())

# Number of unique users in the dataset
# print('Number of unique users in the dataset = ', df['userId'].nunique()) #4201696

# Number of unique products in the dataset
# print('Number of unique products in the dataset = ', df['productId'].nunique()) #476002

# The recommending weight formula (collobrative filtering) is: W = (R*v)/(R+M)
# W = Weight
# R = Number of ratings
# v = Average rating
# M = Minimum ratings required

# Calculating the average rating for all the products
average_rating = df.groupby('productId')['Rating'].mean().reset_index()
# print(average_rating.head())

# Calculating the number of ratings for all the products
count_rating = df.groupby('productId')['Rating'].count().reset_index()
# print(count_rating.head())

# Setting minimum number of ratings required
min_rating = 100

# Merging the average rating and count rating dataframes
average_rating = pd.merge(average_rating, count_rating, on='productId')
average_rating.columns = ['productId', 'average_rating', 'count_rating']
# print(average_rating.head())

# Filtering the products with minimum number of ratings
average_rating = average_rating[average_rating['count_rating'] >= min_rating]
# print(average_rating.head())

# Calculating the recommending weight
average_rating['weight'] = (average_rating['count_rating'] * average_rating['average_rating']) / (average_rating['count_rating'] + min_rating)
# print(average_rating.head())

# Sorting the products based on the recommending weight
average_rating = average_rating.sort_values(by='weight', ascending=False)
# print(average_rating.head())

# Plotting the top 20 products and their recommending weight
plt.figure(figsize=(10, 6))
plt.barh(average_rating['productId'].head(20), average_rating['weight'].head(20), align='center', alpha=0.8)
plt.xlabel('Weight')
plt.ylabel('Products')
plt.title('Top 10 products and their recommending weight')
plt.show()
import pandas as pd
import numpy as np


# Loading the data
df = pd.read_csv('ratings_Electronics (1).csv', names=['userId', 'productId','Rating','timestamp'])

# dropping the timestamp column
df.drop(['timestamp'], axis=1, inplace=True)

# Filtering the ratings greater than or equal to 0
df = df[df['Rating'] >= 0]

# dropping null values
df.dropna(inplace=True)

# Get the number of rows and columns
num_rows = df.shape[0]
num_columns = df.shape[1]

# Cast the variables to 64-bit integer data type
num_rows = np.int64(num_rows)
num_columns = np.int64(num_columns)

# Calculate the number of cells
num_cells = num_rows * num_columns

print('Number of Rows: {}'.format(num_rows))
print('Number of Columns: {}'.format(num_columns))
print('Number of Cells: {}'.format(num_cells))


# Limiting the data (to run on local machine)
df = df.iloc[:100000, :]


# filltering
df = df[df['Rating'] >= 0]

# print(df.dtypes)

# Encryting the productId and userId
df['productId'] = df['productId'].astype('category')
df['productId'] = df['productId'].cat.codes

df['userId'] = df['userId'].astype('category')
df['userId'] = df['userId'].cat.codes

print(df.dtypes)

# Creating pivot table
product_rating_pivot = df.pivot(index='userId', columns='productId', values='Rating').fillna(0)
print(product_rating_pivot.head())

# KNN
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

product_rating_matrix = csr_matrix(product_rating_pivot.values)

knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(product_rating_matrix)

# Testing the model
query_index = np.random.choice(product_rating_pivot.shape[0])
distances, indices = knn_model.kneighbors(product_rating_pivot.iloc[query_index, :].values.reshape(1, -1), n_neighbors=6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(product_rating_pivot.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, product_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]))
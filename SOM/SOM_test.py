# Self Organizing Map

# Importing the libraries
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets

from SelfOrganizingMap import SOM
import pandas as pd

# Importing the dataset

file = input("Δώσε input file(a,b,c,d): ")
input_file = 'data_package_%s.csv' % file
df = pd.read_csv(input_file, header=0)
df = df._get_numeric_data()
# targets
targets_file = 'data_package_values_%s.csv' % file
targets_df = pd.read_csv(targets_file, header=0)
targets_df = targets_df._get_numeric_data()

X = df.values
y = targets_df.values


# Load iris data
# iris = datasets.load_iris()
# iris_data = iris.data
# iris_label = iris.target

# # Extract just two features (just for ease of visualization)
# iris_data = iris_data[:, :2]

# Build a 3x1 SOM (3 clusters)
som = SOM(m=4, n=2, dim=2)

# Fit it to the data
som.fit(X, y)
# Assign each datapoint to its predicted cluster
predictions = som.predict(X)

# Plot the results
fig, ax = plt.subplots(nrows=1, ncols=2, )
# x = iris_data[:,0]
# y = iris_data[:,1]
colors = ['red',  'blue']
print(predictions)
ax[0].scatter(range(len(X)), y, c='b', marker='o')
ax[0].title.set_text('Actual Classes')
ax[0].scatter(range(len(X)), predictions, c='r', marker='.')
ax[0].title.set_text('SOM Predictions')
ax[1].scatter(X[:, 0], X[:, 1], predictions,
              c=predictions, cmap=ListedColormap(colors))
ax[1].title.set_text('SOM Predictions')
plt.show()

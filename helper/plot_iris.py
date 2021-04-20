import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# download and convert the csv into a DataFrame
df = pd.read_csv(url, header=None)
df.head()

# extract the label column
y = df.iloc[:, 4].values
# extract features
x = df.iloc[:, 0:3].values

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.set_title('Iris data set')
ax.set_xlabel("Sepal length in width (cm)")
ax.set_ylabel("Sepal width in width (cm)")
ax.set_zlabel("Petal length in width (cm)")

# plot the samples
ax.scatter(x[:50, 0], x[:50, 1], x[:50, 2], color='red', 
           marker='o', s=4, edgecolor='red', label="Iris Setosa")
ax.scatter(x[50:100, 0], x[50:100, 1], x[50:100, 2], color='blue', 
           marker='^', s=4, edgecolor='blue', label="Iris Versicolour")
ax.scatter(x[100:150, 0], x[100:150, 1], x[100:150, 2], color='green', 
           marker='x', s=4, edgecolor='green', label="Iris Virginica")

plt.legend(loc='upper left')
plt.show()
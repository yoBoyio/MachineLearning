import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

plt.rcParams['figure.figsize'] = (12.0, 9.0)


file = input("Δώσε input file(a,b,c,d): ")
input_file = 'data_package_%s.csv' % file
df = pd.read_csv(input_file, header=0)
df = df._get_numeric_data()
# targets
targets_file = 'data_package_values_%s.csv' % file
targets_df = pd.read_csv(targets_file, header=0)
targets_df = targets_df._get_numeric_data()

X = df.values
Y = targets_df.values

# plot_data(targets)
# values = add_biases(values)
# patterns_train, patterns_test, targets_train, targets_test = train_test_split(
#     values, targets, test_size=0.2, random_state=123)
X = df.iloc[:, 0]
Y = targets_df.iloc[:, 1]
plt.scatter(X, Y)
plt.show()

# Building the model
X_mean = np.mean(X)
Y_mean = np.mean(Y)

num = 0
den = 0
for i in range(len(X)):
    num += (X[i] - X_mean)*(Y[i] - Y_mean)
    den += (X[i] - X_mean)**2
m = num / den
c = Y_mean - m*X_mean

print(m, c)

Y_pred = m*X + c

plt.scatter(X, Y)  # actual
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)],
         color='red')  # predicted
plt.show()

import pandas as pd
import numpy as np
import random

# init i.a data
n = 200
nums = np.zeros(shape=(n, 2))
classes = np.zeros(shape=(n))


for i in range(n):
    key = (random.randint(0, 1))
    if (key == 1):
        nums[i] = [(random.uniform(0.7, 0.9)), (random.uniform(0.7, 0.9))]
    else:
        nums[i] = [(random.uniform(0.0, 0.3)), (random.uniform(0.0, 0.3))]
    classes[i] = key


data = {'X': nums[:, 0],
        'Y': nums[:, 1]}

data_values = {'values': classes}

df = pd.DataFrame(data, columns=['X', 'Y'])
df.to_csv('./data_package_1.csv', index=False)
df2 = pd.DataFrame(data_values, columns=['values'])
df2.to_csv('./data_package_values_1.csv', index=False)


# init i.b data

n = 200
nums = np.zeros(shape=(n, 2))
classes = np.zeros(shape=(n))

for i in range(n):
    key = (random.randint(0, 2))
    if (key == 1):
        nums[i] = [(random.uniform(0.4, 0.9)), (random.uniform(0.0, 0.9))]
        classes[i] = 1
    elif (key == 2):
        nums[i] = [(random.uniform(0.0, 0.3)), (random.uniform(0.4, 0.9))]
        classes[i] = 1
    else:
        nums[i] = [(random.uniform(0.0, 0.3)), (random.uniform(0.0, 0.3))]
        classes[i] = 0


data = {'X': nums[:, 0],
        'Y': nums[:, 1]}

data_values = {'values': classes}

df = pd.DataFrame(data, columns=['X', 'Y'])
df.to_csv('./data_package_2.csv', index=False)
df2 = pd.DataFrame(data_values, columns=['values'])
df2.to_csv('./data_package_values_2.csv', index=False)

# init i.c data

n = 200
nums = np.zeros(shape=(n, 2))
classes = np.zeros(shape=(n))


for i in range(n):
    key = (random.randint(0, 7))
    if (key == 0):
        nums[i] = [(random.uniform(0.0, 0.9)), (random.uniform(0.0, 0.3))]
        classes[i] = 1
    elif (key == 2):
        nums[i] = [(random.uniform(0.0, 0.9)), (random.uniform(0.7, 0.9))]
        classes[i] = 1
    elif (key == 3):
        nums[i] = [(random.uniform(0.0, 0.3)), (random.uniform(0.0, 0.9))]
        classes[i] = 1
    elif (key == 1):
        nums[i] = [(random.uniform(0.7, 0.9)), (random.uniform(0.0, 0.9))]
        classes[i] = 1
    else:
        nums[i] = [(random.uniform(0.4, 0.6)), (random.uniform(0.4, 0.6))]
        classes[i] = -1

data = {'X': nums[:, 0],
        'Y': nums[:, 1]}

data_values = {'values': classes}

df = pd.DataFrame(data, columns=['X', 'Y'])
df.to_csv('./data_package_3.csv', index=False)
df2 = pd.DataFrame(data_values, columns=['values'])
df2.to_csv('./data_package_values_3.csv', index=False)

# init i.d data

n = 200
nums = np.zeros(shape=(n, 2))
classes = np.zeros(shape=(n))

for i in range(n):
    key = (random.randint(0, 3))
    if (key == 0):
        nums[i] = [(random.uniform(0.0, 0.3)), (random.uniform(0.0, 0.3))]
        classes[i] = 1
    elif (key == 1):
        nums[i] = [(random.uniform(0.7, 0.9)), (random.uniform(0.0, 0.3))]
        classes[i] = -1
    elif (key == 2):
        nums[i] = [(random.uniform(0.0, 0.3)), (random.uniform(0.7, 0.9))]
        classes[i] = -1
    else:
        nums[i] = [(random.uniform(0.7, 0.9)), (random.uniform(0.7, 0.9))]
        classes[i] = 1

data = {'X': nums[:, 0],
        'Y': nums[:, 1]}

data_values = {'values': classes}

df = pd.DataFrame(data, columns=['X', 'Y'])
df.to_csv('./data_package_4.csv', index=False)
df2 = pd.DataFrame(data_values, columns=['values'])
df2.to_csv('./data_package_values_4.csv', index=False)


# init ii.a data
n = 200
nums = np.zeros(shape=(n, 3))
classes = np.zeros(shape=(n))


for i in range(n):
    key = (random.randint(0, 1))
    if (key == 0):
        nums[i] = [(random.uniform(0.0, 0.3)),
                   (random.uniform(0.0, 0.3)), (random.uniform(0.0, 0.3))]
        classes[i] = 0
    else:
        nums[i] = [(random.uniform(0.7, 0.9)),
                   (random.uniform(0.7, 0.9)), (random.uniform(0.7, 0.9))]
        classes[i] = 1

data = {'X': nums[:, 0],
        'Y': nums[:, 1],
        'Z': nums[:, 2]}

data_values = {'values': classes}

df = pd.DataFrame(data, columns=['X', 'Y', 'Z'])
df.to_csv('./data_package_2_1.csv', index=False)
df2 = pd.DataFrame(data_values, columns=['values'])
df2.to_csv('./data_package_values_2_1.csv', index=False)

# init i.b data

n = 200
nums = np.zeros(shape=(n, 3))
classes = np.zeros(shape=(n))


for i in range(n):
    key = (random.randint(0, 3))
    if (key == 0):
        nums[i] = [(random.uniform(0.0, 0.3)),
                   (random.uniform(0.0, 0.3)), (random.uniform(0.0, 0.3))]
        classes[i] = 0
    elif (key == 1):
        nums[i] = [(random.uniform(0.7, 0.9)),
                   (random.uniform(0.7, 0.9)), (random.uniform(0.7, 0.9))]
        classes[i] = 0
    elif (key == 2):
        nums[i] = [(random.uniform(0.7, 0.9)),
                   (random.uniform(0.7, 0.9)), (random.uniform(0.0, 0.3))]
        classes[i] = 1
    else:
        nums[i] = [(random.uniform(0.0, 0.3)),
                   (random.uniform(0.0, 0.3)), (random.uniform(0.7, 0.9))]
        classes[i] = 1

data = {'X': nums[:, 0],
        'Y': nums[:, 1],
        'Z': nums[:, 2]}

data_values = {'values': classes}

df = pd.DataFrame(data, columns=['X', 'Y', 'Z'])
df.to_csv('./data_package_2_2.csv', index=False)
df2 = pd.DataFrame(data_values, columns=['values'])
df2.to_csv('./data_package_values_2_2.csv', index=False)

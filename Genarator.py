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
df.to_csv('./data_package_a.csv', index=False)
df2 = pd.DataFrame(data_values, columns=['values'])
df2.to_csv('./data_package_values_a.csv', index=False)


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
df.to_csv('./data_package_b.csv', index=False)
df2 = pd.DataFrame(data_values, columns=['values'])
df2.to_csv('./data_package_values_b.csv', index=False)

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
        classes[i] = 0

data = {'X': nums[:, 0],
        'Y': nums[:, 1]}

data_values = {'values': classes}

df = pd.DataFrame(data, columns=['X', 'Y'])
df.to_csv('./data_package_c.csv', index=False)
df2 = pd.DataFrame(data_values, columns=['values'])
df2.to_csv('./data_package_values_c.csv', index=False)

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
        classes[i] = 0
    elif (key == 2):
        nums[i] = [(random.uniform(0.0, 0.3)), (random.uniform(0.7, 0.9))]
        classes[i] = 0
    else:
        nums[i] = [(random.uniform(0.7, 0.9)), (random.uniform(0.7, 0.9))]
        classes[i] = 1

data = {'X': nums[:, 0],
        'Y': nums[:, 1]}

data_values = {'values': classes}

df = pd.DataFrame(data, columns=['X', 'Y'])
df.to_csv('./data_package_d.csv', index=False)
df2 = pd.DataFrame(data_values, columns=['values'])
df2.to_csv('./data_package_values_d.csv', index=False)


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
df.to_csv('./data_package_ii_a.csv', index=False)
df2 = pd.DataFrame(data_values, columns=['values'])
df2.to_csv('./data_package_values_ii_a.csv', index=False)

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
df.to_csv('./data_package_ii_b.csv', index=False)
df2 = pd.DataFrame(data_values, columns=['values'])
df2.to_csv('./data_package_values_ii_b.csv', index=False)

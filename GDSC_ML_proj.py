# importing the libraries and modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor


# -----------------------------Importing the dataset and cleaning it---------------------------#

dataset = pd.read_csv("FinalData.csv")
dataset = dataset.dropna(how='any')
# Columns dropped because they were not impacting the value of price or one value was occurring too much.
dataset.drop(['Network Type', 'Full HD Recording', 'Audio Jack', 'Primary Camera', 'Resolution', 'Operating System', 'Domestic Warranty', 'Secondary Camera'], axis=1, inplace=True)
x = dataset.iloc[:, 6:].values
y = dataset.iloc[:, 2].values


# Converting data of the price column of the dataset into integer.
price = []
for i in range(len(y)):
    for j in y[i]:
        if j.isdigit():
            price.append(j)
            y[i] = int(''.join(price))
    price = []
y = y.reshape(-1,1)

# Converting the primary clock speed, RAM and storage into integers
pcs = x[:, 3]
for i in range(len(pcs)):
    lst = str(pcs[i]).split(' ')
    if lst[-1] == 'GHz':
        lst[-1] = lst[-1].replace('GHz', '*1000')
    elif lst[-1] == 'MHz':
        lst[-1] = lst[-1].replace('MHz', '')
    pcs[i] = ''.join(lst)
    pcs[i] = eval(pcs[i])


# Converting the display size of the dataset into integer value
display = x[:, 1]
for i in range(len(display)):
    str_dis = display[i].split(" ")[0]
    int_dis = float(str_dis)
    display[i] = int_dis


# Converting the values of the RAM and the storage into integer value
storage = x[:, 4]
for i in range(len(storage)):
    lst = str(storage[i]).split(' ')
    if lst[-1] == 'GB':
        lst[-1] = lst[-1].replace('GB', '*1024')
    elif lst[-1] == 'MB':
        lst[-1] = lst[-1].replace('MB', '')
    storage[i] = ''.join(lst)
    storage[i] = eval(storage[i])

ram = x[:, 5]
for i in range(len(ram)):
    lst = str(ram[i]).split(' ')
    if lst[-1] == 'GB':
        lst[-1] = lst[-1].replace('GB', '*1024')
    elif lst[-1] == 'MB':
        lst[-1] = lst[-1].replace('MB', '')
    ram[i] = ''.join(lst)
    ram[i] = eval(ram[i])


# Converting the battery capacity and the weight column of the dataset into integer
bat_cap = x[:, -7]
for i in range(len(bat_cap)):
    lst = str(bat_cap[i]).split(' ')
    if lst[-1] == 'mAh':
        lst[-1] = lst[-1].replace('mAh', '')
    bat_cap[i] = ''.join(lst)
    bat_cap[i] = int(bat_cap[i])


weight = x[:, -6]
for i in range(len(weight)):
    lst = str(weight[i]).split(' ')
    if lst[-1] == 'g':
        lst[-1] = lst[-1].replace('g', '')
    weight[i] = ''.join(lst)
    weight[i] = float(weight[i])


# Encoding of the categorical data in the matrix of features
df = pd.DataFrame(data=x)
df[[0, 2, 6, 7, 8, 9, 10]] = df[[0, 2, 6, 7, 8, 9, 10]].apply(LabelEncoder().fit_transform)
x = df.iloc[:, :].values

# Dividing the data into training and test dataset
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=3,test_size=0.2)
print(x_train)

# APPLYING FEATURE SCALING #

# Applying feature scaling to the matrix of features(x_train and x_test)
sc_feature_mat = StandardScaler()
x_train = sc_feature_mat.fit_transform(x_train)
x_test = sc_feature_mat.transform(x_test)


# Applying feature scaling to the dependent variable vector(y_train and y_test)
sc_dep_vec = StandardScaler()
y_train = sc_dep_vec.fit_transform(y_train)
y_test = sc_dep_vec.transform(y_test)

# Taking the values till 5 decimal places
np.set_printoptions(suppress=True, precision=5)


# ----------------------------------------- TRAINING THE MODEL ---------------------------------------------#

# Using Random Forest Regression for the data.

regressor = RandomForestRegressor(n_estimators=100, random_state=3)
regressor.fit(x_train,np.ravel(y_train))
y_pred = regressor.predict(x_test)


# -------------------------------------- PLOTTING THE GRAPH --------------------------------------------- #

# Plotting the graph between the predicted and the test values of the dataset.

np.set_printoptions(suppress=True, precision=5)
print(np.concatenate(((y_pred.reshape(len(y_pred),1)),(y_test.reshape(len(y_test),1))),1))
plt.scatter(y_pred, y_test, color='red')
plt.plot(y_test, y_test, color='blue')
plt.title("Graph")
plt.xlabel("Predicted value of price")
plt.ylabel("Original price of the model")
plt.show()

print(r2_score(y_test, y_pred))


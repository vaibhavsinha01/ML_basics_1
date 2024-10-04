import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import os

# Set working directory and path
wd = os.getcwd()
path = f'{wd}/archive/train.csv'
data = pd.read_csv(path)

X = data[['years', 'km', 'rating', 'condition', 'economy', 'top speed', 'hp', 'torque']]
y = data['current price']

# Split data into training and test sets
split = int(len(data) * 0.8)
x_train = X.iloc[:split]
x_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Initialize the StandardScaler for features
scaler = StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

model=RandomForestRegressor(n_estimators=300)

model.fit(x_train_scaled,y_train)

y_pred=model.predict(x_test_scaled)
print(y_pred)
print(f'y_pred: {y_pred[11]}')
print(f'y_test: {y_test.iloc[11]}')
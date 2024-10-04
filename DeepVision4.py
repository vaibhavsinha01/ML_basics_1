import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from sklearn.metrics import mean_absolute_error
import os

# Set working directory and path
wd = os.getcwd()
path = f'{wd}/archive/train.csv'
data = pd.read_csv(path)

# Use the function below to plot the pairplot (uncomment to visualize)
# sns.pairplot(data[['v.id','on road old','on road now','years','km','rating','condition','economy','top speed','hp','torque','current price']], diag_kind='kde')
# plt.show()

# Select features and target variable
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

# Fit the scaler to the training data and transform both training and test data
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Initialize the StandardScaler for the target variable
y_scaler = StandardScaler()

# Fit and transform the target variable
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

# Define the model
model = keras.models.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(x_train_scaled.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Print the model summary
print(model.summary())

# Fit the model to the scaled training data
model.fit(x_train_scaled, y_train_scaled, epochs=100, verbose=1)

# Predict the test set results using scaled test data
y_pred_scaled = model.predict(x_test_scaled)

# Inverse transform the predictions
y_pred = y_scaler.inverse_transform(y_pred_scaled)

# Print the first prediction and the corresponding actual value
print(f'Prediction: {y_pred[0][0]}')
print(f'Actual: {y_test.iloc[0]}')

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

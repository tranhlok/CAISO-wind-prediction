from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

file_path = '/Users/lok/5DE/processed_actual_only.csv'
data = pd.read_csv(file_path)

data['Datetime'] = pd.to_datetime(data['OPR_DT']) + pd.to_timedelta(data['OPR_HR'], unit='h')

# Assuming we are summing up the energy output for each hour across all trading hubs and renewable types
# Group by the new datetime column and sum the energy output
data_grouped = data.groupby('Datetime')['MW'].sum().reset_index()

# Sort the data chronologically
data_grouped.sort_values('Datetime', inplace=True)


# Number of hours to use as look-back for predictions
look_back = 24

# Scaling the 'MW' values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_grouped['MW'].values.reshape(-1,1))

# Function to create dataset for LSTM
def create_dataset(data, look_back):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

# Prepare the dataset
X, Y = create_dataset(scaled_data, look_back)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=False)

# Converting to PyTorch tensors
X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = torch.tensor(y_train).float()
y_test = torch.tensor(y_test).float()

# Reshaping input to be [samples, time steps, features] which is required for LSTM
X_train = X_train.view([X_train.shape[0], X_train.shape[1], 1])
X_test = X_test.view([X_test.shape[0], X_test.shape[1], 1])



import torch
import torch.nn as nn

# Define the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# Instantiate the model, define loss function and optimizer
model = LSTMModel()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

from torch.utils.data import TensorDataset, DataLoader

# Assuming X_train and y_train are already defined and are PyTorch tensors
train_dataset = TensorDataset(X_train, y_train)

# Define a DataLoader for batch processing
batch_size = 64  # You can adjust this based on your preference and GPU memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# Training the model
epochs = 150
for i in range(epochs):
    for seq, labels in train_dataset: # train_data to be created from X_train, y_train
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

import matplotlib.pyplot as plt

# Predicting from X_test
model.eval()
with torch.no_grad():
    predicted_test = model(X_test).view(-1)
predicted_test = scaler.inverse_transform(predicted_test.numpy().reshape(-1, 1))

# Inverse transform the actual values
actual_test = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

# Plotting
plt.figure(figsize=(12,6))
plt.plot(actual_test, label='Actual', color='blue')
plt.plot(predicted_test, label='Predicted', color='red')
plt.title('Energy Output: Predicted vs Actual')
plt.xlabel('Time (hours)')
plt.ylabel('Energy Output (MW)')
plt.legend()
plt.show()

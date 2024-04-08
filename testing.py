import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load the provided time series data
file_path = '/Users/lok/5DE/data/DAM_ACTUAL.csv'
data = pd.read_csv(file_path)

# Displaying the first few rows to understand the data structure
data.head()

# Separating actual and predicted data
actual_data = data[data['MARKET_RUN_ID'] == 'ACTUAL']
predicted_data = data[data['MARKET_RUN_ID'] == 'DAM']

# Renaming 'MW' columns for clarity
actual_data = actual_data.rename(columns={'MW': 'Actual_MW'})
predicted_data = predicted_data.rename(columns={'MW': 'Predicted_MW'})

# Merging the datasets on date and hour
merged_data = pd.merge(actual_data, predicted_data, on=['OPR_DT', 'OPR_HR', 'TRADING_HUB', 'RENEWABLE_TYPE'], how='inner')

# Display the merged dataset structure
merged_data.head()

relevant_data = merged_data[['Actual_MW', 'Predicted_MW']]

# Normalizing the data
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(relevant_data)

# Function to create sequences with both actual and predicted data
def create_sequences(input_data, sequence_length):
    xs = []
    ys = []
    for i in range(len(input_data)-sequence_length):
        x = input_data[i:(i+sequence_length), :]
        y = input_data[i+sequence_length, 0]  # We are predicting the next actual value
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Creating sequences
sequence_length = 24  # using 24 hours of data to predict the next hour
X, y = create_sequences(normalized_data, sequence_length)

# Checking the shape of the created sequences
X.shape, y.shape


train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:], y[train_size:]

# Converting data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)


class WindGenerationLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_layer_size=50, output_size=1):
        super(WindGenerationLSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.reshape(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


# Instantiating the model
model = WindGenerationLSTM()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # adjust step_size and gamma as needed

# Training the model
epochs = 50

train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()  # Set the model to training mode
    total_train_loss = 0

    # Training loop
    for seq, labels in zip(X_train_tensor, y_train_tensor):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)
        train_loss = loss_function(y_pred, labels)
        train_loss.backward()
        optimizer.step()

        total_train_loss += train_loss.item()

    avg_train_loss = total_train_loss / len(X_train_tensor)
    train_losses.append(avg_train_loss)
    # scheduler.step()

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        total_val_loss = 0
        for seq, labels in zip(X_val_tensor, y_val_tensor):
            y_pred = model(seq)
            val_loss = loss_function(y_pred, labels)
            total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(X_val_tensor)
        val_losses.append(avg_val_loss)


    print(f'Epoch {epoch+1} \t Training Loss: {avg_train_loss:.4f} \t Validation Loss: {avg_val_loss:.4f}')

# Plotting the training and validation loss graph
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

import torch
import numpy as np

# Assuming 'model' is your trained PyTorch LSTM model
# 'scaler' is your MinMaxScaler instance
# 'data' is your dataset DataFrame with a column 'MW' for wind generation
last_sequence = normalized_data[-24:]  # Last 24 hours of data

# Reshape to the format (batch_size, sequence_length, input_size)
input_tensor = torch.tensor(last_sequence.reshape(1, 24, 2), dtype=torch.float32)

predictions = []

model.eval()  # Set the model to evaluation mode

# Predicting the next 24 hours
for _ in range(24):
    with torch.no_grad():
        pred = model(input_tensor)
        predictions.append(pred.item())

        # Update the input with the new prediction as both actual and predicted value
        new_pred = np.array([[pred.item(), pred.item()]])
        new_sequence = np.concatenate((last_sequence[1:], new_pred), axis=0)
        input_tensor = torch.tensor(new_sequence.reshape(1, 24, 2), dtype=torch.float32)

# Inverse transform the predictions
predictions_scaled_back = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Output the predictions
print("Wind generation predictions for April 9, 2024:")
for hour, prediction in enumerate(predictions_scaled_back, start=1):
    print(f"Hour {hour}: {prediction[0]} MW")


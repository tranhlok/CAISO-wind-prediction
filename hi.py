import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load the provided time series data
file_path = '/Users/lok/5DE/processed_merged.csv'
data = pd.read_csv(file_path)

# Displaying the first few rows to understand the data structure
data.head()

# Convert 'Datetime' to datetime object and sort the data
data['Datetime'] = pd.to_datetime(data['Datetime'])
data.sort_values('Datetime', inplace=True)

# Using the last 7 days of data as validation set
validation_cutoff_date = data['Datetime'].max() - pd.Timedelta(days=7)

# Splitting the dataset into training and validation sets
train_data = data[data['Datetime'] < validation_cutoff_date]
validation_data = data[data['Datetime'] >= validation_cutoff_date]

# Normalizing the data
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data[['MW']].values)
validation_scaled = scaler.transform(validation_data[['MW']].values)

# Function to create sequences
def create_sequences(input_data, sequence_length):
    xs = []
    ys = []
    for i in range(len(input_data)-sequence_length):
        x = input_data[i:(i+sequence_length)]
        y = input_data[i+sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

sequence_length = 48  # using 24 hours of data to predict the next hour

# Creating sequences for training and validation data
X_train, y_train = create_sequences(train_scaled, sequence_length)
X_val, y_val = create_sequences(validation_scaled, sequence_length)

X_train.shape, y_train.shape, X_val.shape, y_val.shape



# Defining the LSTM model
class WindGenerationLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super(WindGenerationLSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# Instantiating the model
model = WindGenerationLSTM()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Converting data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

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

# Extracting the last 24 hours as the input sequence
last_sequence = data['MW'].values[-48:]

last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))

# Reshape to the format (sequence_length, batch_size, input_size)
input_tensor = torch.tensor(last_sequence_scaled.reshape(48, 1, 1), dtype=torch.float32)

predictions = []

model.eval()  # Set the model to evaluation mode

# Predicting the next 24 hours
for _ in range(48):
    with torch.no_grad():
        pred = model(input_tensor)
        predictions.append(pred.numpy().item())

        # Update the input for the next prediction
        # We remove the first time step and add the new prediction at the end
        input_tensor = torch.cat((input_tensor[0, 1:, :].unsqueeze(0), torch.tensor([[pred.item()]], dtype=torch.float32).view(1, 1, 1)), 1)

# Inverse transform the predictions if you used scaling
predictions_scaled_back = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Output the predictions
print("Wind generation predictions for the next 24 hours:")
for hour, prediction in enumerate(predictions_scaled_back, start=1):
    print(f"Hour {hour}: {prediction[0]} MW")


from data_processing import data_processing


from sklearn.preprocessing import MinMaxScaler
# LSTM模型定义
import torch.nn as nn
import torch

import numpy as np
import pandas as pd

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.5, noise_stddev=0.01):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.noise_stddev = noise_stddev
        
        # LSTM layer with dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Add Gaussian noise to input data
        x = x + torch.randn_like(x) * self.noise_stddev
        
        # LSTM forward
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # Apply dropout
        out = self.dropout(out)
        
        # Fully connected layer
        out = self.fc(out[:, -1, :])
        return out

# 训练函数
def train_lstm_model(X_train_tensor, y_train_tensor, input_size, output_size):
    hidden_size = 64
    num_layers = 2
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    num_epochs = 100
    for epoch in range(num_epochs):
        outputs = model(X_train_tensor)
        optimizer.zero_grad()
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return model

# 预测函数
def predict_lstm_model(model, X_test_tensor, scaler):
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
    predicted = scaler.inverse_transform(test_outputs)
    return predicted

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), :]
        dataX.append(a)
        dataY.append(dataset[i + time_step, :])
    return np.array(dataX), np.array(dataY)
# 获取处理过的数据
df = data_processing(2020,2023)

# Normalize the data for Blue Balls
blue_balls_np = np.array(df['Blue Balls'].tolist())
blue_scaler = MinMaxScaler(feature_range=(0, 1))
blue_balls_norm = blue_scaler.fit_transform(blue_balls_np)

# Normalize the data for Red Ball
red_ball_np = df['Red Ball'].values.reshape(-1, 1)
red_scaler = MinMaxScaler(feature_range=(0, 1))
red_ball_norm = red_scaler.fit_transform(red_ball_np)

# Create dataset
time_step = 10
X_blue, y_blue = create_dataset(blue_balls_norm, time_step)
X_red, y_red = create_dataset(red_ball_norm, time_step)

# Splitting data into training and testing
train_size = int(len(X_blue) * 0.67)
X_blue_train, X_blue_test = X_blue[0:train_size,:], X_blue[train_size:len(X_blue),:]
y_blue_train, y_blue_test = y_blue[0:train_size,:], y_blue[train_size:len(y_blue),:]
X_red_train, X_red_test = X_red[0:train_size,:], X_red[train_size:len(X_red),:]
y_red_train, y_red_test = y_red[0:train_size,:], y_red[train_size:len(y_red),:]


# 转为PyTorch张量
X_blue_train_tensor = torch.tensor(X_blue_train, dtype=torch.float32)
y_blue_train_tensor = torch.tensor(y_blue_train, dtype=torch.float32)
X_blue_test_tensor = torch.tensor(X_blue_test, dtype=torch.float32)

X_red_train_tensor = torch.tensor(X_red_train, dtype=torch.float32)
y_red_train_tensor = torch.tensor(y_red_train, dtype=torch.float32)
X_red_test_tensor = torch.tensor(X_red_test, dtype=torch.float32)

# 训练篮球模型
blue_model = train_lstm_model(X_blue_train_tensor, y_blue_train_tensor, input_size=5, output_size=5)

# 预测篮球号码
blue_predicted = predict_lstm_model(blue_model, X_blue_test_tensor, blue_scaler)
print("预测的篮球号码：")
print(blue_predicted)

# 训练红球模型
red_model = train_lstm_model(X_red_train_tensor, y_red_train_tensor, input_size=1, output_size=1)

# 预测红球号码
red_predicted = predict_lstm_model(red_model, X_red_test_tensor, red_scaler)
print("预测的红球号码：")
print(red_predicted)



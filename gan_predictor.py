import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from data_processing import data_processing

# 数据处理
def process_data():
    # 模拟数据
    df= data_processing()
    nums = [list(map(int, numset.split(','))) for numset in df['numberSet']]

    return torch.Tensor(nums)


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, scale_factor=[68, 68, 68, 68, 68, 25], offset=1, hidden_dim=50):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.scale_factor = torch.tensor(scale_factor).float()

        self.offset = offset
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # 增加一个序列长度的维度
        lstm_out, _ = self.lstm(x)
        return torch.sigmoid(self.linear(lstm_out[:,-1,:])) * self.scale_factor + self.offset

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=50):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # 增加一个序列长度的维度
        lstm_out, _ = self.lstm(x)
        return torch.sigmoid(self.linear(lstm_out[:,-1,:]))


# 训练 GAN
def train_gan(data, generator, discriminator, epochs=5000):
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.001)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.001)

    for epoch in range(epochs):
        # Train discriminator
        optimizer_d.zero_grad()
        real_data = data
        real_labels = torch.ones(real_data.size(0), 1)
        fake_data = generator(torch.randn(real_data.size(0), generator.input_dim))
        fake_labels = torch.zeros(fake_data.size(0), 1)
        logits_real = discriminator(real_data)
        logits_fake = discriminator(fake_data.detach())
        loss_real = criterion(logits_real, real_labels)
        loss_fake = criterion(logits_fake, fake_labels)
        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()

        # Train generator
        optimizer_g.zero_grad()
        logits_fake = discriminator(fake_data)
        loss_g = criterion(logits_fake, real_labels)
        loss_g.backward()
        optimizer_g.step()

    return generator


# 预测
def predict_next(generator, input_dim):
    with torch.no_grad():
        prediction = generator(torch.randn(1, input_dim))
        return prediction.round().int()



data = process_data()
generator = Generator(input_dim=10, output_dim=6, scale_factor=[68, 68, 68, 68, 68, 25])
discriminator = Discriminator(input_dim=6)
train_gan(data, generator, discriminator)

for i in range(5):
    next_numbers = predict_next(generator, 10)
    number_list = next_numbers.squeeze().tolist()

    print(f"numbers: {number_list}")





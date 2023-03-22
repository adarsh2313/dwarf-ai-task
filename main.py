import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import ast

# Reading the input file using pandas and extracting input features and height
df = pd.read_excel('/home/adarsh/projects/dwarf-ai-task/DS_Assignment 2/height_and_pose.xlsx')
heights = df['Height(cm)'].to_list()
in_features = df['Pose'].to_dict()
X_train = []
for i in range(len(in_features)):
    test = in_features[i]
    temp_dict = ast.literal_eval(test)
    store = []
    store.append(temp_dict[0]['key_points_coordinate'][0]['nose']['y'])
    store.append(temp_dict[0]['key_points_coordinate'][5]['left_shoulder']['y'])
    store.append(temp_dict[0]['key_points_coordinate'][6]['right_shoulder']['y'])
    store.append(temp_dict[0]['key_points_coordinate'][11]['left_hip']['y'])
    store.append(temp_dict[0]['key_points_coordinate'][12]['right_hip']['y'])
    store.append(temp_dict[0]['key_points_coordinate'][13]['left_knee']['y'])
    store.append(temp_dict[0]['key_points_coordinate'][14]['right_knee']['y'])
    store.append(temp_dict[0]['key_points_coordinate'][15]['left_ankle']['y'])
    store.append(temp_dict[0]['key_points_coordinate'][16]['right_ankle']['y'])
    X_train.append(store)
X_train = torch.Tensor(X_train)
y_train = torch.Tensor(heights)

# Creating a neural network with 3 layers (9 inputs, 5 hidden, 1 output)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        self.fc1 = nn.Linear(9, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5,1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

model = NeuralNetwork()

# Training the neural network
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.03)

batch_size = 10
dataset = TensorDataset(X_train, y_train)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

num_epochs = 250
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(data_loader):
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 50 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import timeit

table_size = 101 # How many samples are there in the look up table, the higher the number the more accurate it is but more memory.
width_of_network = 0.5 # This is a percentage of how much does the bell curve cover, so 1 would be 100%, 0.5 would be 50%, 2 would be 200%, etc.
pos_of_network = 0 # This is where on the network the bell curve starts as a percentage, so 0 is at the start and 1 is 100% of the way through the network.

minimum_value = 0 # When sampling the curve, what is the lowest value, a sort of bias term so it is never 0, if need be.
maximum_value = 5  # When sampling the curve, what is the value at the top of the curve.

movement_per_batch = 0.05 # How much the bell curve moves every batch, this is a percentage of the network, once it has reached the end it will go back to the start of the network

standard_deviation = 0.2
spread = 2 * standard_deviation ** 2

lookup_table = []

def calculate_lookup_table():
    x_range = np.linspace(-0.5, 0.5, table_size)
    return np.exp(-(x_range**2) / spread)

def bell_curve_sample_lookup(x):
    idx = int((x + 0.5) * (table_size - 1))
    idx = max(0, min(idx, table_size - 1))
    y = lookup_table[idx]
    return minimum_value + y * (maximum_value - minimum_value)

def bellcurve_sample_no_lookup(x):
    y = np.exp(-(x**2) / (2 * spread**2))
    return minimum_value + y * (maximum_value - minimum_value)

lookup_table = calculate_lookup_table()

class CustomModel(nn.Module):
    def __init__(self, sizes):
        super(CustomModel, self).__init__()
        self.layers = nn.Sequential(nn.Flatten())
        for i in range(len(sizes) - 1):
            linear = nn.Linear(sizes[i], sizes[i + 1])
            #nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
            #nn.init.constant_(linear.bias, 0)
            nn.init.xavier_normal_(linear.weight, gain=nn.init.calculate_gain('relu'))
            self.layers.append(linear)
            if i < len(sizes) - 1:
                self.layers.append(nn.LayerNorm(sizes[i + 1]))
            if i < len(sizes) - 2:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        return self.layers(x)

device = "cuda" if torch.cuda.is_available() else "cpu"

size_of_network = [784, 512, 256, 128, 64, 10]
#size_of_network = [784, 10]
feedforward_model = CustomModel(size_of_network).to(device)

params_for_bellcurve = []
for name, param in feedforward_model.named_parameters():
        if param.requires_grad:
            params_for_bellcurve.append(param)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(feedforward_model.parameters(), lr=0.1)
#optimizer = torch.optim.Adam(feedforward_model.parameters(), lr=0.01)

batch_size = 256
num_workers = 0

# Download and load the training data
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# Download and load the test data
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def train(model, train_loader, criterion, optimizer, device):
    global pos_of_network
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        for i, layer in enumerate(params_for_bellcurve):
            # print(layer.grad)
            # print(layer.shape)
            multiplier = bell_curve_sample_lookup((i / (len(params_for_bellcurve) - 1) - pos_of_network) / width_of_network)
            layer.grad.data.mul_(multiplier)
            # print(layer.grad)
            # print(multiplier)
            # input()
        
        pos_of_network += movement_per_batch
        if pos_of_network >= 1:
            pos_of_network = 0
        # pos_of_network = np.random.random()
        
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(train_loader), 100. * correct / total

def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(test_loader), 100. * correct / total

if __name__ == "__main__":

    epochs = 4

    print("Initial weights:", [feedforward_model.layers[1].weight[0][i].item() for i in range(5)])

    for epoch in range(epochs):
        train_loss, train_acc = train(feedforward_model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test(feedforward_model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print("-" * 50)
    
    print("Final weights:", [feedforward_model.layers[1].weight[0][i].item() for i in range(5)])
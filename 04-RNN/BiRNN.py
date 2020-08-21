import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 64
num_epochs = 2
learning_rate = 3e-3

# MNIST dataset
trainset = torchvision.datasets.MNIST(root='.data/', train=True,
                                      transform=transforms.ToTensor(),
                                      download=True)

testset = torchvision.datasets.MNIST(root='.data/', train=False,
                                     transform=transforms.ToTensor())

# Data loaders
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          num_workers=4,
                                          shuffle=True)

testloader = torch.utils.data.DataLoader(testset,
                                          batch_size=batch_size,
                                          num_workers=4,
                                          shuffle=False)

# RNN many-to-one
class BiRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
    
    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
model = BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(trainloader)
for e in range(num_epochs):
    for i, (images, labels) in enumerate(trainloader):
        
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1)%100 == 0:
            print(f'Epoch [{e}/{num_epochs}], Step [{i+1}/{total_step}] => Loss: {loss.item():.4f}')

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in testloader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print(f'Test accuracy: {100 * correct / total}%')

# Saving the model
torch.save(model.state_dict(), 'model.ckpt')
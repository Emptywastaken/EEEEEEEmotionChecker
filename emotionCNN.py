import torch.nn as nn
import torch.nn.functional as F


class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  
        self.pool1 = nn.MaxPool2d(2, 2)                          
    
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) 
        self.pool2 = nn.MaxPool2d(2, 2)                           
        self.dropout2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)                          
        

        self.flattened_size = 64 * 6 * 6
        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.dropout_fc = nn.Dropout(0.2)  # Standard dropout in fully connected

        self.output = nn.Linear(64, 7)  # 7 emotion classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
       

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = x.view(-1, self.flattened_size) 
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)

        x = self.output(x)
        return x

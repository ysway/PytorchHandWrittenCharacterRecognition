import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,      # input height
                out_channels=32,    # n_filters
                kernel_size=3,      # filter size
                stride=1,           # filter movement/step
                padding=1,      # keeping original height and width after con2d opreation, padding=(kernel_size-1)/2 when stride=1
            ),      # output shape (32, 28, 28)
            nn.ReLU(),    # activation
            nn.MaxPool2d(kernel_size=2),    # Maxpooling in 2x2 space, output shape (32, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (32, 14, 14)
            nn.Conv2d(32, 64, 3, 1, 1),  # output shape (64, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (64, 7, 7)
            nn.Dropout(0.25)
        )
        # fully connected layer, output 62 classes
        self.fc1 = nn.Linear(64 * 7 * 7, 8192)
        self.fc2 = nn.Linear(8192, 2048)
        self.fc3 = nn.Linear(2048, 512)
        self.fc4 = nn.Linear(512, 62)
        # Drop out method for fc
        self.dropout1 = nn.Dropout2d(0.75)
        self.dropout2 = nn.Dropout2d(0.5)
        self.dropout3 = nn.Dropout2d(0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 64 * 7 * 7)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        x = self.fc4(x)


        output = F.log_softmax(x, dim=1)
        return output
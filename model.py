import os
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PCamNet(nn.Module):
    def __init__(self):

        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 8, 3).to(memory_format=torch.channels_last)
        self.conv2 = nn.Conv2d(8, 16, 3).to(memory_format=torch.channels_last)
        self.conv3 = nn.Conv2d(16, 32, 3).to(memory_format=torch.channels_last)
        self.conv4 = nn.Conv2d(32, 64, 3).to(memory_format=torch.channels_last)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.sm = nn.Softmax(1)

        # Finds the place of interest in the image
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix.
        # Since the view is now coming in as 10*4*4, this was also adjusted 10*3*3 -> 10*4*4
        self.fc_loc = nn.Sequential(
            # This produces a transformation matrix (Affine matrix) that can be used to fit the region of interest (ROI)
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor(
            [1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        # Return transformed images
        return x

    def forward(self, x):

        # Get transformed images from stn
        # x = self.stn(x)

        # Transformed image is passed to normal network instead of original img
        # Perform the usual forward pass
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sm(self.fc3(x))
        return x


def GetNewModel():

    model = PCamNet()
    model = model.to(device)
    return model


def SaveModel(model, modelPath):
    torch.save(model.state_dict(), modelPath)


def LoadModel(modelPath):

    model = PCamNet()
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    return model.to(device)

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PCamNet(nn.Module):
    def __init__(self):

        super(PCamNet, self).__init__()
        self.forwardsCount = 0
        
        self.pretrained = torchvision.models.densenet201(pretrained=True)

        # Finds the place of interest in the image
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),

            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            
            nn.Conv2d(32, 64, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),

            nn.Conv2d(64, 64, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
        )

        # This produces a transformation matrix (Affine matrix) that can be used to fit the region of interest (ROI)
        self.fc_loc = nn.Sequential(
            nn.Linear(128 * 5 * 5, 32),
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
        xs = xs.view(-1, 128 * 5 * 5)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        # Return transformed images
        return x

    def forward(self, x):

        x = self.stn(x)
        x = self.pretrained(x)
        return x


def ShowImgs(img):
    
    if img.__len__ == None:
        plt.imshow(img.detach().clone().to("cpu").permute(1, 2, 0))
        plt.show()
        return

    numImgs = len(img)
    rows = cols = numImgs//2

    fig = plt.figure()
    for i in range(len(img)):

        fig.add_subplot(rows, cols, i+1)
        
        plt.imshow(img[i].detach().clone().to("cpu").permute(1, 2, 0))
        plt.axis('off')
        plt.title("Img "+str(i+1))

    plt.show()

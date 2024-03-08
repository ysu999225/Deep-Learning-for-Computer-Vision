import torch.nn as nn
class CNN(nn.Module):
    """
    A simple CNN for classifying images in the AnimalDataset.
    """

    def __init__(self):
        """
        Define the layers used in model.
        Hints:
        - Checkout `nn.Conv2d`, `nn.MaxPool2d`, `nn.Linear`, `nn.ReLU`
            in the PyTorch documentation.
        - You may use `nn.Sequential` to chain multiple layers together.
        - Be careful about the input and output shapes of the layers! Print `x.size()` if unsure.

        1. 1st CNN layer:
            - 2D Convolutional with input channels 3, output channels 8, kernel size 5, stride 1, and padding 2.
            - ReLU activation.
            - 2D Max pooling with kernel size 4 and stride 4.
        2. 2nd CNN layer:
            - 2D Convolutional with input channels 8, output channels 16, kernel size 5, stride 1, and padding 2.
            - ReLU activation.
            - 2D Max pooling with kernel size 4 and stride 4.
        3. 3rd CNN layer:
            - 2D Convolutional with input channels 16, output channels 32, kernel size 3, stride 1, and padding 1.
            - ReLU activation.
            - 2D Max pooling with kernel size 4 and stride 4.
        4. A flatten layer. The flattened feature should have shape (N, 32 * 4 * 4).
        5. A fully connected layer with 256 output units and ReLU activation.
        6. A fully connected layer with 10 output units.
        """
        super().__init__()
        #1st CNN layer:
        self.layer1 = nn.Sequential(
            #2D Convolutional with input channels 3, output channels 8, kernel size 5, stride 1, and padding 2.
            #Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
            nn.Conv2d(3,8,5,stride = 1, padding = 2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            #torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
            nn.MaxPool2d(2,stride = 2)
        )
        #2nd CNN layer:
        self.layer2 = nn.Sequential(
            nn.Conv2d(8,16,5,stride = 1, padding = 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2,stride = 2)
        )
        #3rd CNN layer:
        self.layer3 = nn.Sequential(
            nn.Conv2d(16,32,3,stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,stride = 2)
        )

        #Add 4th CNN layer:
        self.layer4 = nn.Sequential(
            nn.Conv2d(32,64,3,stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,stride = 2)
        )

        #Add 5th CNN layer:
        self.layer5 = nn.Sequential(
            nn.Conv2d(64,128,3,stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,stride = 2)
        )
        #Add 6th CNN layer:
        self.layer6 = nn.Sequential(
            nn.Conv2d(128,256,3,stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2,stride = 2)
        )





        #A flatten layer.
        #A flatten layer. The flattened feature should have shape (N, 32 * 4 * 4).
        #A fully connected layer with 256 output units and ReLU activation.
        self.flatten1 = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            #A fully connected layer with 10 output units.
            nn.Linear(256, 10)
        )



    def forward(self, x):
        """
        Forward pass of the model.
        Apply the layers defined in `__init__` in order.
        Args:
            x (Tensor): The input tensor of shape (N, 3, 256, 256).
        Returns:
            output (Tensor): The output tensor of shape (N, 10).
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)


#Be careful about the input and output shapes of the layers! Print `x.size()` if unsure.
#reshape the size here
        x = x.reshape(x.size(0),-1)

        x = self.flatten1(x)


        return x


# My experience in tunning the hp:
#1. Add more convolutional layers to the model or increase the number of hidden channels.
# Because Deeper networks can learn more complex features, it is a good way to improve the accuracy, but it also may overfit. So I considered to use Weight Decay to help prevent overfitting.
#2.Reduce the number of max-pooling layers. make a smaller kernel size and stride from 4 to 2; 
#3. add batch normalization layers to the model to improve the training process (the position before activation)
#4. Add a weight decay 1e-5 to the optimizer, which adds L2 regularization (prevent overfitting)
#5. decrase the gamma from 0.90 to 0.80,by reducing the learning rate more aggressively, the optimizer can converge to a minima more smoothly
#6. increase the epochs from 10 to 20, the model will have more opportunities to learn from the data
#Do not change the lr = 1e-3

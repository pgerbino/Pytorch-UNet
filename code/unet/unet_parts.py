import torch

""" Parts of the U-Net model """

import torch.nn as nn
import torch.nn.functional as F

# you may wonder why we need double convolution and not just one convolutional layer. The reason is that the double convolutional layer allows the network to learn more complex patterns in the input data.
# The first convolutional layer extracts low-level features from the input image, such as edges and corners. The second convolutional layer extracts higher-level features, such as shapes and textures.
# By using two consecutive convolutional layers, the network can learn more complex patterns in the input data and make better predictions.
# The double convolutional layer is a common building block in many modern neural networks because it is simple and effective at learning complex patterns in the input data.

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        DoubleConv module consists of two consecutive convolutional layers with batch normalization and ReLU activation.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            mid_channels (int, optional): Number of intermediate channels. If not provided, it is set to out_channels.
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            #conv2d can be explained to an undergraduate as a function that takes an image as input and outputs another image.
            #The convolutional layer is the building block of a CNN. It is a mathematical operation that takes an input image, applies a filter (or kernel) to it, and produces an output image.
            #The filter is a small matrix that slides over the input image, performing element-wise multiplication with the input values and summing the results to produce a single output value.
            #The filter is then moved to the next position, and the process is repeated until the entire image has been processed.
            #The output image is smaller than the input image because the filter cannot be applied to the edges of the input image.
            #The size of the output image is determined by the size of the filter and the stride, which is the number of pixels the filter moves each time it is applied.
            #The convolutional layer is used to extract features from the input image. It is typically followed by a pooling layer, which reduces the size of the output image by downsampling it.
            #The pooling layer is used to reduce the size of the output image and make the network more computationally efficient. It does this by taking a small region of the input image and computing a single output value for that region.
            #The pooling layer is typically followed by another convolutional layer, which extracts more features from the downsampled image. This process is repeated multiple times to create a deep network that can learn complex patterns in the input data.
            
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),  # 2D convolutional layer
            ##batchnorm2d can be explained to an undergraduate as a function that normalizes the input to a neural network by adjusting and scaling the activations.
            #It is used to make the training process more stable and faster by reducing the internal covariate shift, which is the change in the distribution of the input data to a layer of the network.
            #Batch normalization is typically applied after the activation function in a neural network. It takes the output of the activation function and normalizes it by subtracting the mean and dividing by the standard deviation of the input data.
            #This helps to ensure that the input data to the next layer of the network is in a consistent range, which can improve the performance of the network.
            #Batch normalization is used in many modern neural networks because it can help to prevent overfitting and improve the generalization of the model.

            nn.BatchNorm2d(mid_channels),  # Batch normalization layer
            #relu can be explained to an undergraduate as a function that introduces non-linearity into a neural network.
            #It is used to add non-linearities to the network, which allows it to learn complex patterns in the input data.
            #The ReLU function is a simple non-linear activation function that takes the input value and returns the maximum of that value and zero.
            #This means that if the input value is positive, the output will be the same as the input value. If the input value is negative, the output will be zero.
            #The ReLU function is used in many modern neural networks because it is simple and computationally efficient, and it has been shown to work well in practice.
            #It is typically applied after the convolutional and pooling layers in a neural network to introduce non-linearity into the network and allow it to learn complex patterns in the input data.
        
            nn.ReLU(inplace=True),  # ReLU activation function
            #kernel size means the size of the filter that is applied to the input image. The kernel size is typically a square matrix with an odd number of rows and columns, such as 3x3 or 5x5.
            #The kernel size determines the size of the region of the input image that is processed by the filter. A larger kernel size will capture more information from the input image, but it will also increase the computational cost of the operation.
            #The padding parameter is used to add zeros around the edges of the input image before applying the filter. This is done to ensure that the output image has the same size as the input image.
            #The padding parameter is typically set to half the kernel size, so that the filter is applied to the center of the input image.
            #The bias parameter is used to add a constant value to the output of the convolutional operation. It is typically set to False, because the batch normalization layer already includes a bias term.
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),  # 2D convolutional layer
            nn.BatchNorm2d(out_channels),  # Batch normalization layer
            nn.ReLU(inplace=True)  # ReLU activation function
        )

    def forward(self, x):
        """
        Forward pass of the DoubleConv module.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after passing through the double convolutional layers.
        """
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            #maxpool2d can be explained to an undergraduate as a function that reduces the size of an input image by taking the maximum value in a small region of the image.
            #It is used to downsample the input image and make the network more computationally efficient. It does this by taking a small region of the input image and computing the maximum value for that region.
            #The max pooling layer is typically applied after the convolutional layer in a neural network. It is used to reduce the size of the output image and make the network more computationally efficient.
            #The max pooling layer is typically followed by another convolutional layer, which extracts more features from the downsampled image. This process is repeated multiple times to create a deep network that can learn complex patterns in the input data.

            nn.MaxPool2d(2),  # Max pooling layer
            DoubleConv(in_channels, out_channels)  # DoubleConv layer
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            #upsample can be explained to an undergraduate as a function that increases the size of an input image by interpolating the pixel values.
            #It is used to upsample the input image and make the network more computationally efficient. It does this by increasing the size of the input image by a factor of two.
            #The upsample layer is typically used in the decoder part of a neural network to increase the size of the feature maps before concatenating them with the feature maps from the encoder part.
            #The upsample layer is typically followed by a convolutional layer, which extracts more features from the upsampled image. This process is repeated multiple times to create a deep network that can learn complex patterns in the input data.
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # Upsampling layer
            #even though we have already upsampled we use doubleconv to decode the features
            #The reason for this is that the upsample layer only increases the size of the input image, but it does not extract any new features from the image.
            #The convolutional layer is used to extract features from the input image. It takes the upsampled image as input and applies a filter to it to extract features such as edges, corners, and shapes.
            #The convolutional layer is typically followed by a batch normalization layer, which normalizes the input to the layer and makes the training process more stable and faster.            
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)  # DoubleConv layer
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)  # Transposed convolutional layer
            self.conv = DoubleConv(in_channels, out_channels)  # DoubleConv layer

    def forward(self, x1, x2):
        x1 = self.up(x1)  # Upsampling
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]  # Calculate the difference in height
        diffX = x2.size()[3] - x1.size()[3]  # Calculate the difference in width

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])  # Pad the input tensor
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)  # Concatenate the tensors along the channel dimension
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 2D convolutional layer

    def forward(self, x):
        return self.conv(x)

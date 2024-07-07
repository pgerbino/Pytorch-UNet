""" Full assembly of the parts to form the complete network """

from .unet_parts import *  # Importing necessary parts from the unet_parts module


# why does uunet work please see the following link
# https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47
# the U-Net architecture is composed of two parts: the encoder and the decoder
# the encoder is responsible for capturing the low level features of the image
# the decoder is responsible for upsampling the image back to its original resolution
# while reducing the number of channels
# the U-Net architecture is composed of a series of convolutional blocks
# each convolutional block is composed of two convolutional layers followed by a ReLU activation function
# the first convolutional layer has a kernel size of 3x3 and a padding of 1
# the second convolutional layer has a kernel size of 3x3 and a padding of 1
# the convolutional layers are followed by a ReLU activation function
# the output of the ReLU activation function is passed to the next convolutional block
# the U-Net architecture is composed of a series of downsample blocks
# each downsample block is composed of a max pooling layer followed by a convolutional block
# the max pooling layer has a kernel size of 2x2 and a stride of 2
# the convolutional block is composed of two convolutional layers followed by a ReLU activation function
# the output of the ReLU activation function is passed to the next downsample block
# the U-Net architecture is composed of a series of upsample blocks
# each upsample block is composed of an upsampling layer followed by a convolutional block
# the upsampling layer has a kernel size of 2x2 and a stride of 2
# the convolutional block is composed of two convolutional layers followed by a ReLU activation function
# the output of the ReLU activation function is passed to the next upsample block
# the U-Net architecture is composed of an output convolutional block
# the output convolutional block is composed of a convolutional layer followed by a ReLU activation function
# the output of the ReLU activation function is passed to the next convolutional layer
# the output of the last convolutional layer is the output of the U-Net architecture
# the output of the U-Net architecture is a segmentation mask
# the segmentation mask is a binary image that indicates the presence or absence of an object in the image
# the U-Net architecture is trained using a loss function called the dice loss
# the dice loss is a measure of the overlap between the predicted segmentation mask and the ground truth segmentation mask
# the dice loss is defined as the intersection over the union of the predicted segmentation mask and the ground truth segmentation mask
# the dice loss is minimized during training by adjusting the weights of the U-Net architecture
# the U-Net architecture is trained using a dataset of images and their corresponding segmentation masks
# the U-Net architecture is trained using a technique called backpropagation

# for a deeper mathemtical treatment of uunet please see the following link
# https://arxiv.org/abs/1505.04597


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels  # Number of input channels
        self.n_classes = n_classes  # Number of output classes
        self.bilinear = bilinear  # Flag to indicate whether to use bilinear upsampling

        self.inc = (DoubleConv(n_channels, 64))  # First convolutional block
        self.down1 = (Down(64, 128))  # Downsample block 1
        self.down2 = (Down(128, 256))  # Downsample block 2
        self.down3 = (Down(256, 512))  # Downsample block 3
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))  # Downsample block 4
        # at this point we have the original image resolution reduced by a factor of 16
        # and the number of channels increased to 1024
        # we can now start the upsampling process
        # what this represents is the decoder part of the U-Net architecture
        # the decoder part is responsible for upsampling the image back to its original resolution
        # while reducing the number of channels
        # so in a sense we have a low level representation of the original image 1024 times
        # and we want to upsample this representation to the original resolution
        # while reducing the number of channels to the number of classes
        # the number of classes is the number of output classes
        # in the case of the dataset we are using, the number of classes is 1
        self.up1 = (Up(1024, 512 // factor, bilinear))  # Upsample block 1
        self.up2 = (Up(512, 256 // factor, bilinear))  # Upsample block 2
        self.up3 = (Up(256, 128 // factor, bilinear))  # Upsample block 3
        self.up4 = (Up(128, 64, bilinear))  # Upsample block 4
        self.outc = (OutConv(64, n_classes))  # Output convolutional block

    def forward(self, x):
        x1 = self.inc(x)  # Pass input through the first convolutional block
        x2 = self.down1(x1)  # Pass through downsample block 1
        x3 = self.down2(x2)  # Pass through downsample block 2
        x4 = self.down3(x3)  # Pass through downsample block 3
        x5 = self.down4(x4)  # Pass through downsample block 4
        x = self.up1(x5, x4)  # Pass through upsample block 1
        x = self.up2(x, x3)  # Pass through upsample block 2
        x = self.up3(x, x2)  # Pass through upsample block 3
        x = self.up4(x, x1)  # Pass through upsample block 4
        logits = self.outc(x)  # Pass through the output convolutional block
        return logits

    def use_checkpointing(self):
        # Enable checkpointing for each block
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
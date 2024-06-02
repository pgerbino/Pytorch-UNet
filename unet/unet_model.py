""" Full assembly of the parts to form the complete network """

from .unet_parts import *  # Importing necessary parts from the unet_parts module


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